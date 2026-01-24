import argparse
import csv
import json
import os
import random
from datetime import date, datetime
from logging import INFO
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, Features, Value, Image as HFImage
from datasets.utils.logging import disable_progress_bar
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from pathlib import Path

import mak
from mak.servers.custom_server import ServerSaveData
from mak.servers.fedklsvd_server import FedKLSVDServer
from mak.servers.ffa_lora_server import FFALoRAServer
from mak.servers.fednova_server import FedNovaServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.servers.pfedmoap_server import PFedMoAPServer
from mak.servers.fedsa_lora_server import FedSALoRAServer
from mak.servers.flex_lora_server import FlexLoRAServer

from mak.strategies.fednova_strategy import FedNovaStrategy
from mak.strategies.scaffold_strategy import ScaffoldStrategy
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy
from mak.strategies.ffa_lora_strategy import FFALoRAStrategy
from mak.strategies.pfedmoap_strategy import PFedMoAPStrategy
from mak.strategies.fedsa_lora_strategy import FedSALoRAStrategy
from mak.strategies.flex_lora_strategy import FlexLoRAStrategy

from mak.utils.dataset_info import dataset_info
from mak.utils.general import set_params, test, weighted_average
from mak.models.svd_model import SVDAdapter, ConvAdapter
import math
from collections import Counter
import torch.nn.init as init
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import tempfile
import zipfile
import shutil

def get_target_keys(model, bias=True) -> List[str]:
    """Return deterministic sorted list of target parameter names for FFA/Flex LoRA.

    Includes:
    1) Adapter matrices (.A, .B)
    2) Biases after A and B, not all bias (.bias)

    This function is intentionally model-agnostic and must remain deterministic.
    """
    model_state = model.state_dict()

    # Generic adapter-based models (incl. CustomCLIP + SVDAdapter)
    # If adapters exist, prefer selecting adapter params directly (robust across architectures).
    a_keys = [k for k in model_state.keys() if k.endswith(".A")]
    b_keys = [k for k in model_state.keys() if k.endswith(".B")]
    if a_keys or b_keys:
        bases = {k[:-2] for k in (a_keys + b_keys)}  # strip ".A"/".B"
        bias_keys = [f"{base}.bias" for base in bases if f"{base}.bias" in model_state] if bias else []
        return sorted(set(a_keys + b_keys + bias_keys))

    if any(key.startswith("distilbert.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("lin" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("roberta.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self" in k or ("dense" in k and "classifier" not in k))]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("bert.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self" in k or "dense" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("model.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self_attn" in k or "mlp" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("layer") for key in model_state.keys()): #RESNET models
        if bias:
            lora_keys = [k for k in model_state.keys() if ("conv" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]

    # Unique + deterministic order
    return sorted(set(lora_keys))

def get_device_and_resources(config_sim):
    # Check if GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config_sim["client"]["gpu"] else "cpu"
    )
    # Assign GPU and CPU resources
    if device.type == "cuda":
        # Assign GPU resources
        num_gpus_total = config_sim["client"]["total_gpus"]
        if num_gpus_total > 0:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": num_gpus_total,
            }
        else:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": 0,
            }
    else:
        # Assign CPU resources
        ray_init_args = {"num_cpus": config_sim["client"]["total_cpus"], "num_gpus": 0}

    # Assign client resources
    client_res = {
        "num_cpus": config_sim["client"]["num_cpus"],
        "num_gpus": config_sim["client"]["num_gpus"] if device.type == "cuda" else 0.0,
    }
    if config_sim["common"]["multi_node"]:
        ray_init_args = {}
        ray_init_args["address"] = "auto"
        ray_init_args["runtime_env"] = {"py_modules": [mak]}
    return device, ray_init_args, client_res


def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    BASE_DIR = "output"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on strategy
    if not os.path.exists(
        os.path.join(BASE_DIR, str(today), config["server"]["strategy"])
    ):
        os.mkdir(os.path.join(BASE_DIR, str(today), config["server"]["strategy"]))

    # create saperate folder based on data distribution type
    if not os.path.exists(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    ):
        os.mkdir(
            os.path.join(
                BASE_DIR,
                str(today),
                config["server"]["strategy"],
                config["common"]["data_type"],
            )
        )

    dirs = os.listdir(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    )
    final_dir_path = os.path.join(
        BASE_DIR,
        str(today),
        config["server"]["strategy"],
        config["common"]["data_type"],
        str(len(dirs)),
    )

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)
    if not os.path.exists(os.path.join(final_dir_path, "clients")):
        os.mkdir(os.path.join(final_dir_path, "clients"))
    # models_dir = os.path.join(final_dir_path,'models')
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    # save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path, json_file_name), "w") as fp:
        json.dump(config, fp, indent=4)
    dataset_str = config["common"]["dataset"].replace("/", "_")
    file_name = f"{config['server']['strategy']}_{dataset_str}_{config['common']['data_type']}_{config['client']['batch_size']}_{config['client']['lr']}_{config['client']['epochs']}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, "w", encoding="UTF8") as f:
            # create the csv writer
            header = ["round", "global_accuracy", "global_f1_score", "global_loss", "local_accuracy", "local_f1", "local_loss", "processing_time", "upload_gb", "download_gb"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path, final_dir_path

def get_partitioner(config_sim):
    num_clients = config_sim["server"]["num_clients"]
    dataset_name = config_sim["common"]["dataset"]
    
    if config_sim["common"]["data_type"] == "dirichlet_niid":
        # alpha value
        dirichlet_alpha = config_sim["common"]["dirichlet_alpha"]
        # dataset's label column
        label = dataset_info[dataset_name]["output_column"]
        # create partitioner
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=label,
            alpha=dirichlet_alpha,
            min_partition_size=1,  # minimum number of samples in each partition
            self_balancing=False,
            shuffle=True,
            seed=config_sim["common"]["seed"],
        )
    else:
        partitioner = IidPartitioner(num_partitions=num_clients)
    # return train data
    return {"train": partitioner}

def load_upmc_food101_local(dataset_path: str):
    """
    Load UPMC-Food101 dataset from local files.
    
    Expected structure:
    dataset_path/
        images/
            train/
                apple_ipe/
                    *.jpg
                baby_back_ribs/
                    *.jpg
                ...
            test/
                apple_ipe/
                    *.jpg
                ...
        texts/
            train_titles.csv (columns: filename, title)
            test_titles.csv (columns: filename, title)
        train.csv (columns: filename, label or class_id)
        test.csv (columns: filename, label or class_id)
    
    Returns:
        train_dataset, test_dataset: HuggingFace Dataset objects with 'image', 'text', 'label' columns
    """
    dataset_path = Path(dataset_path)
    images_train_dir = dataset_path / "images" / "train"
    images_test_dir = dataset_path / "images" / "test"
    texts_train_file = dataset_path / "texts" / "train_titles.csv"
    texts_test_file = dataset_path / "texts" / "test_titles.csv"
    train_csv = dataset_path / "train.csv"
    test_csv = dataset_path / "test.csv"
    
    # Verify paths exist
    if not images_train_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {images_train_dir}")
    if not images_test_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_test_dir}")
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    # Load CSV files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_texts_df = pd.read_csv(texts_train_file) if texts_train_file.exists() else pd.DataFrame()
    test_texts_df = pd.read_csv(texts_test_file) if texts_test_file.exists() else pd.DataFrame()
    
    # Create text lookup dictionary (filename -> text)
    train_text_dict = {}
    if not train_texts_df.empty:
        # Assume first column is filename, second is text
        text_col = train_texts_df.columns[1] if len(train_texts_df.columns) > 1 else train_texts_df.columns[0]
        filename_col = train_texts_df.columns[0]
        train_text_dict = dict(zip(train_texts_df[filename_col], train_texts_df[text_col]))
    
    test_text_dict = {}
    if not test_texts_df.empty:
        text_col = test_texts_df.columns[1] if len(test_texts_df.columns) > 1 else test_texts_df.columns[0]
        filename_col = test_texts_df.columns[0]
        test_text_dict = dict(zip(test_texts_df[filename_col], test_texts_df[text_col]))
    
    train_data = []
    test_data = []
    
    # Process train data
    for idx, row in train_df.iterrows():
        # Get filename and label from CSV
        # Try common column names
        filename = row.get('filename', row.get('image', row.get('image_path', '')))
        if pd.isna(filename) or filename == '':
            continue
            
        label = row.get('label', row.get('class', row.get('class_id', row.get('class_name', 0))))
        
        # Get text from text dictionary
        text = train_text_dict.get(filename, train_text_dict.get(Path(filename).name, ""))
        
        # Find image file - check if filename includes class name or just filename
        filename_path = Path(filename)
        if filename_path.parent.name:  # Has directory in filename
            class_name = filename_path.parent.name
            img_filename = filename_path.name
        else:
            # Extract class name from filename (format: class_name_xxxxx.jpg)
            img_filename = filename_path.name
            class_name = img_filename.split('_')[0] if '_' in img_filename else filename_path.stem
        
        # Try to find image in class directory
        img_path = images_train_dir / class_name / img_filename
        if not img_path.exists():
            # Try direct filename match
            img_path = images_train_dir / img_filename
        if not img_path.exists():
            # Try searching in all class directories
            found = False
            for class_dir in images_train_dir.iterdir():
                if class_dir.is_dir():
                    potential_path = class_dir / img_filename
                    if potential_path.exists():
                        img_path = potential_path
                        found = True
                        break
            if not found:
                continue
        
        train_data.append({
            'image': Image.open(img_path).convert('RGB'),
            'text': str(text) if text else "",
            'label': int(label)
        })
    
    # Process test data
    for idx, row in test_df.iterrows():
        filename = row.get('filename', row.get('image', row.get('image_path', '')))
        if pd.isna(filename) or filename == '':
            continue
            
        label = row.get('label', row.get('class', row.get('class_id', row.get('class_name', 0))))
        text = test_text_dict.get(filename, test_text_dict.get(Path(filename).name, ""))
        
        filename_path = Path(filename)
        if filename_path.parent.name:
            class_name = filename_path.parent.name
            img_filename = filename_path.name
        else:
            img_filename = filename_path.name
            class_name = img_filename.split('_')[0] if '_' in img_filename else filename_path.stem
        
        img_path = images_test_dir / class_name / img_filename
        if not img_path.exists():
            img_path = images_test_dir / img_filename
        if not img_path.exists():
            found = False
            for class_dir in images_test_dir.iterdir():
                if class_dir.is_dir():
                    potential_path = class_dir / img_filename
                    if potential_path.exists():
                        img_path = potential_path
                        found = True
                        break
            if not found:
                continue
        
        test_data.append({
            'image': Image.open(img_path).convert('RGB'),
            'text': str(text) if text else "",
            'label': int(label)
        })
    
    # Create HuggingFace Dataset
    features = Features({
        'image': HFImage(),
        'text': Value('string'),
        'label': Value('int64')
    })
    
    train_dataset = Dataset.from_list(train_data, features=features)
    test_dataset = Dataset.from_list(test_data, features=features)
    
    log(INFO, f"Loaded {len(train_data)} train samples and {len(test_data)} test samples from local dataset")
    
    return train_dataset, test_dataset


def _find_upmc_root(extract_dir: Path) -> Path | None:
    """Find extracted UPMC-Food-101 root directory under extract_dir."""
    candidate = extract_dir / "UPMC-Food-101"
    if candidate.exists():
        return candidate
    # Fallback: search for a directory containing train.csv and images/
    for root, dirs, files in os.walk(extract_dir):
        root_p = Path(root)
        if (root_p / "train.csv").exists() and (root_p / "images").exists():
            return root_p
    return None


def _ensure_upmc_images_resized(
    upmc_root: Path,
    resize_to: int,
    jpeg_quality: int = 85,
) -> Path | None:
    """Create resized copy of UPMC images once, keep originals intact.

    Returns the resized images root (e.g., upmc_root/images_224) if created/existed,
    otherwise None.
    """
    if resize_to <= 0:
        return None

    src_images = upmc_root / "images"
    if not src_images.exists():
        return None

    dst_images = upmc_root / f"images_{resize_to}"
    done_flag = dst_images / ".done"
    if done_flag.exists():
        return dst_images

    # Create resized tree
    dst_images.mkdir(parents=True, exist_ok=True)
    # Build worklist first (so we can show an accurate progress bar)
    to_process: List[tuple[Path, Path]] = []
    seen = 0
    for split in ("train", "test"):
        split_dir = src_images / split
        if not split_dir.exists():
            continue
        for root, _, files in os.walk(split_dir):
            root_p = Path(root)
            rel = root_p.relative_to(src_images)
            out_dir = dst_images / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            for fn in files:
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                seen += 1
                src_path = root_p / fn
                dst_path = out_dir / Path(fn).with_suffix(".jpg").name
                if dst_path.exists():
                    continue
                to_process.append((src_path, dst_path))

    total = len(to_process)
    converted = 0
    failed = 0
    if total == 0:
        log(INFO, f"Resized images already exist at: {dst_images} (seen={seen})")
    else:
        from tqdm import tqdm

        log(INFO, f"Resizing {total} images to {resize_to}x{resize_to} (seen={seen}) ...")
        pbar = tqdm(
            to_process,
            desc=f"Resizing images_{resize_to}",
            unit="img",
            leave=True,
            mininterval=0.5,
        )
        converted = 0
        for src_path, dst_path in pbar:
            try:
                with Image.open(src_path) as im:
                    im = im.convert("RGB")
                    # Center-crop square then resize (cheap + deterministic)
                    im = ImageOps.fit(
                        im,
                        (resize_to, resize_to),
                        method=Image.BICUBIC,
                    )
                    im.save(
                        dst_path,
                        format="JPEG",
                        quality=int(jpeg_quality),
                        optimize=True,
                    )
                converted += 1
            except Exception as e:
                log(INFO, f"Failed to resize {src_path}: {e}")
                failed += 1
            # Keep postfix lightweight (avoid slowing down)
            if converted and converted % 500 == 0:
                pbar.set_postfix_str(f"ok={converted}/{total}")

    # Mark complete
    try:
        done_flag.write_text(
            f"resize_to={resize_to}\nquality={jpeg_quality}\nseen={seen}\nconverted={converted}\nfailed={failed}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    log(INFO, f"Resized images ready at: {dst_images} (to_process={total}, seen={seen})")
    return dst_images


def load_text_from_zip(repo_id: str, resize_images: bool = False, resize_to: int = 96):
    """
    Download zip file from HuggingFace Hub to project data directory, extract, and load text CSV files.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "kkim0451/UPMC-Food101")
    
    Returns:
        dict: {"train": list of texts, "test": list of texts} or None if failed
    """
    try:
        # Get project root directory (assuming helper.py is in mak/utils/)
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create dataset-specific directory
        dataset_name = repo_id.replace("/", "_")
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        extract_dir = dataset_dir / "extracted"
        zip_path = dataset_dir / "UPMC-Food-101.zip"
        
        # Check if zip file exists and is valid
        zip_exists_and_valid = False
        if zip_path.exists():
            try:
                # Try to open zip file to verify it's valid
                with zipfile.ZipFile(zip_path, 'r') as test_zip:
                    test_zip.testzip()  # Test zip file integrity
                zip_exists_and_valid = True
                log(INFO, f"Using existing valid zip file: {zip_path}")
            except (zipfile.BadZipFile, zipfile.LargeZipFile, Exception) as e:
                log(INFO, f"Existing zip file is invalid or corrupted: {e}. Will re-download.")
                zip_path.unlink()  # Remove corrupted zip file
        
        # Download zip file if not exists or invalid
        if not zip_exists_and_valid:
            log(INFO, f"Downloading zip file from HuggingFace Hub: {repo_id}")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename="UPMC-Food-101.zip",
                repo_type="dataset"
            )
            # Copy to project data directory
            shutil.copy2(downloaded_path, zip_path)
            log(INFO, f"Downloaded zip to: {zip_path}")
        
        # Check if extraction is needed
        # Verify extracted files exist and are complete
        texts_dir = extract_dir / "texts"
        train_csv = texts_dir / "train_titles.csv"
        test_csv = texts_dir / "test_titles.csv"
        
        # Check if texts directory exists, if not try to find it
        if not texts_dir.exists():
            for root, dirs, files in os.walk(extract_dir):
                if "texts" in dirs:
                    texts_dir = Path(root) / "texts"
                    train_csv = texts_dir / "train_titles.csv"
                    test_csv = texts_dir / "test_titles.csv"
                    break
        
        # Check if both CSV files exist
        extraction_needed = False
        if not extract_dir.exists():
            extraction_needed = True
            log(INFO, f"Extract directory does not exist: {extract_dir}")
        elif not texts_dir.exists():
            extraction_needed = True
            log(INFO, f"Texts directory does not exist: {texts_dir}")
        elif not train_csv.exists() or not test_csv.exists():
            extraction_needed = True
            log(INFO, f"CSV files incomplete. Train CSV exists: {train_csv.exists()}, Test CSV exists: {test_csv.exists()}")
        
        # Extract zip if needed
        if extraction_needed:
            log(INFO, f"Extracting zip to: {extract_dir}")
            extract_dir.mkdir(exist_ok=True)
            # Remove existing extraction if incomplete
            if extract_dir.exists() and (not train_csv.exists() or not test_csv.exists()):
                log(INFO, f"Removing incomplete extraction directory: {extract_dir}")
                shutil.rmtree(extract_dir)
                extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            log(INFO, f"Extracted zip to: {extract_dir}")
            
            # Re-find texts directory after extraction
            texts_dir = extract_dir / "texts"
            if not texts_dir.exists():
                for root, dirs, files in os.walk(extract_dir):
                    if "texts" in dirs:
                        texts_dir = Path(root) / "texts"
                        break
        else:
            log(INFO, f"Using existing extracted files in: {extract_dir}")
        
        # CSV files should already be located above, but ensure paths are correct
        train_csv = texts_dir / "train_titles.csv"
        test_csv = texts_dir / "test_titles.csv"
        
        result = {}

        # If extracted dataset root exists, optionally create resized images once
        upmc_root = _find_upmc_root(extract_dir)
        if upmc_root is not None:
            result["upmc_root"] = str(upmc_root)
            if resize_images:
                _ensure_upmc_images_resized(upmc_root=upmc_root, resize_to=int(resize_to))
        
        # Load train CSV
        if train_csv.exists():
            log(INFO, f"Loading train CSV from: {train_csv}")
            train_df = pd.read_csv(train_csv)
            # Find text column (usually 'title' or second column)
            text_col = None
            for col in train_df.columns:
                if 'title' in col.lower() or 'text' in col.lower():
                    text_col = col
                    break
            if text_col is None and len(train_df.columns) > 1:
                text_col = train_df.columns[1]  # Assume second column is text
            elif text_col is None:
                text_col = train_df.columns[0]  # Fallback to first column
            
            result["train"] = [str(row[text_col]).strip() for _, row in train_df.iterrows()]
            log(INFO, f"Loaded {len(result['train'])} train text entries")
        else:
            log(INFO, f"Train CSV not found at: {train_csv}")
            result["train"] = None
        
        # Load test CSV
        if test_csv.exists():
            log(INFO, f"Loading test CSV from: {test_csv}")
            test_df = pd.read_csv(test_csv)
            # Find text column
            text_col = None
            for col in test_df.columns:
                if 'title' in col.lower() or 'text' in col.lower():
                    text_col = col
                    break
            if text_col is None and len(test_df.columns) > 1:
                text_col = test_df.columns[1]
            elif text_col is None:
                text_col = test_df.columns[0]
            
            result["test"] = [str(row[text_col]).strip() for _, row in test_df.iterrows()]
            log(INFO, f"Loaded {len(result['test'])} test text entries")
        else:
            log(INFO, f"Test CSV not found at: {test_csv}")
            result["test"] = None
        
        return result if (result.get("train") or result.get("test")) else None
        
    except Exception as e:
        log(INFO, f"Failed to load text from zip: {e}")
        import traceback
        log(INFO, f"Traceback: {traceback.format_exc()}")
        return None


def add_text_to_dataset(dataset, text_data, split: str = "train"):
    """
    Add text field to HuggingFace dataset.
    
    Args:
        dataset: HuggingFace Dataset object
        text_data: Dictionary or list from load_text_from_hf_hub
        split: Dataset split name
    
    Returns:
        Dataset with 'text' field added
    """
    if text_data is None:
        # If no text data, add empty strings
        def add_empty_text(example, idx):
            return {"text": ""}
        return dataset.map(add_empty_text, with_indices=True)
    
    text_type = text_data.get("type")
    text_content = text_data.get("data")
    
    if text_type == "dict":
        # Map by filename or index
        def add_text_from_dict(example, idx):
            # Try to get filename from example
            filename = None
            
            # Check common filename fields
            for key in ["filename", "file_name", "image_path", "path", "image"]:
                if key in example:
                    value = example[key]
                    if isinstance(value, str):
                        filename = value
                        break
                    elif hasattr(value, "filename"):
                        filename = value.filename
                        break
                    elif isinstance(value, dict) and "path" in value:
                        filename = value["path"]
                        break
            
            text = ""
            if filename:
                # Try full filename, then just name, then path parts
                filename_str = str(filename)
                text = text_content.get(filename_str, "")
                if not text:
                    # Try with just the filename (without path)
                    filename_name = Path(filename_str).name
                    text = text_content.get(filename_name, "")
                if not text:
                    # Try with different path separators
                    for sep in ['/', '\\']:
                        if sep in filename_str:
                            parts = filename_str.split(sep)
                            if parts:
                                text = text_content.get(parts[-1], "")
                                if text:
                                    break
            
            # If still no text found and we have a list-like dict, try index
            if not text and idx < len(text_content):
                # Convert dict to list if possible (assuming ordered dict)
                text_list = list(text_content.values())
                if idx < len(text_list):
                    text = text_list[idx]
            
            return {"text": text if text else ""}
        
        return dataset.map(add_text_from_dict, with_indices=True)
    
    elif text_type == "list":
        # Map by index
        def add_text_from_list(example, idx):
            if idx < len(text_content):
                return {"text": text_content[idx]}
            else:
                return {"text": ""}
        
        return dataset.map(add_text_from_list, with_indices=True)
    
    else:
        # Fallback: empty text
        def add_empty_text(example, idx):
            return {"text": ""}
        return dataset.map(add_empty_text, with_indices=True)


class LocalFederatedDataset:
    """Minimal FederatedDataset-like wrapper for local datasets (train partitions + splits)."""

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, partitions: List[List[int]]):
        self._train = train_dataset
        self._test = test_dataset
        self._partitions = partitions

    def load_partition(self, partition_id: int) -> Dataset:
        return self._train.select(self._partitions[int(partition_id)])

    def load_split(self, split: str) -> Dataset:
        if split in ("test", "validation", "val"):
            return self._test
        if split == "train":
            return self._train
        raise ValueError(f"Unknown split: {split}")


def _iid_partitions(n: int, num_clients: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    chunks = np.array_split(idxs, num_clients)
    return [c.astype(int).tolist() for c in chunks]


def _dirichlet_partitions(labels: List[int], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    y = np.asarray(labels, dtype=np.int64)
    client_idxs: List[List[int]] = [[] for _ in range(num_clients)]

    classes = np.unique(y)
    for c in classes:
        cls_idxs = np.where(y == c)[0]
        if cls_idxs.size == 0:
            continue
        rng.shuffle(cls_idxs)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        counts = (props * cls_idxs.size).astype(int)
        # Fix rounding to match total
        while counts.sum() < cls_idxs.size:
            counts[rng.randint(num_clients)] += 1
        while counts.sum() > cls_idxs.size:
            j = rng.randint(num_clients)
            if counts[j] > 0:
                counts[j] -= 1
        splits = np.split(cls_idxs, np.cumsum(counts)[:-1])
        for cid, part in enumerate(splits):
            if part.size:
                client_idxs[cid].extend(part.astype(int).tolist())

    # Ensure no empty partitions (steal 1 sample from the largest partition)
    for cid in range(num_clients):
        if client_idxs[cid]:
            continue
        largest = max(range(num_clients), key=lambda k: len(client_idxs[k]))
        if client_idxs[largest]:
            client_idxs[cid].append(client_idxs[largest].pop())

    for cid in range(num_clients):
        rng.shuffle(client_idxs[cid])
        client_idxs[cid] = sorted(client_idxs[cid])

    return client_idxs


def get_dataset(config_sim):
    partitioner = get_partitioner(config_sim=config_sim)
    dataset_name = config_sim["common"]["dataset"]
    if dataset_name not in dataset_info.keys():
        raise Exception(f"Dataset name should be among : {list(dataset_info.keys())}")

    # --------------------------------------------
    # UPMC-Food101: prefer local extracted dataset
    # --------------------------------------------
    if dataset_name == "kkim0451/UPMC-Food101":
        resize_to = int(config_sim.get("common", {}).get("upmc_resize_to", 96))
        _ = load_text_from_zip(dataset_name, resize_images=True, resize_to=resize_to)

        project_root = Path(__file__).parent.parent.parent
        upmc_root = (
            project_root
            / "data"
            / dataset_name.replace("/", "_")
            / "extracted"
            / "UPMC-Food-101"
        )

        train_csv = upmc_root / "train.csv"
        test_csv = upmc_root / "test.csv"
        images_dir = upmc_root / f"images_{resize_to}"
        if not images_dir.exists():
            images_dir = upmc_root / "images"

        if train_csv.exists() and test_csv.exists() and images_dir.exists():
            log(INFO, f"Loading UPMC-Food101 from local files: {upmc_root}")
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)

            # Build image paths (keep as plain strings, decode lazily via HFImage)
            tr_paths = [
                str(images_dir / "train" / str(ann) / str(img_id))
                for ann, img_id in zip(train_df["annotation"], train_df["id"])
            ]
            te_paths = [
                str(images_dir / "test" / str(ann) / str(img_id))
                for ann, img_id in zip(test_df["annotation"], test_df["id"])
            ]

            # Create datasets (store paths; HFImage will decode lazily)
            features = Features(
                {"image": HFImage(), "text": Value("string"), "label": Value("int64")}
            )
            train_ds = Dataset.from_dict(
                {
                    "image": tr_paths,
                    "text": train_df["text"].fillna("").astype(str).tolist(),
                    "label": train_df["label"].astype(int).tolist(),
                },
                features=features,
            )
            test_ds = Dataset.from_dict(
                {
                    "image": te_paths,
                    "text": test_df["text"].fillna("").astype(str).tolist(),
                    "label": test_df["label"].astype(int).tolist(),
                },
                features=features,
            )

            # Partition train set for federated simulation
            num_clients = int(config_sim["server"]["num_clients"])
            if config_sim["common"]["data_type"] == "dirichlet_niid":
                alpha = float(config_sim["common"]["dirichlet_alpha"])
                parts = _dirichlet_partitions(
                    labels=train_ds[dataset_info[dataset_name]["output_column"]],
                    num_clients=num_clients,
                    alpha=alpha,
                    seed=int(config_sim["common"]["seed"]),
                )
            else:
                parts = _iid_partitions(
                    n=len(train_ds),
                    num_clients=num_clients,
                    seed=int(config_sim["common"]["seed"]),
                )
            fds = LocalFederatedDataset(train_dataset=train_ds, test_dataset=test_ds, partitions=parts)

            # Centralized test set (keep existing truncation behavior)
            centralized_testset = test_ds
            max_test_samples = 200
            n = min(max_test_samples, len(centralized_testset))
            centralized_testset = centralized_testset.select(
                range(len(centralized_testset) - n, len(centralized_testset))
            )
            log(INFO, f"UPMC-Food101 test set truncated to {n} samples")

            num_classes = dataset_info[dataset_name]["num_classes"]
            classnames = [f"class{i}" for i in range(num_classes)]
            return fds, centralized_testset, classnames

        log(INFO, f"Local UPMC-Food101 not ready at {upmc_root}, falling back to HuggingFace Hub")

    # Load from HuggingFace Hub (default)
    log(INFO, f"Loading dataset from HuggingFace Hub: {dataset_name}")
    fds = FederatedDataset(dataset=dataset_name, partitioners=partitioner)
    # get test column name
    test_set = dataset_info[dataset_name]["test_set"]
    if test_set is None:
        # If no test set, use train split and create validation split
        train_data = fds.load_split("train")
        # Split train into train/val (80/20)
        train_data = train_data.train_test_split(test_size=0.2, seed=config_sim["common"]["seed"])
        centralized_testset = train_data["test"]
    else:
        centralized_testset = fds.load_split(test_set)
    
    # For UPMC-Food101, download zip and load text from CSV
    if dataset_name == "kkim0451/UPMC-Food101":
        max_test_samples = 200
        n = min(max_test_samples, len(centralized_testset))
        centralized_testset = centralized_testset.select(range(len(centralized_testset) - n, len(centralized_testset)))
        log(INFO, f"UPMC-Food101 test set truncated to {n} samples")
        
        log(INFO, "Loading text data from zip file for UPMC-Food101")
        
        # Download zip and extract CSV files
        text_data = load_text_from_zip(dataset_name)
        
        if text_data:
            # Add text to test dataset
            if text_data.get("test"):
                def add_text_test(example, idx):
                    if idx < len(text_data["test"]):
                        return {"text": text_data["test"][idx]}
                    return {"text": ""}
                centralized_testset = centralized_testset.map(add_text_test, with_indices=True)
                log(INFO, f"Added {len(text_data['test'])} text entries to test dataset")
            else:
                def add_empty_text(example, idx):
                    return {"text": ""}
                centralized_testset = centralized_testset.map(add_empty_text, with_indices=True)
                log(INFO, "Added empty text field to test dataset")
            
            # Note: Cannot add text to train and recreate FederatedDataset(DatasetDict)
            # because flwr_datasets.FederatedDataset only supports dataset: str; it
            # calls datasets.load_dataset(path=...) and fails when path is DatasetDict.
            # Train partitions stay image-only; only test set has text for evaluation.
            if text_data.get("train"):
                log(INFO, "Train text data loaded from zip but not merged: FederatedDataset requires dataset name (str). Train partitions remain image-only.")
            else:
                log(INFO, "No train text data found")
        else:
            log(INFO, "Failed to load text from zip, adding empty text fields")
            def add_empty_text(example, idx):
                return {"text": ""}
            centralized_testset = centralized_testset.map(add_empty_text, with_indices=True)

    # get class names for pFedMoAP
    out_col = dataset_info[dataset_name]["output_column"]
    feat = centralized_testset.features.get(out_col, None)
    if feat is not None and hasattr(feat, "names") and feat.names:
        classnames = list(feat.names)
    else:
        num_classes = dataset_info[dataset_name]["num_classes"]
        classnames = [f"class{i}" for i in range(num_classes)]

    return fds, centralized_testset, classnames
    
def extract_linear_layers(model, config):
    """Return a dict of {layer_name: layer_module} for all linear layers in the model.
    Optionally skips layers specified in layers_to_skip.
    """
    linear_layers = {}
    skip_layer_names = ["pre_classifier", "classifier", "model.norm", "score", "classifier.dense", "classifier.out_proj"]
    attenion_layer_names = ["self_attn", "attn", "attention"]

    for name, module in model.named_modules():
        # Check if the module is a Linear layer
        if isinstance(module, torch.nn.Linear):
            if name in skip_layer_names: # Check if any part of the layer_to_skip is in the current layer's name
                continue
            if config["peft"]["layer"] == "attention_only":
                if any(att_name in name for att_name in attenion_layer_names):
                    linear_layers[name] = module
            else:
                linear_layers[name] = module

    return linear_layers

def extract_conv2_layers(model):
    conv2_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and (name.endswith("conv2") or name.endswith(".conv1")):
            conv2_layers[name] = module
    return conv2_layers

def apply_svd_to_model(model, config, kl_norm = None, client_id = None):
    """
    Apply SVD to the specified linear layers of the model, replacing them with SVDAdapter.
    The SVDAdapter class itself ensures W_res is frozen (as a buffer).
    Args:
        model (nn.Module): The model to modify.
        config (dict): Configuration dictionary, must contain config["peft"]["rank"],
                       config["peft"]["alpha"], and config["peft"]["method"].
        layers_to_skip_svd (list, optional): List of string name parts of layers
                                             to skip during SVD adaptation.
                                             E.g., ["classifier", "pre_classifier"].
    Returns:
        nn.Module: The modified model.
    """    
    if config["common"]["model"] in ["Resnet18", "ResNet18Pretrained", "ResNet34", "ResNet34Pretrained"]:
        layers_to_svd = extract_conv2_layers(model) 
        log(INFO, f"Found {len(layers_to_svd)} conv2 layers to adapt with SVD.")
    else:
        layers_to_svd = extract_linear_layers(model, config) 
        log(INFO, f"Found {len(layers_to_svd)} linear layers to adapt with SVD.")

    rank = config["peft"]["rank"]
    alpha = config["peft"]["alpha"]
    method = config["peft"]["method"]

    for name, layer in layers_to_svd.items():
        weight_matrix = layer.weight.data
        original_bias = layer.bias.data if layer.bias is not None else None

        if method == 'lora':
            # Original LoRA: Random initialization without SVD
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                A = torch.randn(c_out, rank, device=weight_matrix.device) * 0.01  # Gaussian init
                B = torch.zeros(rank, c_in * k1 * k2, device=weight_matrix.device)  # Zero init
                W_res = weight_matrix
            else:
                d_out, d_in = weight_matrix.shape
                A = torch.randn(d_out, rank, device=weight_matrix.device) * 0.01  # Gaussian init
                B = torch.zeros(rank, d_in, device=weight_matrix.device)  # Zero init
                W_res = weight_matrix
            log(INFO, f"Layer {name}: Applied LoRA with rank {rank}.")

        elif method == 'ffa_lora':
            # FFA-LoRA: Initialize A (configurable), B = 0, freeze A forever (external control)
            ffa_cfg = config.get("ffa_lora_config", {})
            init_method = ffa_cfg.get("init_method", "kaiming")

            if init_method not in {"kaiming", "gaussian", "orthogonal", "svd"}:
                raise ValueError(
                    f"Unknown init_method: {init_method}. Options: kaiming | gaussian | orthogonal | svd"
                )
            # --------------------------------------------------
            # Conv2d
            # --------------------------------------------------
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                d_in = c_in * k1 * k2
                W_flat = weight_matrix.view(c_out, -1)

                # A init
                if init_method =="svd":
                    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                    max_possible_rank = S.size(0)
                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]
                    
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [c_out, rank]
                    B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, d_in]
                    W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select).view(c_out, c_in, k1, k2)
                else:
                    A = torch.empty(c_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
            # --------------------------------------------------
            # Linear
            # --------------------------------------------------
            else:
                d_out, d_in = weight_matrix.shape

                # A init
                if init_method =="svd":
                    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
                    max_possible_rank = S.size(0)
                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]
                    
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [d_out, rank]
                    B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, d_in]
                    W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select)
                else:
                    A = torch.empty(d_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            log(INFO, f"Layer {name}: Applied FFA-LoRA with rank {rank} (A init={init_method}, B=zero, A frozen).")

        elif method == "fedsa_lora":
            # FedSA-LoRA: Train both A and B locally, but only A is aggregated (handled in client/strategy)
            # Init: A configurable (default kaiming), B = 0, W_res = W

            init_method = config["fedsa_lora_config"]["init_method"]
            
            if init_method not in {"kaiming", "gaussian", "orthogonal", "svd"}:
                raise ValueError(
                    f"Unknown init_method: {init_method}. Options: kaiming | gaussian | orthogonal | svd"
                )

            # --------------------------
            # Conv2d
            # --------------------------
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                d_in = c_in * k1 * k2
                W_flat = weight_matrix.view(c_out, -1)

                if init_method == "svd":
                    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                    max_possible_rank = S.size(0)

                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]

                    A = U_select @ torch.diag(torch.sqrt(S_select))  # [c_out, r]
                    # FedSA default: B zero, do not preload low-rank recon into adapter
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
                else:
                    A = torch.empty(c_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            # --------------------------
            # Linear
            # --------------------------
            else:
                d_out, d_in = weight_matrix.shape

                if init_method == "svd":
                    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
                    max_possible_rank = S.size(0)

                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    # A from SVD, B zero
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # [d_out, r]
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
                else:
                    A = torch.empty(d_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            log(INFO, f"Layer {name}: Applied FedSA-LoRA with rank {rank} (A init={init_method}, B=zero).")

        else: # Other SVD-based methods
            if isinstance(layer, torch.nn.Conv2d): #Conv2d layer SVD
                # weight_matrix = weight_matrix.view(weight_matrix.size(0), -1)  # Flatten Conv2d weights
                c_out, c_in, k1, k2 = weight_matrix.shape
                W_flat = weight_matrix.view(c_out, -1)  # Shape: [c_out, c_in * k1 * k2]

                # SVD decompistion
                U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                max_possible_rank = S.size(0)
                if rank > max_possible_rank:
                    log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                    rank = max_possible_rank
                
                # Select components based on method
                if method == 'pissa' or method == "flex_lora":
                    # Principal component as adapter (PiSSA)
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]

                elif method == 'milora':
                    # Minor component as adapter (MiLoRA)
                    U_select = U[:, -rank:]
                    S_select = S[-rank:]
                    Vt_select = Vt[-rank:, :]

                elif method == 'middle':
                    middle_index_start = math.floor(max_possible_rank/2)
                    middle_index_end = middle_index_start + rank
                    U_select = U[:, middle_index_start:middle_index_end]
                    S_select = S[middle_index_start:middle_index_end]
                    Vt_select = Vt[middle_index_start:middle_index_end, :]

                elif method == 'fedkls':
                    index_start = math.floor(kl_norm * (max_possible_rank - rank)) if kl_norm is not None else 0
                    index_end = index_start + rank
                    if client_id is not None:
                        log(INFO, f"Client {client_id}: SVD applied with index range {index_start} to {index_end} with rank {rank} for layer {name}.")
                    U_select = U[:, index_start:index_end]
                    S_select = S[index_start:index_end]
                    Vt_select = Vt[index_start:index_end, :]

                else:
                    raise ValueError(f"Unknown method: {method}")
            
                # Construct A and B matrices for Conv2d layer 
                W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select).view(c_out, c_in, k1, k2)
                A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [c_out, rank]
                B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, c_in * k1 * k2]

                # ----- Compute relative differences -----
                rel_recon_error = torch.norm(weight_matrix - (A @ B).view(c_out, c_in, k1, k2)) / torch.norm(weight_matrix)

            else: #Linear layer SVD
                U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False) 
                max_possible_rank = S.size(0)
                if rank > max_possible_rank:
                    log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                    rank = max_possible_rank

                # Select components based on method
                if method == 'pissa' or method == "flex_lora":
                    # Principal component as adapter (PiSSA)
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]

                elif method == 'milora':
                    # Minor component as adapter (MiLoRA)
                    U_select = U[:, -rank:]
                    S_select = S[-rank:]
                    Vt_select = Vt[-rank:, :]

                elif method == 'middle':
                    middle_index_start = math.floor(max_possible_rank/2)
                    middle_index_end = middle_index_start + rank
                    U_select = U[:, middle_index_start:middle_index_end]
                    S_select = S[middle_index_start:middle_index_end]
                    Vt_select = Vt[middle_index_start:middle_index_end, :]

                elif method == 'fedkls':
                    index_start = math.floor(kl_norm * (max_possible_rank - rank)) if kl_norm is not None else 0
                    index_end = index_start + rank
                    if client_id is not None:
                        log(INFO, f"Client {client_id}: SVD applied with index range {index_start} to {index_end} with rank {rank} for layer {name}.")
                    U_select = U[:, index_start:index_end]
                    S_select = S[index_start:index_end]
                    Vt_select = Vt[index_start:index_end, :]
                else:
                    raise ValueError(f"Unknown method: {method}")

                W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select)
                A = U_select @ torch.diag(torch.sqrt(S_select))
                B = torch.diag(torch.sqrt(S_select)) @ Vt_select

                rel_recon_error = torch.norm(weight_matrix - A @ B) / torch.norm(weight_matrix)
                log(INFO, f"Layer {name}: Relative reconstruction error (W vs ΔW): {rel_recon_error:.6f}")


            log(INFO, f"Layer {name}: Applied {method} with rank {rank}.")

        # Create appropriate adapter
        if isinstance(layer, torch.nn.Conv2d):
            new_layer = ConvAdapter(original_conv=layer, W_res=W_res, A=A, B=B, alpha=alpha, rank=rank)
        else:
            new_layer = SVDAdapter(W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, original_bias=original_bias)
        
        # Freeze A for FFA-LoRA (external control)
        if method == "ffa_lora":
            try:
                new_layer.A.requires_grad = False
            except Exception as e:
                log(INFO, f"Warning: Could not freeze A for layer {name}: {e}")

        # Split layer name and replace the original layer
        # Handle top-level modules (no '.' in name) like 'classifier', 'visual_projection'
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_layer)
        else:
            # Top-level module - set directly on model
            setattr(model, name, new_layer)  
    
    return model

def compute_KL_divergence(client_distributions: dict, num_classes: int) -> dict:
    # Step 1: Compute global/ideal IID label distribution
    global_prob = 1/num_classes

    # Step 2: Compute local distribution on each client
    local_prob = {}
    for client, dist in client_distributions.items():
        total_sample = sum([s for s in dist.values()])
        local_prob[client] = {cls: s/total_sample for cls, s in dist.items()}

    # Step 3: Compute KL divergence per client
    kl_values = {}
    for client, dist in local_prob.items():
        kl = 0
        for c in range(num_classes):
            p_ic = dist.get(c, 0) # P_i(c), 0 if class c not present
            q_c = global_prob     # Q(c), ideal IID distribution

            if p_ic > 0: # skip zero to avoid log(0)
                kl += p_ic * math.log(p_ic/q_c)
        kl_values[client] = kl

    # Step 4: Normalize using Min-Max
    kl_min = min(kl_values.values())
    kl_max = max(kl_values.values())
    # Prevent divide-by-zero in case all KLs are the same
    if kl_max == kl_min:
        kl_normalized = {client: 0.0 for client in kl_values}
    else:
        kl_normalized = {
            client: (kl - kl_min) / (kl_max - kl_min)
            for client, kl in kl_values.items()
        }

    return kl_normalized  

def compute_client_distributions(config, dataset, num_clients: int) -> dict:
    """
    Compute the label distribution for each client in the federated dataset.
    
    Args:
        dataset: FederatedDataset object (e.g., from flwr_datasets)
        num_clients: Number of clients in the federated dataset
    
    Returns:
        dict: Mapping of client IDs to their label distributions
    """
    client_distributions = {}
    dataset_name = config["common"]["dataset"]
    is_multi_label = dataset_info.get(dataset_name, {}).get("multi_label", False)
    output_column = dataset_info[dataset_name]["output_column"]
    
    log(INFO, "=>>>>> CLASS DISTRIBUTIONS OF ALL CLIENTS <<<<<<=")
    for cid in range(num_clients):
        # Load partition for all datasets using load_partition method
        client_data = dataset.load_partition(cid)
        
        # Optimized: Use direct column access instead of iterating through all items
        # This is much faster for large datasets (e.g., 405K samples)
        try:
            # Try direct column access (HuggingFace datasets support this)
            labels = client_data[output_column]
        except (TypeError, KeyError):
            # Fallback to iteration if direct access not supported
            labels = [item[output_column] for item in client_data]
        
        if is_multi_label:
            # For multi-label datasets, labels are lists - flatten and count individual labels
            flattened_labels = []
            for label_list in labels:
                if isinstance(label_list, list):
                    flattened_labels.extend(label_list)
                else:
                    flattened_labels.append(label_list)
            client_distributions[cid] = dict(sorted(Counter(flattened_labels).items()))
        else:
            # For single-label datasets, count labels directly
            client_distributions[cid] = dict(sorted(Counter(labels).items()))
        
        log(INFO, f"Client {cid} ({len(client_distributions[cid])} classes, {len(client_data)} samples) : {client_distributions[cid]}")
    log(INFO, f"Total samples from all clients: {sum([sum(dist.values()) for dist in client_distributions.values()])}")
    log(INFO, "*" * 150)
    
    return client_distributions

def get_model(config, shape, classnames=None):
    model_name = config["common"]["model"]
    # get num_classes
    dataset_name = config["common"]["dataset"]
    num_classes = dataset_info[dataset_name]["num_classes"]

    TEXT_ONLY_DATASETS = {"SetFit/20_newsgroups", "legacy-datasets/banking77", "fancyzhx/dbpedia_14"}
    # PFedMoAP CLIP guard
    if model_name == "Clip":
        if dataset_name in TEXT_ONLY_DATASETS:
            raise ValueError(f"PFedMoAP CLIP requires image dataset, got text dataset: {dataset_name}")

        pf = config["pfedmoap_config"]
        if classnames is None:
            classnames = [f"class{i}" for i in range(num_classes)]
        model = getattr(__import__("mak.models", fromlist=[model_name]), model_name)(
            num_classes=num_classes,
            input_shape=shape,
            backbone_name=pf.get("backbone_name", "ViT-B/32"),
            classnames=classnames,
            prompt_len=pf["prompt_len"],
            num_experts=pf.get("num_experts", 4),
            dgating=pf["dgating"],
            gating_heads=pf.get("heads", pf.get("gating_heads", 4)),
            lambda_local=pf.get("lambda_local", 1.0),
            freeze_text=True,
        )
        return model
    # check if model is from huggingface
    elif model_name in [
        "distilbert-base-uncased", 
        "bert-base-uncased", 
        "roberta-base",
        "roberta-large",
        "microsoft/deberta-v3-base",
        "Qwen/Qwen1.5-0.5B", 
        "meta-llama/Llama-2-7b-hf",
        "openai/clip-vit-base-patch32",
        ]:  # Add more as needed
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, CLIPModel
        if model_name == "Qwen/Qwen1.5-0.5B": #Need to check again when applying the quantization -> still error
            quantization_8_bit_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            quantization_4_bit_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Use 4-bit quantization
                    bnb_4bit_quant_type="nf4",  # Normal float 4-bit quantization
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Optimize compute dtype
                    bnb_4bit_use_double_quant=True,  # Enable double quantization
                )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                # quantization_config=quantization_8_bit_config,
                device_map="auto",
            )
            # Set pad_token_id to eos_token_id
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = base_model.config.eos_token_id
        elif model_name in ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]: #For multimodal datasets MM-IMDb and UPMC-Food101
            clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            class CustomCLIP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vision_model = clip_model.vision_model
                    self.text_model = clip_model.text_model
                    self.visual_projection = clip_model.visual_projection
                    self.text_projection = clip_model.text_projection
                    embed_dim = clip_model.config.projection_dim #512
                    self.classifier = torch.nn.Linear(embed_dim*2, num_classes) # Concatenate vision + text embeds
                def forward(self, pixel_values, input_ids, attention_mask):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                    vision_embeds = self.visual_projection(vision_outputs.pooler_output)
                    text_embeds = self.text_projection(text_outputs.pooler_output)
                    # Concatenate vision and text embeddings
                    combined = torch.cat([vision_embeds, text_embeds], dim=1)
                    logits = self.classifier(combined)
                    return logits
            base_model = CustomCLIP()
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, device_map="auto")

        return base_model

    # check custom models
    model = getattr(__import__("mak.models", fromlist=[model_name]), model_name)(
        num_classes=num_classes, input_shape=shape
    )

    return model

def get_evaluate_fn(
    centralized_testset: Dataset,
    config_sim,
    device,
    save_model_dir,
    metrics_file,
    apply_transforms_test,
    model,
    clip_collator=None,  # NEW: Use shared CLIPCollator from main.py
):
    """Return an evaluation function for centralized evaluation."""
    dataset_name = config_sim["common"]["dataset"]
    # shape = dataset_info[dataset_name]["input_shape"]

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):  
        
        strategy = config_sim.get("server", {}).get("strategy", "")
        method = config_sim.get("peft", {}).get("method", "")
        bias = config_sim.get("peft", {}).get("bias", "")

        # Only use PFedMoAP-specific logic if strategy is explicitly "PFedMoAP"
        if strategy == "PFedMoAP":
            if len(parameters) != 1:
                raise ValueError(f"PFedMoAP centralized eval expects 1 prompt, got {len(parameters)}")

            prompt = torch.from_numpy(np.asarray(parameters[0])).to(device=device)
            # Use model API directly
            if hasattr(model, "set_prompt"):
                model.set_prompt(prompt)
            if hasattr(model, "clear_nonlocal"):
                model.clear_nonlocal()
        elif strategy == "FlexLoRA" or method == "flex_lora":
            # Lazy import to avoid circular dependency
            from mak.utils.flex_lora_utils import load_server_eval_params_flex_lora
            load_server_eval_params_flex_lora(
                model=model,
                parameters=parameters,
                device=device,
            )
        else:
            set_params(model, parameters, method=method, bias=bias)

        model.to(device)

        # Handle multimodal datasets (no transform, use CLIPCollator)
        is_multimodal = dataset_name in ['pranavmr/MM-IMDb', 'kkim0451/UPMC-Food101']
        # NOTE: don't assign to `clip_collator` in this scope (would shadow outer var)
        local_clip_collator = clip_collator
        
        # Use shared clip_collator if provided, otherwise create new one (fallback)
        if local_clip_collator is None and is_multimodal and apply_transforms_test is None:
            # Fallback: Create CLIPCollator if not provided (should not happen in normal flow)
            from mak.utils.pytorch_transformations import CLIPCollator
            from transformers import CLIPProcessor
            
            model_name = config_sim["common"]["model"]
            clip_processor = CLIPProcessor.from_pretrained(model_name)
            output_column = dataset_info[dataset_name]["output_column"]
            is_multi_label = dataset_info[dataset_name].get("multi_label", False)
            num_classes = dataset_info[dataset_name]["num_classes"]
            
            # Get classnames if available
            label_to_idx = None
            try:
                if hasattr(centralized_testset, 'features') and 'label' in centralized_testset.features:
                    classnames = centralized_testset.features['label'].names
                    if classnames:
                        label_to_idx = {name: idx for idx, name in enumerate(classnames)}
            except:
                pass
            
            local_clip_collator = CLIPCollator(
                processor=clip_processor,
                label_key=output_column,
                multi_label=is_multi_label,
                num_classes=num_classes,
                label_to_idx=label_to_idx
            )
        
        if local_clip_collator is not None:
            # Keep dataset raw (no transform) for multimodal
            testset = centralized_testset
        else:
            # Apply transform to dataset for non-multimodal
            if apply_transforms_test is not None:
                testset = centralized_testset.with_transform(apply_transforms_test)
            else:
                testset = centralized_testset

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        # Create DataLoader with collator for multimodal, without for others
        if local_clip_collator is not None:
            testloader = DataLoader(
                testset,
                batch_size=config_sim["client"]["test_batch_size"],
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if device.type == 'cuda' else False,
                persistent_workers=True if min(4, os.cpu_count() or 1) > 0 else False,
                prefetch_factor=2,
                collate_fn=local_clip_collator,
            )
        else:
            testloader = DataLoader(testset, batch_size=config_sim["client"]["test_batch_size"])

        feature_key = dataset_info[dataset_name]["feature_key"]
        # Customize progress bar description based on server round
        if server_round == 0:
            desc = "Evaluating initial parameters"
        else:
            desc = f"Evaluating round {server_round}"
        loss, accuracy, f1 = test(model, testloader, device=device, feature_key=feature_key, dataset_name=dataset_name, desc=desc)
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df["global_loss"].min() > loss:
            log(
                INFO,
                f" =>>>>> Min Loss improved from {metrics_df['global_loss'].min()} to : {loss} =>>>>> Saving best model with accuracy : {accuracy}, f1_score : {f1}", 
            )
        #     torch.save(
        #         model.state_dict(), os.path.join(save_model_dir, "saved_best_model.pth")
        #     )

        # if server_round == config_sim["server"]["num_rounds"]:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(save_model_dir, "saved_final_model.pth"),
        #     )
        return loss, {"accuracy": accuracy, "f1_score": f1}
    return evaluate


def save_simulation_history(hist: fl.server.history.History, path):
    losses_distributed = hist.losses_distributed
    losses_centralized = hist.losses_centralized
    metrics_distributed_fit = hist.metrics_distributed_fit
    metrics_distributed = hist.metrics_distributed
    metrics_centralized = hist.metrics_centralized

    rounds = []
    losses_centralized_dict = {}
    losses_distributed_dict = {}
    accuracy_distributed_dict = {}
    accuracy_centralized_dict = {}
    f1_score_distributed_dict = {}
    f1_score_centralized_dict = {}

    for loss in losses_centralized:
        c_rnd = loss[0]
        rounds.append(c_rnd)
        losses_centralized_dict[c_rnd] = loss[1]

    for loss in losses_distributed:
        c_rnd = loss[0]
        losses_distributed_dict[c_rnd] = loss[1]
    if "accuracy" in metrics_distributed.keys():
        for acc in metrics_distributed["accuracy"]:
            c_rnd = acc[0]
            accuracy_distributed_dict[c_rnd] = acc[1]
    if "accuracy" in metrics_centralized.keys():
        for acc in metrics_centralized["accuracy"]:
            c_rnd = acc[0]
            accuracy_centralized_dict[c_rnd] = acc[1]
    if "f1_score" in metrics_distributed.keys():
        for f1 in metrics_distributed["f1_score"]:
            c_rnd = f1[0]
            f1_score_distributed_dict[c_rnd] = f1[1]
    if "f1_score" in metrics_centralized.keys():
        for f1 in metrics_centralized["f1_score"]:
            c_rnd = f1[0]
            f1_score_centralized_dict[c_rnd] = f1[1]

    if len(metrics_distributed_fit) != 0:
        pass  # TODO  check its implemetation later

    data = {
        "round": rounds,
        "global_loss": losses_centralized_dict,
        "local_loss": losses_distributed_dict,
        "global_accuracy": accuracy_centralized_dict,
        "local_accuracy": accuracy_distributed_dict,
        "global_f1_score": f1_score_centralized_dict,
        "local_f1_score": f1_score_distributed_dict,
    }

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each key in the data dictionary
    for key in data.keys():
        # If the key is 'rounds', set the 'rounds' column of the DataFrame to the rounds list
        if key == "round":
            df["round"] = data[key]
        # Otherwise, create a new column in the DataFrame with the key as the column name
        else:
            column_data = []
            # Iterate over each round in the 'rounds' list and add the corresponding value for the current key
            for round_num in data["round"]:
                # If the round number does not exist in the current key's dictionary, set the value to None
                if round_num not in data[key]:
                    column_data.append(None)
                else:
                    column_data.append(data[key][round_num])
            df[key] = column_data
    df.to_csv(os.path.join(path), index=False)


def get_server(strategy, client_manager, out_file_path, target_acc, num_train_thread, num_test_thread):
    if isinstance(strategy, ScaffoldStrategy):
        return ScaffoldServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FedNovaStrategy):
        return FedNovaServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FedKLSVDStrategy):
        return FedKLSVDServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FFALoRAStrategy):
        return FFALoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, PFedMoAPStrategy):
        return PFedMoAPServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedSALoRAStrategy):
        return FedSALoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FlexLoRAStrategy):
        return FlexLoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    else:           
        return ServerSaveData(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )


def get_strategy(
    config,
    test_data,
    save_model_dir,
    out_file_path,
    device,
    apply_transforms_test,
    size_weights,
    model,
    clip_collator=None,  # NEW: Use shared CLIPCollator from main.py
):
    STRATEGY = config["server"]["strategy"]
    MIN_CLIENTS_FIT = config["server"]["min_fit_clients"]
    MIN_CLIENTS_EVAL = config["server"]["min_evaluate_clients"]
    NUM_CLIENTS = config["server"]["num_clients"]
    FRACTION_FIT = config["server"]["fraction_fit"]
    FRACTION_EVAL = config["server"]["fraction_evaluate"]

    kwargs = {
        "FedAvgM": {
            "server_learning_rate": 1.0,
            "server_momentum": 0.2,
        },
        "FedAdam": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.9,
            "beta_2": 0.99,
            "tau": 1e-9,
        },
        "FedOpt": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.0,
            "beta_2": 0.0,
            "tau": 1e-9,
        },
        "FedProx": {
            "proximal_mu": config["fedprox"]["proximal_mu"],
        },
        "FedLaw": {
            "config": config,
            "model": model,
            "test_data": test_data,
            "size_weights": size_weights,
            "apply_transforms": apply_transforms_test,
            "apply_transforms_test": apply_transforms_test,
        },
        "FedAWA": {
            "config": config,
            "model": model,
            "test_data": test_data,
            "apply_transforms_test": apply_transforms_test,
        },
        "FFALoRA": {
            "config": config,
        },
        "PFedMoAP": {
            "config": config,
        },
        "FedSALoRA": {
            "config": config,
        },
        "FlexLoRA": {
            "config": config,
            "model": model,
            "global_rank": config.get("flex_lora_config", {}).get(
                "global_rank", config.get("peft", {}).get("rank", 32)
            ),
        },
        "PowD": {
            "candidate_client_set": config["powd_config"]["candidate_client_set"],
        },
    } 

    if STRATEGY == "PFedMoAP":
        prompt_len = config["pfedmoap_config"]["prompt_len"]
        
        prompt_dim = int(model.prompt_learner.ctx.shape[1])
        config["pfedmoap_config"]["prompt_dim"] = prompt_dim 

        # init global prompt
        prompt0 = (0.02 * np.random.randn(prompt_len, prompt_dim)).astype(np.float32)
        init_params = fl.common.ndarrays_to_parameters([prompt0])
    else:
        init_params = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model.state_dict().items()]
        )

    return getattr(__import__("mak.strategies", fromlist=[STRATEGY]), STRATEGY)(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_CLIENTS_FIT,
        min_evaluate_clients=MIN_CLIENTS_EVAL,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(
            centralized_testset=test_data,
            config_sim=config,
            save_model_dir=save_model_dir,
            metrics_file=out_file_path,
            device=device,
            apply_transforms_test=apply_transforms_test,
            model=model,
            clip_collator=clip_collator,  # NEW: Pass shared CLIPCollator
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_fit_config_fn(config_sim=config),
        on_evaluate_config_fn=get_evaluate_config_fn(config_sim=config),
        initial_parameters=init_params,
        **kwargs.get(STRATEGY, {}),
    )


def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    log(INFO, f"All random seeds set to {seed}")


def get_config(file_path):
    # Open the YAML file
    with open(file_path, "r") as file:
        # Parse the YAML data
        config = yaml.safe_load(file)
        return config


def get_fit_config_fn(config_sim):
    def fit_config(server_round: int):
        """Return training configuration dict for each round.
        passes the current round number to the client
        """
        config = {
            "round": server_round,
            "current_round": server_round, #Add current_round for dynamic data updates
            "batch_size": config_sim["client"]["batch_size"],
            "epochs": config_sim["client"]["epochs"],
            "lr": config_sim["client"]["lr"],
            "optimizer": config_sim["common"]["optimizer"],
            "sgd_momentum": config_sim["common"]["sgd_momentum"],
            "strategy": config_sim["server"]["strategy"],
            "proximal_mu": config_sim["fedprox"]["proximal_mu"],
            "loss": config_sim["client"]["loss"],
        }
        # P2 FIX: Add explicit payload kind for FlexLoRA
        if config["strategy"] == "FlexLoRA":
            if server_round == 1:
                config["payload_kind"] = "full"
            else:
                config["payload_kind"] = "partial"
        return config

    return fit_config

def get_evaluate_config_fn(config_sim):
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.
        
        passes the current round number to the client
        """
        config = {
            "round": server_round,
            "current_round": server_round,  # Add current_round for dynamic data updates
        }
        return config
    
    return evaluate_config

def get_mode_and_shape(partition):
    data_set_keys = list(partition.features.keys())
    x_column = data_set_keys[0]
    shape = partition[x_column][0].size
    mode = partition[x_column][0].mode
    if mode == "RGB":
        channel = 3
    else:
        channel = 1
    return (channel, shape[0], shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLNCLAB")

    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to the config.yaml file.",
    )
    parser.add_argument("--strategy", type=str, help="FL Strategy/algorithm")
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--noise", type=float, default=None, help="add dp noise to data or not"
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method for SVD adaptation (e.g., 'lora', 'pissa', 'milora', 'middle', 'fedkls')",
    )
    parser.add_argument(
        "--enabled",
        type=bool,
        default=None,
        help="Enable or disable the method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to be used for the simulation",
    )

    args = parser.parse_args()
    return args


def get_optimizer(model, client_config):
    if client_config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=client_config["lr"])
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=client_config["lr"],
            momentum=client_config["sgd_momentum"],
        )

# for fedlaw
def get_size_weights(federated_dataset, num_clients):
    sample_size = []
    for i in range(num_clients):
        sample_size.append(len(federated_dataset.load_partition(i)))
    size_weights = [i / sum(sample_size) for i in sample_size]
    return size_weights

