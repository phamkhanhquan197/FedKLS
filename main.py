import flwr as fl
from logging import INFO
from flwr.common.logger import log
from datasets.utils.logging import disable_progress_bar
import os
from mak.utils.helper import get_device_and_resources
from mak.utils.helper import (
    gen_dir_outfile_server,
    get_model,
    get_strategy,
    get_server,
    save_simulation_history,
    get_dataset,
    get_size_weights,
)
from mak.utils.pytorch_transformations import (
    TransformationPipeline,
    TextTransformationPipeline,
    CLIPTransformationPipeline,
)
from mak.clients import get_client_fn
from mak.utils.dataset_info import dataset_info
from mak.utils.helper import get_config, set_seed, parse_args, apply_svd_to_model
from mak.utils.helper import compute_client_distributions, compute_KL_divergence
from mak.utils.flex_lora_utils import generate_rank_map, setup_server_config
import copy
import gc
import torch
import numpy as np


def main():
    # Parse arguments and configs
    args = parse_args()
    config_sim = get_config(args.config)
    seed = args.seed if args.seed else config_sim.get("common", {}).get("seed", 42)
    strategy_name = (
        args.strategy if args.strategy else config_sim.get("server", {}).get("strategy", "")
    )
    dirichlet_alpha = (
        args.dirichlet_alpha
        if args.dirichlet_alpha
        else config_sim.get("common", {}).get("dirichlet_alpha", "")
    )
    dataset_name = (
        args.dataset
        if args.dataset
        else config_sim.get("common", {}).get("dataset", "SetFit/20_newsgroups")
    )
    method = args.method if args.method else config_sim.get("peft", {}).get("method", "lora")
    enabled = (
        args.enabled if args.enabled else config_sim.get("peft", {}).get("enabled", False)
    )
    lr = args.lr if args.lr else config_sim.get("client", {}).get("lr", 0.001)

    config_sim["common"]["seed"] = seed
    config_sim["server"]["strategy"] = strategy_name
    config_sim["common"]["dirichlet_alpha"] = dirichlet_alpha
    config_sim["common"]["dataset"] = dataset_name
    config_sim["peft"]["method"] = method
    config_sim["peft"]["enabled"] = enabled
    config_sim["client"]["lr"] = lr

    set_seed(seed=config_sim["common"]["seed"])

    fds, centralized_testset, classnames = get_dataset(config_sim=config_sim)

    if config_sim["server"]["strategy"] == "FedLaw":
        size_weights = get_size_weights(
            federated_dataset=fds, num_clients=config_sim["server"]["num_clients"]
        )
    else:
        size_weights = []

    model_name = config_sim["common"]["model"]
    shape = dataset_info[dataset_name]["input_shape"]

    if model_name == "clip" or config_sim["server"]["strategy"] == "PFedMoAP":
        transformation_pipeline = CLIPTransformationPipeline(
            dataset_name=dataset_name, img_size=224
        )

    if dataset_name in [
        "SetFit/20_newsgroups",
        "legacy-datasets/banking77",
        "fancyzhx/dbpedia_14",
    ] or model_name in [
        "distilbert-base-uncased",
        "microsoft/deberta-v3-base",
        "llama2-7b",
    ]:
        transformation_pipeline = TextTransformationPipeline(
            dataset_name=dataset_name, model_name=model_name
        )
    else:
        transformation_pipeline = TransformationPipeline(dataset_name=dataset_name)

    apply_transforms, apply_transforms_test = transformation_pipeline.get_transformations()

    device, ray_init_args, client_res = get_device_and_resources(config_sim=config_sim)
    out_file_path, saved_models_path = gen_dir_outfile_server(config=config_sim)

    if config_sim["common"]["save_log"]:
        fl.common.logger.configure(
            identifier="FLNCLAB", filename=os.path.join(saved_models_path, "log.txt")
        )

    # Base model
    base_model = get_model(config_sim, shape=shape, classnames=classnames)
    base_model = base_model.cpu()

    log(INFO, f"Base model structure: {base_model}")
    for name, tensor in base_model.state_dict().items():
        log(INFO, f"{name}: shape {tuple(tensor.shape)}")
    log(INFO, f"=>>>>>>>>>>>>>>>>>Number of layers: {len(base_model.state_dict())}")

    log(INFO, "*" * 75)

    lora_enabled = config_sim["peft"]["enabled"]
    peft_method = config_sim["peft"]["method"]

    # FlexLoRA: generate rank map once and store in config
    if config_sim.get("server", {}).get("strategy") == "FlexLoRA":
        num_clients = int(config_sim["server"]["num_clients"])
        rank_map = generate_rank_map(config_sim, num_clients=num_clients)
        config_sim.setdefault("flex_lora_config", {})["client_rank_map"] = rank_map
        log(INFO, f"FlexLoRA rank_map generated: {rank_map}")

    if lora_enabled:
        if peft_method == "fedkls":
            client_distributions = compute_client_distributions(
                config=config_sim, dataset=fds, num_clients=config_sim["server"]["num_clients"]
            )
            kl_normalized_per_client = compute_KL_divergence(
                client_distributions, num_classes=dataset_info[dataset_name]["num_classes"]
            )

            client_models = {}
            os.makedirs(os.path.join(saved_models_path, "client_models"), exist_ok=True)
            for cid in range(config_sim["server"]["num_clients"]):
                kl_norm = kl_normalized_per_client[cid]
                client_model = copy.deepcopy(base_model)
                client_model = apply_svd_to_model(
                    model=client_model,
                    config=config_sim,
                    kl_norm=kl_norm,
                    client_id=cid,
                )
                model_path = os.path.join(
                    saved_models_path, "client_models", f"client_{cid}_model.pt"
                )
                torch.save(client_model, model_path)
                client_models[cid] = model_path
                del client_model
                gc.collect()

            model_for_svd = copy.deepcopy(base_model)
            svd_model = apply_svd_to_model(
                model=model_for_svd,
                config=config_sim,
                kl_norm=sum(kl_normalized_per_client.values()) / len(kl_normalized_per_client),
            )
            server_model = svd_model
            client_model = svd_model

        elif peft_method in ["pissa", "milora", "middle", "lora", "ffa_lora", "flex_lora"]:
            model_for_svd = copy.deepcopy(base_model)

            if peft_method == "flex_lora":
                flex_cfg = config_sim.get("flex_lora_config", {})
                global_rank = int(
                    flex_cfg.get("global_rank", config_sim.get("peft", {}).get("rank", 32))
                )
                server_cfg = setup_server_config(config_sim, global_rank=global_rank)
                svd_model = apply_svd_to_model(model=model_for_svd, config=server_cfg)
            else:
                svd_model = apply_svd_to_model(model=model_for_svd, config=config_sim)

            server_model = svd_model
            client_model = svd_model

            _ = compute_client_distributions(
                config=config_sim, dataset=fds, num_clients=config_sim["server"]["num_clients"]
            )
        else:
            import sys

            sys.exit(0)
    else:
        _ = compute_client_distributions(
            config=config_sim, dataset=fds, num_clients=config_sim["server"]["num_clients"]
        )
        server_model = base_model
        client_model = base_model

    try:
        dir_alpha = fds._partitioners["train"]._alpha[0]
    except (AttributeError):
        dir_alpha = "NA"

    log(INFO, f" =>>>>> Dataset : {dataset_name}")
    log(INFO, f" =>>>>> Model : {base_model._get_name()} Device : {device}")
    log(INFO, f" =>>>>> Partitoner : {config_sim['common']['data_type']} Alpha : {dir_alpha}")
    log(INFO, f" =>>>>> Ray init args : {ray_init_args} Client Res : {client_res}")

    data_scheduler = None
    dyn_cfg = config_sim.get("dynamic_data", {})
    if dyn_cfg.get("enabled", False):
        from mak.utils.dynamic_data import DynamicDataScheduler

        data_scheduler = DynamicDataScheduler(
            federated_dataset=fds,
            num_clients=config_sim["server"]["num_clients"],
            total_rounds=config_sim["server"]["num_rounds"],
            seed=config_sim["common"]["seed"],
            config_sim=config_sim,
            mode=dyn_cfg.get("mode", "incremental"),
            round_step=dyn_cfg.get("round_step", 10),
            start_fraction=dyn_cfg.get("start_fraction", 0.3),
        )

    strategy = get_strategy(
        config=config_sim,
        test_data=centralized_testset,
        save_model_dir=saved_models_path,
        out_file_path=out_file_path,
        device=device,
        apply_transforms_test=apply_transforms_test,
        size_weights=size_weights,
        model=server_model,
    )

    server = get_server(
        strategy=strategy,
        client_manager=fl.server.client_manager.SimpleClientManager(),
        out_file_path=out_file_path,
        target_acc=config_sim["common"]["target_acc"],
        num_train_thread=config_sim["common"]["num_train_thread"],
        num_test_thread=config_sim["common"]["num_test_thread"],
    )

    log(INFO, f" =>>>>> Using Strategy : {strategy.__class__} Server : {server.__class__}")

    def client_fn_with_models(cid):
        cid = int(cid)

        if peft_method == "fedkls":
            model_path = client_models[cid]
            model = torch.load(model_path, map_location=device, weights_only=False)
            model = model.to(device)
        else:
            model = client_model

        rank_map = config_sim.get("flex_lora_config", {}).get("client_rank_map", None)

        return get_client_fn(
            config_sim=config_sim,
            dataset=fds,
            model=model,
            device=device,
            apply_transforms=apply_transforms,
            save_dir=saved_models_path,
            kl_norm_dict=kl_normalized_per_client if peft_method == "fedkls" else None,
            data_scheduler=data_scheduler,
            rank_map=rank_map,
        )(cid)

    hist = fl.simulation.start_simulation(
        client_fn=client_fn_with_models,
        num_clients=config_sim["server"]["num_clients"],
        client_resources=client_res,
        config=fl.server.ServerConfig(num_rounds=config_sim["server"]["num_rounds"]),
        strategy=strategy,
        server=server,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
        ray_init_args=ray_init_args,
    )

    simu_data_file_path = out_file_path.replace(".csv", "_metrics.csv")
    save_simulation_history(hist=hist, path=simu_data_file_path)


if __name__ == "__main__":
    main()
