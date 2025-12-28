# mak/clients/pfedmoap_clip_client.py

from __future__ import annotations

from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from flwr.common import Scalar

from mak.clients.base_client import BaseClient


class PFedMoAPClient(BaseClient):
    """
    Client behavior:
      1) receives global prompt only
      2) loads nonlocal expert prompts from config (if provided)
      3) trains prompt and gating only
      4) returns updated local prompt only
    """
    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None, dataset=None, apply_transforms=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir, dataset=dataset, apply_transforms=apply_transforms
        )
    
    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        pf = self.config_sim.get("pfedmoap_config", {})
        prompt_lr = float(pf.get("prompt_lr", lr))
        gating_lr = float(pf.get("gating_lr", lr))
        wd = float(pf.get("weight_decay", 0.0))

        # Ensure only prompt and gating are trainable
        params_prompt = []
        params_gating = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "prompt_learner" in name or name.endswith("prompt") or "ctx" in name:
                params_prompt.append(p)
            else:
                params_gating.append(p)

        # If the model names differ, fallback: everything trainable goes into one group
        if (not params_prompt) and (not params_gating):
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        param_groups = []
        if params_prompt:
            param_groups.append({"params": params_prompt, "lr": prompt_lr, "weight_decay": wd})
        if params_gating:
            param_groups.append({"params": params_gating, "lr": gating_lr, "weight_decay": wd})

        return torch.optim.AdamW(param_groups)

    def get_parameters(self, config):
        # Only send prompt to server
        prompt = self.model.get_prompt()
        return [prompt.numpy()]

    def set_parameters(self, parameters):
        # Only set prompt from server
        if parameters is None or len(parameters) != 1:
            raise ValueError(f"PFedMoAP expects 1 tensor (prompt) from server, got {0 if parameters is None else len(parameters)}")
        prompt = torch.tensor(parameters[0], dtype=torch.float32, device=self.device)
        self.model.set_prompt(prompt)

    def fit(self, parameters, config: Dict) -> Tuple[list, int, Dict]:
        # 1) Set global prompt
        self.set_parameters(parameters)

        # 2) Load nonlocal experts from config
        has_experts = bool(config.get("pfedmoap_has_experts", False))
        if has_experts:
            expert_prompts = config.get("pfedmoap_expert_prompts", [])
            expert_tensors = [torch.tensor(p, dtype=torch.float32, device=self.device) for p in expert_prompts]
            self.model.load_nonlocal_prompts(expert_tensors)
        else:
            self.model.clear_nonlocal()

        # 3) Train
        batch_size = int(config["batch_size"])
        epochs = int(config["epochs"])
        lr = float(config["lr"])

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.optimizer = self._build_optimizer(lr=lr)
        scheduler = self.scheduler  # keep existing if your BaseClient uses it
        if scheduler is None:
            scheduler = None

        # Use BaseClient.train for image branch, it calls criterion(net(images), labels)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=self.optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
            scheduler=scheduler,
        )

        # 4) Return updated prompt only
        new_prompt = self.model.get_prompt()
        return [new_prompt.numpy()], len(trainloader.dataset), {}
    
    def train(
        self,
        net,
        trainloader,
        optim,
        epochs: int,
        device,
        config: Dict[str, Scalar],
        scheduler=None,
    ):
        net.train()

        criterion = self.get_loss(loss=config["loss"])

        feature_key = getattr(self, "feature_key", None)
        label_key = getattr(self, "output_column", None)

        if feature_key is None or label_key is None:
            raise ValueError("PFedMoAPClient requires self.feature_key and self.output_column to be set")

        for _ in range(int(epochs)):
            for batch in trainloader:
                if feature_key not in batch or label_key not in batch:
                    raise KeyError(
                        f"Batch missing keys. Need ({feature_key}, {label_key}), got {list(batch.keys())}"
                    )

                images = batch[feature_key].to(device)
                labels = batch[label_key].to(device)

                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=False):
                    logits = net(images.float())
                
                if not torch.isfinite(logits).all():
                    print("[NaN/Inf] logits", torch.isnan(logits).any().item(), torch.isinf(logits).any().item())
                    print("logits min/max", logits.nan_to_num().min().item(), logits.nan_to_num().max().item())
                    # optional: stop early to avoid contaminating optimizer state
                    raise RuntimeError("Non-finite logits detected")
                loss = criterion(logits, labels)
                loss.backward()
                optim.step()

            if scheduler is not None:
                scheduler.step()
