from __future__ import annotations

from typing import Dict, Tuple

import flwr as fl

from mak.clients.base_client import BaseClient


class FedPOEClient(BaseClient):
    """Fed-POE client (Flower port, Hedge mixture signals).

    This client behaves like the standard training client (inherits BaseClient),
    but during evaluation it reports *two* losses:

    - loss_fed: loss when evaluating the current *server/global* parameters
    - loss_loc: loss when evaluating the client's *local/personalized* parameters

    The server/strategy can then update per-client weights (a,b) similar to the
    3rd-party Fed-POE code (cifar10.py):
        a <- a * exp(-eta * loss_fed)
        b <- b * exp(-eta * loss_loc)
    """

    def __repr__(self) -> str:
        return " FedPOE client"

    def __init__(
        self,
        client_id,
        model,
        trainset,
        valset,
        config_sim,
        device,
        save_dir,
        kl_norm=None,
        dataset=None,
        apply_transforms=None,
        data_scheduler=None,
        bias=None,
    ):
        super().__init__(
            client_id,
            model,
            trainset,
            valset,
            config_sim,
            device,
            save_dir,
            dataset=dataset,
            apply_transforms=apply_transforms,
            data_scheduler=data_scheduler,
            bias=bias,
        )

        # Keep a copy of the *local* parameters after fit.
        self._local_params = None

    def fit(self, parameters, config):
        params_to_send, num_examples, metrics = super().fit(parameters, config)
        # Store local parameters (numpy ndarrays) for Fed-POE local evaluation.
        self._local_params = params_to_send
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate both fed and local models.

        Flower uses the returned (loss, num_examples, metrics) for aggregation.
        We return `loss_fed` as the primary loss (so standard aggregation still
        makes sense) and include both losses in metrics.
        """

        # 1) Evaluate fed/global parameters (the ones provided by server)
        loss_fed, num_examples, metrics = super().evaluate(parameters, config)

        # 2) Evaluate local/personalized parameters (stored after last fit)
        loss_loc = loss_fed
        try:
            if self._local_params is not None:
                loss_loc, _, _ = super().evaluate(self._local_params, config)
        except Exception:
            # Best-effort: if local eval fails, fall back to loss_fed
            loss_loc = loss_fed

        poe_metrics: Dict[str, fl.common.Scalar] = dict(metrics)
        poe_metrics["loss_fed"] = float(loss_fed) if loss_fed is not None else None
        poe_metrics["loss_loc"] = float(loss_loc) if loss_loc is not None else None

        # Return loss_fed as the main loss so existing logs stay consistent.
        return loss_fed, num_examples, poe_metrics


class FedPOERegressionTextClient(BaseClient):
    """FedPOERegressionText client.

    Regression-style Fed-POE update over an RFF head on top of frozen text
    embeddings.

    Protocol:
      - Server sends theta as a single ndarray parameter vector.
      - Client computes per-kernel gradients on local data.
      - Client maintains Hedge weights w (num_kernels,) from per-kernel losses.

    The backbone model parameters are *not* trained nor exchanged here.
    """

    def __repr__(self) -> str:
        return "FedPOERegressionText client"

    def __init__(
        self,
        client_id,
        model,
        trainset,
        valset,
        config_sim,
        device,
        save_dir,
        kl_norm=None,
        dataset=None,
        apply_transforms=None,
        data_scheduler=None,
        bias=None,
    ):
        super().__init__(
            client_id,
            model,
            trainset,
            valset,
            config_sim,
            device,
            save_dir,
            dataset=dataset,
            apply_transforms=apply_transforms,
            data_scheduler=data_scheduler,
            bias=bias,
        )

        # Local imports to avoid heavy deps unless this client is used
        import torch

        poe_cfg = config_sim.get("fedpoe_regression_text_config", {}) or {}
        self.eta = float(poe_cfg.get("eta", 0.0) or 0.0)
        self.lam = float(poe_cfg.get("lam", 0.0) or 0.0)
        self.num_kernels = int(poe_cfg.get("num_kernels", 4) or 4)
        self.n_components = int(poe_cfg.get("n_components", 256) or 256)
        self.pooling = str(poe_cfg.get("pooling", "auto") or "auto")

        # Construct head lazily once we know embedding dim
        self._head = None
        self._w: torch.Tensor | None = None

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _ensure_head(self, embed_dim: int) -> None:
        import torch
        from mak.models.poe_rff_head import RFFConfig, RFFExpertsHead
        from mak.utils.dataset_info import dataset_info

        if self._head is not None:
            return

        dataset_name = self.config_sim["common"]["dataset"]
        num_classes = int(dataset_info[dataset_name]["num_classes"])
        cfg = RFFConfig(
            num_classes=num_classes,
            num_kernels=self.num_kernels,
            n_components=self.n_components,
            lam=self.lam,
            pooling=self.pooling,
        )
        self._head = RFFExpertsHead(embed_dim=embed_dim, cfg=cfg, seed=0).to(self.device)
        self._w = torch.ones(self.num_kernels, device=self.device) / float(self.num_kernels)

    def get_parameters(self, config):
        if self._head is None:
            return []
        return [self._head.get_theta_vector().cpu().numpy()]

    def set_parameters(self, parameters):
        if not parameters:
            return
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from mak.models.poe_rff_head import extract_text_embedding

        vec = torch.from_numpy(np.asarray(parameters[0]))
        if self._head is None:
            dl = DataLoader(self.trainset, batch_size=1)
            batch = next(iter(dl))
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                emb = extract_text_embedding(self.model.to(self.device), input_ids, attention_mask, pooling=self.pooling)
            self._ensure_head(embed_dim=int(emb.shape[-1]))
        self._head.set_theta_vector_(vec.to(self.device))

    def _compute_losses_and_grads(self, loader):
        import torch
        from torch import nn
        from mak.models.poe_rff_head import extract_text_embedding

        assert self._head is not None
        K = self.num_kernels
        ce = nn.CrossEntropyLoss(reduction="mean")

        losses = torch.zeros(K, device=self.device)
        grads = torch.zeros(K, self._head.theta.numel(), device=self.device)
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                x = extract_text_embedding(self.model, input_ids, attention_mask, pooling=self.pooling)

            for k in range(K):
                logits_k = self._head.logits(x, k)
                loss_k = ce(logits_k, labels)
                if self.lam > 0:
                    loss_k = loss_k + 0.5 * self.lam * (self._head.theta[k].pow(2).mean())

                grad_k = torch.autograd.grad(loss_k, self._head.theta, retain_graph=False, create_graph=False)[0]
                losses[k] += loss_k.detach()
                grads[k] += grad_k.detach().flatten()

            n_batches += 1

        if n_batches > 0:
            losses /= float(n_batches)
            grads /= float(n_batches)
        return losses, grads

    def fit(self, parameters, config):
        import torch
        from torch.utils.data import DataLoader

        self.set_parameters(parameters)

        trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        if self._head is None:
            batch = next(iter(trainloader))
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            from mak.models.poe_rff_head import extract_text_embedding
            with torch.no_grad():
                emb = extract_text_embedding(self.model.to(self.device), input_ids, attention_mask, pooling=self.pooling)
            self._ensure_head(embed_dim=int(emb.shape[-1]))

        assert self._head is not None and self._w is not None

        losses, grads = self._compute_losses_and_grads(trainloader)

        if self.eta > 0:
            with torch.no_grad():
                self._w = self._w * torch.exp(-self.eta * losses)
                self._w = self._w / self._w.sum().clamp_min(1e-12)

        theta_vec = self._head.get_theta_vector().cpu().numpy()
        metrics: Dict[str, fl.common.Scalar] = {
            "client_id": self.client_id,
            "poe_w": self._w.detach().cpu().numpy(),
            "poe_losses": losses.detach().cpu().numpy(),
            "poe_grads": grads.detach().cpu().numpy(),
        }

        return [theta_vec], len(trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from mak.models.poe_rff_head import extract_text_embedding

        self.set_parameters(parameters)

        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        if self._head is None:
            return 0.0, len(valloader.dataset), {"accuracy": 0.0}

        ce = nn.CrossEntropyLoss(reduction="mean")
        correct = 0
        total = 0
        loss_sum = 0.0
        n_batches = 0

        for batch in valloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                x = extract_text_embedding(self.model, input_ids, attention_mask, pooling=self.pooling)
                logits = self._head(x)
                loss = ce(logits, labels)

            loss_sum += float(loss.item())
            n_batches += 1
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

        loss_avg = loss_sum / float(max(1, n_batches))
        acc = float(correct) / float(max(1, total))
        return loss_avg, len(valloader.dataset), {"accuracy": acc}
