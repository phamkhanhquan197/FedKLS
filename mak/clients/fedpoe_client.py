from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import flwr as fl

from mak.clients.base_client import BaseClient


class FedPOEClient(BaseClient):
    """Fed-POE client (Flower port, Hedge mixture signals).

    NOTE: In Flower Ray simulations, client objects can be recreated each round.
    This class persists its snapshot pool and Hedge weights to disk so state is
    preserved across rounds.
    """

    def _fedpoe_state_path(self) -> Path:
        """Return path to persisted FedPOE state for this client."""
        # BaseClient sets save_dir; keep FedPOE state isolated to avoid clutter
        root = Path(self.save_dir) if getattr(self, "save_dir", None) else Path(".")
        return root / "fedpoe_state" / f"client_{int(self.client_id)}.pt"

    def _load_fedpoe_state(self) -> None:
        """Load snapshot pool + Hedge weights if they exist on disk."""
        import torch

        p = self._fedpoe_state_path()
        if not p.exists():
            return
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
            self.round = int(state.get("round", 0) or 0)
            self.dic = list(state.get("dic", []) or [])
            self.w = list(state.get("w", []) or [])

            # Basic sanity: align lengths
            n = min(len(self.dic), len(self.w))
            self.dic = self.dic[:n]
            self.w = self.w[:n]
        except Exception:
            # If state is corrupted/incompatible, ignore and start fresh
            self.round = 0
            self.dic = []
            self.w = []

    def _save_fedpoe_state(self) -> None:
        """Persist snapshot pool + Hedge weights to disk (best-effort)."""
        import torch

        p = self._fedpoe_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "round": int(getattr(self, "round", 0) or 0),
            "dic": getattr(self, "dic", []) or [],
            "w": getattr(self, "w", []) or [],
        }
        try:
            torch.save(state, p)
        except Exception:
            # Best-effort only: don't crash training/eval because of IO errors
            return

    def ensemble_predict_and_update_hedge(self, x, y, M=1, eta=0.1):
        """
        Chọn M snapshot từ pool dic theo trọng số Hedge, ensemble predict, update w giống 3rd-party.
        x, y: input (tensor or numpy), single sample or batch.
        M: số snapshot lấy từ pool.
        eta: learning rate Hedge update.
        Returns: ensemble prediction (numpy), list of snapshot losses.
        """
        import torch
        import numpy as np
        device = self.device
        indices = self.model_selection(M)
        if not indices or not self.dic:
            return None, []
        preds = []
        losses = []
        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        # If dataset returns dict features (common in some pipelines), pick a tensor payload
        if isinstance(x, dict):
            for k in ("x", "image", "input", "input_ids"):
                if k in x:
                    x = x[k]
                    break
            else:
                # fall back to first tensor-like value
                for v in x.values():
                    if hasattr(v, "shape"):
                        x = v
                        break

        if isinstance(y, dict):
            for k in ("y", "labels", "label"):
                if k in y:
                    y = y[k]
                    break

        # Lưu lại state_dict hiện tại để restore sau khi predict xong
        import copy
        orig_state = copy.deepcopy(self.model.state_dict())

        # Convert inputs to tensors
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(device)
        else:
            x_tensor = torch.as_tensor(x, device=device)

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).to(device)
        else:
            y_tensor = torch.as_tensor(y, device=device)
        # CrossEntropyLoss expects class indices (Long)
        try:
            y_tensor = y_tensor.long()
        except Exception:
            pass
        for idx in indices:
            self.model.load_state_dict(self.dic[idx])
            self.model.eval()
            with torch.no_grad():
                out = self.model(x_tensor)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                preds.append(out.cpu().numpy())
                loss = ce(out, y_tensor)
                losses.append(float(loss.item()))
        # Ensemble: trung bình xác suất (softmax)
        preds_np = np.stack(preds, axis=0)  # (M, batch, num_class)
        probs = np.mean(preds_np, axis=0)   # (batch, num_class)
        # Update Hedge weights
        self.update_hedge_weights(losses, eta=eta)
        # Restore model state
        self.model.load_state_dict(orig_state)
        return probs, losses

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

        # Pool of model snapshots (PyTorch state_dict)
        self.dic = []  # List of state_dicts (snapshots)
        # Hedge weights for each snapshot
        self.w = []    # List of floats, same length as dic
        # Period for saving snapshot
        self.period = int(config_sim.get('fedpoe_config', {}).get('period', 20))
        # Fixed number of snapshot models to sample per Hedge step (like 3rd-party self.M)
        self.M = int(config_sim.get('fedpoe_config', {}).get('M', 3))
        self.round = 0

        # IMPORTANT: Flower simulation may recreate client objects each round.
        # Persist/restore state so self.dic/self.w survive across rounds.
        self._load_fedpoe_state()
        # print("init client fedpoe")
        
    def model_selection(self, M: int | None = None):
        """Model selection matching the 3rd-party Fed-POE implementation.

        This samples M times from the categorical distribution induced by w
        (via prefix sums + binary search). If a sampled index is duplicated,
        it is skipped (no resampling), so the returned list can be shorter
        than M.
        """
        import numpy as np

        if not self.w:
            return []

        m = int(self.M if M is None else M)
        if m <= 0:
            return []

        total = []
        s = 0
        for weight in self.w:
            s += weight
            total.append(s)

        # If all weights are zero (or negative), total[-1] can be 0 => no sampling
        if not total or total[-1] <= 0:
            return []

        indices = []
        for _ in range(m):
            n = np.random.rand() * total[-1]
            l = 0
            r = len(self.w)
            while l < r:
                mid = (l + r) // 2
                if n > total[mid]:
                    l = mid + 1
                else:
                    r = mid
            if l not in indices:
                indices.append(l)

        return indices

    def fit(self, parameters, config):
        # print("client fittttttttttttt")
        # print("self.period : ", self.period)
        params_to_send, num_examples, metrics = super().fit(parameters, config)
        self._local_params = params_to_send
        # print("id client : ", self.client_id)
        # print(id(self))

        # Pool snapshot logic: save snapshot mỗi period round
        self.round += 1
        # print("self.round : ", self.round)
        if self.round % self.period == 0:
            # Save a deep copy of current model state_dict
            import copy
            self.dic.append(copy.deepcopy(self.model.state_dict()))
            # Khởi tạo trọng số Hedge cho snapshot mới
            self.w.append(1.0)
            # print("self.w in fit : " , self.w)

        # Persist state after each fit so stateless simulations keep progress
        self._save_fedpoe_state()

        return params_to_send, num_examples, metrics
    
    def update_hedge_weights(self, losses, eta=0.1):
        """Update Hedge weights self.w theo losses (list of floats, cùng thứ tự với dic)."""
        # print("update_hedge_weights")
        # x = input()
        import numpy as np
        if not self.w or not losses:
            return
        for i, loss in enumerate(losses):
            self.w[i] *= np.exp(-eta * loss)
        # Optional: normalize w
        s = sum(self.w)
        if s > 0:
            self.w = [w_i / s for w_i in self.w]

        # Persist updated weights
        self._save_fedpoe_state()

    def evaluate(self, parameters, config):
        """Evaluate both fed and local models.

        Flower uses the returned (loss, num_examples, metrics) for aggregation.
        We return `loss_fed` as the primary loss (so standard aggregation still
        makes sense) and include both losses in metrics.
        """

        print("evaluate client model")
        
        import torch
        import numpy as np

        # 1) Evaluate fed/global parameters (the ones provided by server)
        loss_fed, num_examples, metrics = super().evaluate(parameters, config)

        # 2) Evaluate local/personalized parameters (stored after last fit)
        loss_loc = loss_fed
        # print("self.w : " , self.w)
        # # print("self.dic : " , self.dic)
        # print("metrics : ", metrics)
        # print("num_examples : ", num_examples)
        # print("loss_fed : ", loss_fed)
        # print("id client : ", self.client_id)
        # print(id(self))
        # x = input()

        if self.dic and self.w:
            # Lấy batch đầu tiên từ valset để tính ensemble loss
            loader = torch.utils.data.DataLoader(self.valset, batch_size=self.test_batch_size)
            try:
                # print("Try calculate loc loss")
                batch = next(iter(loader))
                x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"] if "x" in batch else batch
                y = batch[1] if isinstance(batch, (list, tuple)) else batch["y"] if "y" in batch else batch
                # Nếu batch là dict, ưu tiên "x", "y" hoặc "input", "label"
                if isinstance(batch, dict):
                    # IMPORTANT: don't use `or` on tensors (ambiguous truth value)
                    if "x" in batch:
                        x = batch["x"]
                    elif "input" in batch:
                        x = batch["input"]
                    elif "input_ids" in batch:
                        x = batch["input_ids"]

                    if "y" in batch:
                        y = batch["y"]
                    elif "labels" in batch:
                        y = batch["labels"]
                    elif "label" in batch:
                        y = batch["label"]
                # Chuyển về numpy nếu là tensor
                if hasattr(x, "cpu"):
                    x = x.cpu().numpy()
                if hasattr(y, "cpu"):
                    y = y.cpu().numpy()
                # Gọi ensemble_predict_and_update_hedge
                # print("x : {}, y :{}".format(x, y))
                # Use fixed M like 3rd-party; ensemble will naturally use <=M if pool is smaller
                probs, losses = self.ensemble_predict_and_update_hedge(x, y, M=self.M, eta=0.1)
                # print("probs : {}, losses : {}".format(probs, losses))
                if losses:
                    loss_loc = float(np.mean(losses))
            except Exception as e:
                # print("loc loss exception:", repr(e))
                loss_loc = loss_fed
        else:
            # Nếu chưa có snapshot, fallback về loss_fed
            loss_loc = loss_fed

        poe_metrics: Dict[str, fl.common.Scalar] = dict(metrics)
        poe_metrics["loss_fed"] = float(loss_fed) if loss_fed is not None else None
        poe_metrics["loss_loc"] = float(loss_loc) if loss_loc is not None else None

        # Persist state in case evaluate updated weights (ensemble path)
        self._save_fedpoe_state()

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
