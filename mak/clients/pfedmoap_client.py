# mak/clients/pfedmoap_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from mak.clients.base_client import BaseClient


class PFedMoAPState(nn.Module):
    """
    State container attached to self.model so optimizer(model.parameters()) can see prompt and gating params.

    Phase 1 scope:
      - store trainable local prompt
      - store trainable gating network (local only)
      - store non-local prompts received from server each round (not trainable)
    """

    def __init__(self, prompt_len: int, prompt_dim: int, dgating: int = 128, heads: int = 8):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.prompt_dim = int(prompt_dim)

        # Trainable local prompt: [L, D]
        self.local_prompt = nn.Parameter(torch.randn(self.prompt_len, self.prompt_dim) * 0.02)

        # Local gating network (trainable, never uploaded)
        # Phase 1: keep as simple MLP to avoid relying on model internals.
        # Phase 2: you can swap to MHA gating and expert feature caching.
        self.gating = nn.Sequential(
            nn.Linear(self.prompt_dim, dgating),
            nn.ReLU(),
            nn.Linear(dgating, heads),
        )

        # Non-local prompts buffer: [K, L, D]
        self.register_buffer("non_local_prompts", torch.zeros(0, self.prompt_len, self.prompt_dim), persistent=False)

        # Optional metadata for debugging
        self.last_round: int = 0

    @torch.no_grad()
    def set_global_prompt(self, prompt: torch.Tensor):
        """Initialize local prompt from global prompt (server broadcast)."""
        if prompt.shape != self.local_prompt.shape:
            raise ValueError(
                f"Global prompt shape {tuple(prompt.shape)} does not match local_prompt {tuple(self.local_prompt.shape)}"
            )
        self.local_prompt.copy_(prompt)

    @torch.no_grad()
    def set_non_local_prompts(self, prompts: torch.Tensor):
        """Set non-local prompts for current round."""
        # prompts: [K, L, D] or empty [0, L, D]
        if prompts.ndim != 3:
            raise ValueError(f"non_local_prompts must be rank-3 [K, L, D], got shape {tuple(prompts.shape)}")
        if prompts.shape[1:] != self.local_prompt.shape:
            raise ValueError(
                f"non_local_prompts trailing dims {tuple(prompts.shape[1:])} "
                f"must match local_prompt {tuple(self.local_prompt.shape)}"
            )
        self.non_local_prompts = prompts


class PFedMoAPClient(BaseClient):
    """
    Phase 1: Client protocol implementation for pFedMoAP

    - Flower parameters are interpreted as a single global prompt [L, D]
    - Client uploads only its local prompt [L, D]
    - Gating network is local only
    - Non-local prompts received through config (configure_fit), stored in PFedMoAPState
    - Backbone is frozen so optimizer(model.parameters()) only updates prompt and gating
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self.config_sim.get("pfedmoap_config", {})
        self.prompt_len = int(cfg.get("prompt_len", cfg.get("num_tokens", 16)))
        self.prompt_dim = int(cfg.get("prompt_dim", 768))
        self.dgating = int(cfg.get("dgating", 128))
        self.heads = int(cfg.get("heads", 8))

        # Attach state to model so optimizer sees it
        if not hasattr(self.model, "pfedmoap"):
            self.model.pfedmoap = PFedMoAPState(
                prompt_len=self.prompt_len,
                prompt_dim=self.prompt_dim,
                dgating=self.dgating,
                heads=self.heads,
            )

        # Cache device string for convenience
        self._device_str = str(self.device)

    # -------------------------
    # Flower parameter protocol
    # -------------------------

    def set_parameters(self, parameters):
        """
        Interpret Flower parameters as global prompt and copy into local prompt.
        parameters format depends on Flower internals. BaseClient uses list of NDArrays.
        Here we accept both:
          - list-like of numpy arrays [prompt]
          - flwr Parameters converted upstream to ndarrays (still list-like)
        """
        if parameters is None:
            return

        # parameters could be List[np.ndarray] or List[torch.Tensor]
        if isinstance(parameters, (list, tuple)) and len(parameters) == 1:
            arr = parameters[0]
        else:
            # Defensive: allow passing already as ndarray list of length 1
            if isinstance(parameters, (list, tuple)) and len(parameters) > 1:
                raise ValueError(
                    f"PFedMoAP expects exactly 1 parameter (prompt), got {len(parameters)}. "
                    "This usually means initial_parameters is still model.state_dict()."
                )
            arr = parameters

        if isinstance(arr, torch.Tensor):
            prompt = arr.detach().to(self.device)
        else:
            prompt = torch.from_numpy(np.asarray(arr)).to(self.device)

        # Ensure correct dtype
        prompt = prompt.to(dtype=self.model.pfedmoap.local_prompt.dtype)

        self.model.pfedmoap.set_global_prompt(prompt)

    def get_parameters(self, config):
        """
        Upload only local prompt as numpy array list [prompt].
        Do not send gating or backbone.
        """
        prompt = self.model.pfedmoap.local_prompt.detach().cpu().numpy()
        return [prompt]

    # -------------------------
    # Round-specific config
    # -------------------------

    def _parse_non_local_prompts_from_config(self, config: Dict[str, Any]) -> torch.Tensor:
        """
        Strategy Phase 2 will set:
          config["pfedmoap_non_local_prompts"] as nested lists or base64 payload
        Phase 1: support nested lists (List[List[List[float]]]) for simplicity.
        """
        raw = config.get("pfedmoap_non_local_prompts", None)
        if raw is None:
            return torch.zeros(0, self.prompt_len, self.prompt_dim, device=self.device)

        # raw expected: List[K] of [L, D]
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size == 0:
            return torch.zeros(0, self.prompt_len, self.prompt_dim, device=self.device)

        prompts = torch.from_numpy(arr).to(self.device)
        prompts = prompts.to(dtype=self.model.pfedmoap.local_prompt.dtype)
        return prompts

    def _freeze_backbone_only_train_pfedmoap(self):
        # Freeze everything first
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # Unfreeze prompt + gating
        self.model.pfedmoap.local_prompt.requires_grad = True
        for p in self.model.pfedmoap.gating.parameters():
            p.requires_grad = True

    def fit(self, parameters, config):
        """
        Keep training loop semantics:
          - set_parameters first
          - prepare per-round state from config
          - freeze backbone before optimizer creation
          - call BaseClient.fit to reuse dataloaders, logging, loss, metrics
        """
        # 1) set global prompt
        self.set_parameters(parameters)

        # 2) per-round non-local prompts
        non_local = self._parse_non_local_prompts_from_config(config)
        self.model.pfedmoap.set_non_local_prompts(non_local)
        self.model.pfedmoap.last_round = int(config.get("current_round", 0))

        # 3) freeze backbone, allow prompt + gating
        self._freeze_backbone_only_train_pfedmoap()

        # 4) reuse BaseClient.fit
        return super().fit(parameters, config)
