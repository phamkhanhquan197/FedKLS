from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn


Pooling = Literal["auto", "cls", "mean"]


@dataclass
class RFFConfig:
    num_classes: int
    num_kernels: int = 4
    n_components: int = 256
    lam: float = 0.0
    pooling: Pooling = "auto"


class RFFExpertsHead(nn.Module):
    """Random Fourier Features experts head (classification).

    This is a lightweight head which maps an embedding vector x (d,) to a set of
    expert logits. Each expert k has its own random projection W_k and phase b_k,
    and a trainable linear classifier theta_k.

    Features per expert:
      z_k(x) = sqrt(2/D) * cos(x W_k + b_k)  where D = n_components

    Then logits:
      logits_k = z_k(x) @ theta_k  with theta_k shape (D, C)

    The server aggregates theta_k across clients; W_k and b_k are fixed random
    features shared across all clients.
    """

    def __init__(self, *, embed_dim: int, cfg: RFFConfig, seed: int = 0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(cfg.num_classes)
        self.num_kernels = int(cfg.num_kernels)
        self.n_components = int(cfg.n_components)
        self.lam = float(cfg.lam)
        self.pooling: Pooling = cfg.pooling

        g = torch.Generator()
        g.manual_seed(int(seed))

        # Random features (fixed)
        # W: (K, d, D)
        W = torch.randn(self.num_kernels, self.embed_dim, self.n_components, generator=g) / math.sqrt(
            max(1, self.embed_dim)
        )
        b = 2 * math.pi * torch.rand(self.num_kernels, self.n_components, generator=g)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        # Trainable thetas per kernel: (K, D, C)
        theta = torch.zeros(self.num_kernels, self.n_components, self.num_classes)
        nn.init.normal_(theta, std=0.02)
        self.theta = nn.Parameter(theta)

    @torch.no_grad()
    def get_theta_vector(self) -> torch.Tensor:
        return self.theta.detach().flatten()

    @torch.no_grad()
    def set_theta_vector_(self, vec: torch.Tensor) -> None:
        vec = vec.to(self.theta.device, dtype=self.theta.dtype)
        self.theta.copy_(vec.view_as(self.theta))

    def rff(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: (B, d)
        proj = x @ self.W[k]  # (B, D)
        proj = proj + self.b[k]
        z = math.sqrt(2.0 / float(self.n_components)) * torch.cos(proj)
        return z

    def logits(self, x: torch.Tensor, k: int) -> torch.Tensor:
        z = self.rff(x, k)  # (B, D)
        return z @ self.theta[k]  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mixture logits as a simple average across experts."""
        logits = None
        for k in range(self.num_kernels):
            lk = self.logits(x, k)
            logits = lk if logits is None else (logits + lk)
        return logits / float(self.num_kernels)


def choose_pooling(model) -> Pooling:
    # Heuristic: seq-classification encoder -> CLS, causal LM -> mean
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "mean"

    model_type = getattr(cfg, "model_type", "")
    if model_type in {"bert", "distilbert", "roberta", "deberta", "albert", "electra"}:
        return "cls"
    if model_type in {"llama", "qwen2", "qwen", "mistral", "gpt2", "gpt_neox"}:
        return "mean"

    # fallback
    return "mean"


def extract_text_embedding(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling: Pooling = "auto",
) -> torch.Tensor:
    """Extract a per-example embedding from a HF model.

    Works for most transformer encoders/decoders by using `output_hidden_states`.

    Returns: (B, d)
    """

    if pooling == "auto":
        pooling = choose_pooling(model)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; cannot extract embeddings")

    last = hidden_states[-1]  # (B, T, d)

    if pooling == "cls":
        return last[:, 0, :]

    # mean pooling over tokens with attention_mask
    if attention_mask is None:
        return last.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(last.dtype)  # (B,T,1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (last * mask).sum(dim=1) / denom
