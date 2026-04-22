# mak/models/clip.py
# PFedMoAP-ready clip wrapper built on the local "clip/" folder you copied:
# clip/__init__.py, clip/clip.py, clip/model.py, clip/simple_tokenizer.py
#
# Notes:
# - Image encoder is always frozen (required).
# - Text encoder is frozen by default (paper setting). You can unfreeze via freeze_text=False.
# - Designed to be driven by PFedMoAPClient/PFedMoAPStrategy:
#   - set_prompt / get_prompt
#   - load_nonlocal_prompts / clear_nonlocal
#
# Forward returns logits: (B, num_classes), compatible with your BaseClient image branch.

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import mak.models.clip_backbone as clip


class _TextEncoder(nn.Module):
    """Thin wrapper around CLIP text tower to encode prompt embeddings."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        # prompt_embeddings: (C, L, D), expected float32 for numeric stability
        x = prompt_embeddings + self.positional_embedding.float()
        x = x.permute(1, 0, 2)  # (L, C, D)
        x = self.transformer(x)  # keep in float32
        x = x.permute(1, 0, 2)  # (C, L, D)
        x = self.ln_final(x).float()

        # Take features at EOT position
        eot_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection.float()
        return x


class _PromptLearner(nn.Module):
    """
    CoOp-like prompt learner.
    Builds prompt embeddings by replacing placeholder tokens with learnable ctx.

    token layout per class:
    [SOS] [CTX_1 ... CTX_m] [rest tokens from template including class + EOS + padding]
    """

    def __init__(
        self,
        clip_model: nn.Module,
        classnames: Sequence[str],
        prompt_len: int,
        template: str,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.classnames = list(classnames)
        self.n_class = len(self.classnames)
        self.prompt_len = int(prompt_len)

        # Build tokenized prompts with placeholders "X"
        placeholder = " ".join(["X"] * self.prompt_len)
        texts = [template.format(placeholder, name) for name in self.classnames]
        tokenized = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
        self.register_buffer("tokenized_prompts", tokenized)

        # Embed tokens once to get fixed prefix/suffix structure
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).float()

        # Prefix: SOS token only
        self.register_buffer("token_prefix", embedding[:, :1, :])  # (C, 1, D)
        # Suffix: tokens after ctx positions
        self.register_buffer("token_suffix", embedding[:, 1 + self.prompt_len :, :])  # (C, L-1-m, D)

        # Learnable context vectors (shared across classes)
        ctx = torch.empty(self.prompt_len, embedding.size(-1), dtype=torch.float32, device=self.device)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)  # (m, D)

    def forward(self) -> torch.Tensor:
        # Expand ctx to (C, m, D)
        ctx = self.ctx.unsqueeze(0).expand(self.n_class, -1, -1)
        # Concatenate to full prompt embeddings
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1).float()  # (C, L, D)

    def get_ctx(self) -> torch.Tensor:
        return self.ctx.detach()

    def set_ctx(self, ctx: torch.Tensor) -> None:
        assert ctx.shape == self.ctx.shape, f"ctx shape mismatch: got {tuple(ctx.shape)}, need {tuple(self.ctx.shape)}"
        self.ctx.data.copy_(ctx.to(device=self.ctx.device, dtype=self.ctx.dtype))


class _MultiheadAttention(nn.Module):
    """Simple MHA for query length 1 (we still implement generally)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q: (B, Nq, D), K/V: (B, Nk, D)
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Qh = self.Wq(Q).view(B, Nq, self.num_heads, self.dk).transpose(1, 2)  # (B, h, Nq, dk)
        Kh = self.Wk(K).view(B, Nk, self.num_heads, self.dk).transpose(1, 2)  # (B, h, Nk, dk)
        Vh = self.Wv(V).view(B, Nk, self.num_heads, self.dk).transpose(1, 2)  # (B, h, Nk, dk)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.dk ** 0.5)

        # numeric safety: fp32 softmax with max subtraction
        scores = scores.float()
        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-1).to(dtype=Vh.dtype)

        out = torch.matmul(attn, Vh)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)  # (B, Nq, D)
        return self.Wo(out)


class Clip(nn.Module):
    """
    PFedMoAP-ready CLIP model for FedKLS.

    Key API used by PFedMoAP client/strategy:
    - get_prompt() -> torch.Tensor (prompt_len, D)
    - set_prompt(prompt_tensor)
    - load_nonlocal_prompts(list_of_prompt_tensors)
    - clear_nonlocal()
    """

    def __init__(
        self,
        num_classes: int,
        input_shape=None,
        weights=None,
        *,
        backbone_name: str = "ViT-B/32",
        classnames: Optional[Sequence[str]] = None,
        prompt_len: int = 16,
        num_experts: int = 4,
        dgating: int = 128,
        gating_heads: int = 4,
        lambda_local: float = 1.0,
        freeze_text: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        template: str = "{} {}",
        debug: bool = False,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.lambda_local = float(lambda_local)
        self.dgating = int(dgating)
        self.debug = bool(debug)

        # Default classnames if not provided
        if classnames is None:
            classnames = [f"class{i}" for i in range(self.num_classes)]
        if len(classnames) != self.num_classes:
            classnames = list(classnames)[: self.num_classes]

        # Load CLIP from local "clip/" folder
        clip_model, _ = clip.load(backbone_name, device=self.device, jit=False)
        clip_model.eval()

        # Force float32 for numeric stability (fix NaN in text branch)
        clip_model = clip_model.float()

        self.clip_model = clip_model
        self.dtype = torch.float32

        # Encoders
        self.image_encoder = clip_model.visual
        self.text_encoder = _TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        # Freeze image encoder (required)
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # Freeze text encoder (paper default)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for name, p in clip_model.named_parameters():
                if "visual" in name:
                    continue
                p.requires_grad = False

        # Prompt learner
        self.prompt_learner = _PromptLearner(
            clip_model=clip_model,
            classnames=classnames,
            prompt_len=prompt_len,
            template=template,
            device=self.device,
        )

        # Gating: pool down to dgating, apply MHA, then map back up
        clip_width = clip_model.ln_final.weight.shape[0]
        self.img_pool = nn.Linear(clip_width, self.dgating)
        self.txt_pool = nn.Linear(clip_width, self.dgating)
        self.attn = _MultiheadAttention(self.dgating, gating_heads)
        self.txt_up = nn.Linear(self.dgating, clip_width)

        # Non-local buffers
        self.nonlocal_ctx_list: Optional[List[torch.Tensor]] = None
        self.nonlocal_text_features: Optional[torch.Tensor] = None  # (K, C, D)

        # Load weights if provided
        if weights is not None:
            self.load_state_dict(weights, strict=False)

        self.to(self.device)

    # -------------------------
    # PFedMoAP control surface
    # -------------------------
    def get_prompt(self) -> torch.Tensor:
        # returns (prompt_len, D) on CPU for safe serialization
        return self.prompt_learner.get_ctx().detach().float().cpu()

    def set_prompt(self, prompt_tensor: torch.Tensor) -> None:
        assert prompt_tensor.ndim == 2, f"prompt must be 2D, got {prompt_tensor.shape}"
        assert prompt_tensor.shape == self.prompt_learner.ctx.shape, (
            f"ctx shape mismatch: got {prompt_tensor.shape}, need {self.prompt_learner.ctx.shape}"
        )
        self.prompt_learner.set_ctx(prompt_tensor.to(self.device))

    def load_nonlocal_prompts(self, prompt_list: Sequence[torch.Tensor]) -> None:
        if prompt_list is None or len(prompt_list) == 0:
            self.clear_nonlocal()
            return
        self.nonlocal_ctx_list = [p.detach().to(self.device, dtype=torch.float32) for p in prompt_list]
        self._compute_nonlocal_text_features()

    def clear_nonlocal(self) -> None:
        self.nonlocal_ctx_list = None
        self.nonlocal_text_features = None

    # -------------------------
    # Internal: compute cached non-local text features
    # -------------------------
    @torch.no_grad()
    def _compute_nonlocal_text_features(self) -> None:
        if not self.nonlocal_ctx_list:
            self.nonlocal_text_features = None
            return

        local_ctx = self.prompt_learner.get_ctx().clone()

        feats = []
        for ctx in self.nonlocal_ctx_list:
            self.prompt_learner.set_ctx(ctx)
            prompt_embeddings = self.prompt_learner()  # (C, L, D) 
            tokenized = self.prompt_learner.tokenized_prompts  # (C, L)

            text_feat = self.text_encoder(prompt_embeddings, tokenized)
            text_feat = F.normalize(text_feat, dim=-1)

            feats.append(text_feat)

        self.prompt_learner.set_ctx(local_ctx)
        self.nonlocal_text_features = torch.stack(feats, dim=0)  # (K, C, D)

    # -------------------------
    # Debug helper
    # -------------------------
    def _stat(self, name: str, x: torch.Tensor) -> None:
        if not self.debug:
            return
        x_f = x.detach().float()
        ok = torch.isfinite(x_f).all().item()
        print(
            f"[PFedMoAP][DBG] {name} finite={ok} "
            f"nan={torch.isnan(x_f).any().item()} inf={torch.isinf(x_f).any().item()} "
            f"min={x_f.nan_to_num().min().item():.6f} max={x_f.nan_to_num().max().item():.6f}"
        )

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Encode image
        img_feat = self.image_encoder(images.type(self.dtype))
        img_feat = F.normalize(img_feat, dim=-1)  # (B, D)

        # Encode local text prompts
        prompt_embeddings = self.prompt_learner()  # (C, L, D)
        tokenized = self.prompt_learner.tokenized_prompts
        txt_feat_local = self.text_encoder(prompt_embeddings, tokenized)  # (C, D)
        txt_feat_local = F.normalize(txt_feat_local, dim=-1)

        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * (img_feat @ txt_feat_local.t())  # (B, C)

        # If no non-local experts, return local logits
        if self.nonlocal_text_features is None:
            return local_logits

        # Experts for each class: local + nonlocal
        # local: (C, D)
        # nonlocal: (K, C, D)
        # stack to (C, E, D) where E = 1 + K
        experts = torch.cat(
            [txt_feat_local.unsqueeze(0), self.nonlocal_text_features],
            dim=0,
        )  # (E, C, D)
        experts = experts.permute(1, 0, 2).contiguous()  # (C, E, D)

        # Pool for gating
        img_g = self.img_pool(img_feat)  # (B, dg)
        exp_g = self.txt_pool(experts)  # (C, E, dg)

        # Build Q, K, V for attention per (B, C)
        B = img_g.size(0)
        C = exp_g.size(0)
        E = exp_g.size(1)
        dg = exp_g.size(2)

        Q = img_g.unsqueeze(1).expand(B, C, dg).reshape(B * C, 1, dg)  # (BC, 1, dg)
        K = exp_g.unsqueeze(0).expand(B, C, E, dg).reshape(B * C, E, dg)  # (BC, E, dg)
        V = K

        fused_g = self.attn(Q, K, V).squeeze(1)  # (BC, dg)
        fused = self.txt_up(fused_g).view(B, C, -1)  # (B, C, D)
        fused = F.normalize(fused, dim=-1)

        # Similarity between image feat and fused per class
        moe_logits = logit_scale * torch.sum(img_feat.unsqueeze(1) * fused, dim=-1)  # (B, C)

        return moe_logits + (self.lambda_local * local_logits)