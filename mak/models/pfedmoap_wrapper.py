# mak/models/pfedmoap_wrapper.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PFedMoAPGating(nn.Module):
    """
    Attention-style gating producing expert weights over (local + non-local) prompts.

    We keep it lightweight:
      - pool query from token embeddings -> dgating
      - pool keys from each prompt -> dgating
      - dot-product attention -> weights over experts
    Works with variable #experts each round.
    """

    def __init__(self, prompt_dim: int, dgating: int = 128):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.dgating = int(dgating)

        self.q_proj = nn.Linear(self.prompt_dim, self.dgating)
        self.k_proj = nn.Linear(self.prompt_dim, self.dgating)

    def forward(self, token_embeds: torch.Tensor, expert_prompts: torch.Tensor) -> torch.Tensor:
        """
        token_embeds: [B, T, D]  (original token embeddings, no prompt)
        expert_prompts: [E, L, D] (E experts, each prompt length L)
        return weights: [B, E]
        """
        # Query: mean pool token embeddings
        q = token_embeds.mean(dim=1)  # [B, D]
        q = self.q_proj(q)            # [B, G]

        # Keys: mean pool each prompt
        k = expert_prompts.mean(dim=1)  # [E, D]
        k = self.k_proj(k)              # [E, G]

        # scores: [B, E]
        scores = torch.matmul(q, k.t()) / (self.dgating ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return weights


class PFedMoAPPromptWrapper(nn.Module):
    """
    Wrap a HuggingFace-style sequence classification model so that prompts
    affect forward() without changing BaseClient.train loop.

    Requirement: base_model forward must accept inputs_embeds and attention_mask.
    (BERT/DistilBERT/DeBERTa typically do.)
    """

    def __init__(
        self,
        base_model: nn.Module,
        prompt_len: int,
        prompt_dim: int,
        dgating: int = 128,
        lambda_local: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.prompt_len = int(prompt_len)
        self.prompt_dim = int(prompt_dim)
        self.dgating = int(dgating)
        self.lambda_local = float(lambda_local)
        self.temperature = float(temperature)

        self.gating = PFedMoAPGating(prompt_dim=self.prompt_dim, dgating=self.dgating)

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        raise AttributeError("Base model does not expose get_input_embeddings()")

    def _prepend_prompt(
        self,
        inputs_embeds: torch.Tensor,       # [B, T, D]
        attention_mask: Optional[torch.Tensor],  # [B, T]
        prompt_embeds: torch.Tensor,       # [B, L, D]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = inputs_embeds.shape
        L = prompt_embeds.shape[1]
        x = torch.cat([prompt_embeds, inputs_embeds], dim=1)  # [B, L+T, D]

        if attention_mask is None:
            return x, None

        prompt_mask = torch.ones((B, L), device=attention_mask.device, dtype=attention_mask.dtype)
        m = torch.cat([prompt_mask, attention_mask], dim=1)  # [B, L+T]
        return x, m

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # If no pfedmoap state exists, fall back to vanilla model forward
        if not hasattr(self, "pfedmoap"):
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        state = self.pfedmoap

        # Build token embeddings from input_ids
        emb_layer = self.get_input_embeddings()
        token_embeds = emb_layer(input_ids)  # [B, T, D]

        # Expert prompts: local + non-local
        # local_prompt: [L, D], non_local_prompts: [K, L, D]
        local = state.local_prompt.unsqueeze(0)  # [1, L, D]
        non_local = state.non_local_prompts      # [K, L, D] maybe empty

        if non_local is None or non_local.numel() == 0:
            experts = local  # [1, L, D]
        else:
            experts = torch.cat([local, non_local], dim=0)  # [E, L, D]

        # Gating weights over experts using pooled token embeddings
        weights = self.gating(token_embeds, experts)  # [B, E]

        # Mixture prompt: sum_e w_e * P_e  -> [B, L, D]
        mixed_prompt = torch.einsum("be,eld->bld", weights, experts)

        # Optional: add explicit local residual term (lambda) similar spirit to Eq(10)
        # Here we bias the prompt embedding toward local prompt.
        if self.lambda_local != 0.0:
            mixed_prompt = mixed_prompt + self.lambda_local * state.local_prompt.unsqueeze(0)

        # Prepend prompt and call base model using inputs_embeds
        inputs_embeds, attention_mask = self._prepend_prompt(token_embeds, attention_mask, mixed_prompt)

        # Temperature: apply at logits level (if available)
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

        if hasattr(outputs, "logits") and self.temperature != 1.0:
            outputs.logits = outputs.logits / self.temperature
        return outputs
