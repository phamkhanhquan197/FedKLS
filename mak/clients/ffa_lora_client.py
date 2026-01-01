from logging import INFO
from typing import List

import numpy as np
import torch

from flwr.common.logger import log
from mak.clients.base_client import BaseClient
from collections import OrderedDict

class FFALoRAClient(BaseClient):
    """
    FFA-LoRA Client (paper-faithful implementation)

    - A is initialized once and frozen forever
    - First communication: full (A + B)
    - Later communications: B only
    - Client infers protocol from parameter length or server flag
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None, dataset=None, apply_transforms=None, data_scheduler=None
    ):
        super().__init__(client_id, model, trainset, valset, config_sim, device, save_dir, dataset=dataset, apply_transforms=apply_transforms, data_scheduler=data_scheduler)

    def __repr__(self) -> str:
        return " FFA-LoRA client"
    
    def _freeze_all_A(self) -> None:
        for name, p in self.model.named_parameters():
            if name.endswith(".A"):
                p.requires_grad = False

    def get_parameters(self, config):
        """
        Only send B adapters to the server.
        """
        model_state = self.model.state_dict()

        if any(key.startswith("distilbert.") for key in model_state.keys()):  # DistilBERT-based model
            params_to_send = {
                name: tensor for name, tensor in model_state.items()
                if name.endswith(".B") or (name.endswith(".bias") and "lin" in name)
            }

        elif any(key.startswith("bert.") for key in model_state.keys()):  # BERT-based model
            params_to_send = {
                name: tensor for name, tensor in model_state.items()
                if name.endswith(".B")
                or (name.endswith(".bias") and "self" in name)
                or (name.endswith(".bias") and "dense" in name)
            }

        elif any(key.startswith("model.") for key in model_state.keys()):
            params_to_send = {
                name: tensor for name, tensor in model_state.items()
                if name.endswith(".B")
                or (name.endswith(".bias") and "self_attn" in name)
                or (name.endswith(".bias") and "mlp" in name)
            }

        else:
            raise NotImplementedError("Unsupported model type for FFA-LoRA")

        # Minimal but critical fix: enforce deterministic order
        return [
            tensor.cpu().numpy()
            for _, tensor in sorted(params_to_send.items())
        ]

    def set_parameters(self, parameters, device: str = "cuda"):
        """Set model weights from a list of NumPy ndarrays."""
        if parameters is None:
            return

        model_state = self.model.state_dict()

        log(INFO, f"FFA len(parameters): {len(parameters)}")
        log(INFO, f"FFA len(model_state.items()): {len(model_state.items())}")

        # ------------------------------------------------------
        # Case 1: Full model update (Round 1)
        # ------------------------------------------------------
        if len(parameters) == len(model_state):
            params_dict = zip(model_state.keys(), parameters)
            state_dict = OrderedDict(
                (k, torch.from_numpy(v).to(device))
                for k, v in params_dict
            )

            self.model.load_state_dict(state_dict, strict=False)

            log(INFO, "FFA-LoRA Client: Freezing all LoRA-A parameters after full model initialization.")
            self._freeze_all_A()
            return

        # ------------------------------------------------------
        # Case 2: LoRA-only update (Round > 1)
        # ------------------------------------------------------
        # Identify LoRA-B + bias keys explicitly (order does NOT matter globally,
        # but MUST be deterministic locally)
        if any(k.startswith("distilbert.") for k in model_state.keys()):
            lora_keys = [
                k for k in model_state.keys()
                if k.endswith(".B") or (k.endswith(".bias") and "lin" in k)
            ]
        elif any(k.startswith("bert.") for k in model_state.keys()):
            lora_keys = [
                k for k in model_state.keys()
                if (
                    k.endswith(".B")
                    or (k.endswith(".bias") and "self" in k)
                    or (k.endswith(".bias") and "dense" in k)
                )
            ]
        elif any(k.startswith("model.") for k in model_state.keys()):
            lora_keys = [
                k for k in model_state.keys()
                if (
                    k.endswith(".B")
                    or (k.endswith(".bias") and "self_attn" in k)
                    or (k.endswith(".bias") and "mlp" in k)
                )
            ]
        else:
            raise ValueError("Unsupported model type for FFA-LoRA set_parameters")

        # Safety check
        assert len(lora_keys) == len(parameters), (
            f"Mismatch: {len(lora_keys)} LoRA keys vs {len(parameters)} received tensors"
        )

        # Name-based update (this is the critical fix)
        for key, array in zip(lora_keys, parameters):
            model_state[key] = torch.from_numpy(array).to(device)

        self.model.load_state_dict(model_state, strict=True)

    # def set_parameters(self, parameters, device: str = "cuda"):
    #     """Set model weights from a list of NumPy ndarrays."""
    #     model_state = self.model.state_dict()
    #     if parameters is None:
    #         return

    #     print(f"FFA len(parameters): {len(parameters)}")
    #     print(f"FFA len(model_state.items()): {len(model_state.items())}")

    #     # ------------------------------------------------------
    #     # LoRA-only update (Round > 1)
    #     # ------------------------------------------------------
    #     if len(model_state.items()) != len(parameters):

    #         if any(key.startswith("distilbert.") for key in model_state.keys()):
    #             lora_keys = sorted(
    #                 k for k in model_state.keys()
    #                 if k.endswith(".B") or (k.endswith(".bias") and "lin" in k)
    #             )
    #             log(INFO, f"FFA LoRA keys: {len(lora_keys)}")

    #         elif any(key.startswith("bert.") for key in model_state.keys()):
    #             lora_keys = sorted(
    #                 k for k in model_state.keys()
    #                 if (
    #                     k.endswith(".B")
    #                     or (k.endswith(".bias") and "self" in k)
    #                     or (k.endswith(".bias") and "dense" in k)
    #                 )
    #             )

    #         elif any(key.startswith("model.") for key in model_state.keys()):
    #             lora_keys = sorted(
    #                 k for k in model_state.keys()
    #                 if (
    #                     k.endswith(".B")
    #                     or (k.endswith(".bias") and "self_attn" in k)
    #                     or (k.endswith(".bias") and "mlp" in k)
    #                 )
    #             )
    #         else:
    #             raise NotImplementedError("Unsupported model type for FFA-LoRA")

    #         # Minimal but critical fix: stable key ↔ tensor alignment
    #         lora_params = OrderedDict()
    #         for key, array in zip(lora_keys, parameters):
    #             lora_params[key] = torch.from_numpy(array).to(device)

    #         model_state.update(lora_params)
    #         self.model.load_state_dict(model_state, strict=True)

    #     # ------------------------------------------------------
    #     # Full model update (Round 1)
    #     # ------------------------------------------------------
    #     else:
    #         params_dict = zip(model_state.keys(), parameters)
    #         state_dict = OrderedDict(
    #             (k, v.clone().detach().to(device) if isinstance(v, torch.Tensor)
    #             else torch.tensor(v, device=device))
    #             for k, v in params_dict
    #         )
    #         print("len(state_dict): ", len(state_dict))
    #         print("len(params_dict): ", len(list(params_dict)))
    #         self.model.load_state_dict(state_dict, strict=False)
    #         log(INFO, "FFA-LoRA Client: Freezing all LoRA-A parameters after full model initialization.")
    #         self._freeze_all_A()



