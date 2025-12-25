from logging import INFO
from typing import List

import numpy as np

from flwr.common.logger import log

from mak.clients.base_client import BaseClient


class FFALoRAClient(BaseClient):
    """FFA-LoRA Client: A frozen forever, train and communicate B only after round 1.

    Communication protocol (List NDArrays, ordered):
    - Round 1 downlink: full LoRA params [A1, B1, A2, B2, ...]
    - Round >1 downlink: B-only [B1, B2, ...]

    - Uplink Round 1: full LoRA params [A1, B1, A2, B2, ...] (as requested)
    - Uplink Round >1: B-only [B1, B2, ...]

    NOTE: We freeze A externally (do not modify adapter classes).
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None
    ):
        super().__init__(client_id, model, trainset, valset, config_sim, device, save_dir)
        self._full_lora_param_names = self._get_full_lora_param_names()
        self._b_only_param_names = [n for n in self._full_lora_param_names if n.endswith(".B")]

    def __repr__(self) -> str:
        return " FFA-LoRA client"

    def _get_full_lora_param_names(self) -> List[str]:
        """Full LoRA parameter order as they appear in model.named_parameters()."""
        names: List[str] = []
        for name, _ in self.model.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                names.append(name)
        return names

    def _freeze_all_A(self) -> None:
        for name, p in self.model.named_parameters():
            if name.endswith(".A"):
                p.requires_grad = False

    def set_parameters(self, parameters):
        """Set parameters based on list length: full (A+B) or B-only."""
        # Defensive conversion (Flower provides list of numpy arrays)
        if parameters is None:
            return

        num_incoming = len(parameters)
        full_len = len(self._full_lora_param_names)
        b_len = len(self._b_only_param_names)

        if num_incoming == full_len:
            # Round 1: load A and B
            state = self.model.state_dict()
            for name, arr in zip(self._full_lora_param_names, parameters):
                tensor = np.array(arr)
                state[name] = state[name].new_tensor(tensor)
            self.model.load_state_dict(state, strict=False)
            self._freeze_all_A()
            log(INFO, f"Client {self.client_id}: Loaded full LoRA params (A+B) and froze A.")

        elif num_incoming == b_len:
            # Round >1: load only B
            state = self.model.state_dict()
            for name, arr in zip(self._b_only_param_names, parameters):
                tensor = np.array(arr)
                state[name] = state[name].new_tensor(tensor)
            self.model.load_state_dict(state, strict=False)
            self._freeze_all_A()
            log(INFO, f"Client {self.client_id}: Loaded B-only params and kept A frozen.")

        else:
            raise ValueError(
                f"Client {self.client_id}: Unexpected parameters length {num_incoming}. "
                f"Expected full_len={full_len} or b_len={b_len}."
            )

    def get_parameters(self, config):
        """Return parameters based on round:

        - Round 1 (config['round']==1): full [A1,B1,...]
        - Round >1: B-only [B1,B2,...]

        If 'round' not provided, default to B-only (safer for comms).
        """
        server_round = None
        if isinstance(config, dict):
            server_round = config.get("round") or config.get("server_round")

        # Extract tensors from state_dict in the same order
        state = self.model.state_dict()

        if server_round == 1:
            names = self._full_lora_param_names
        else:
            names = self._b_only_param_names

        params_to_send = [state[n].cpu().numpy() for n in names]
        return params_to_send

