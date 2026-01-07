# mak/clients/fedsa_lora_client.py
from mak.clients.base_client import BaseClient


class FedSALoRAClient(BaseClient):
    """
    FedSA-LoRA Client:
    - Uplink: only send LoRA A (and optional bias if enabled)
    - Downlink: handled by set_params(method="fedsa_lora") in mak.utils.general
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir,
        kl_norm=None, dataset=None, apply_transforms=None, data_scheduler=None, bias=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir,
            dataset=dataset, apply_transforms=apply_transforms, data_scheduler=data_scheduler, bias=bias
        )

    def __repr__(self) -> str:
        return "FedSA-LoRA client"

    def get_parameters(self, config):
        """
        Only send A adapters to the server.
        Deterministic order: sort keys.
        """
        model_state = self.model.state_dict()

        if any(key.startswith("distilbert.") for key in model_state.keys()):
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".A") or (name.endswith(".bias") and "lin" in name)
                }
            else:
                params_to_send = {name: tensor for name, tensor in model_state.items() if name.endswith(".A")}

        elif any(key.startswith("bert.") for key in model_state.keys()):
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".A")
                    or (name.endswith(".bias") and "self" in name)
                    or (name.endswith(".bias") and "dense" in name)
                }
            else:
                params_to_send = {name: tensor for name, tensor in model_state.items() if name.endswith(".A")}

        elif any(key.startswith("model.") for key in model_state.keys()):
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if (
                        name.endswith(".A")
                        or (name.endswith(".bias") and "self_attn" in name)
                        or (name.endswith(".bias") and "mlp" in name)
                    )
                }
            else:
                params_to_send = {name: tensor for name, tensor in model_state.items() if name.endswith(".A")}

        else:
            raise NotImplementedError("Unsupported model type for FedSA-LoRA")

        return [tensor.cpu().numpy() for _, tensor in sorted(params_to_send.items())]
