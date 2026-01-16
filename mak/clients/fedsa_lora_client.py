from mak.clients.base_client import BaseClient

class FFALoRAClient(BaseClient):
    """FFA-LoRA Client (Phase 1) - deterministic name-based mapping.

    Standards:
    - Communication is based on model.state_dict().
    - Client uplink ALWAYS returns PARTIAL tensors using get_target_keys(model).
    - Round 1 downlink is FULL state_dict values (handled via BaseClient super().set_parameters).
    - Round >1 downlink is PARTIAL tensors aligned with the same sorted keys.
    """
    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None, dataset=None, apply_transforms=None, data_scheduler=None, bias=None, rank_policy_map=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir, dataset=dataset, apply_transforms=apply_transforms, data_scheduler=data_scheduler, bias=bias, rank_policy_map=rank_policy_map   
        )

    def __repr__(self) -> str:
        return "FFA-LoRA client"

    def get_parameters(self, config):
        """
        Only send B adapters to the server.
        """
        model_state = self.model.state_dict()

        if any(key.startswith("distilbert.") for key in model_state.keys()):  # DistilBERT-based model
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B") or (name.endswith(".bias") and "lin" in name)
                }
            else:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B")
                }
        elif any(key.startswith("roberta.") for key in model_state.keys()):  # RoBERTa-based model
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B")
                    or (name.endswith(".bias") and "self" in name)
                    or (name.endswith(".bias") and "dense" in name and "classifier" not in name)
                }
            else:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B")
                }
        elif any(key.startswith("bert.") for key in model_state.keys()):  # BERT-based model
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B")
                    or (name.endswith(".bias") and "self" in name)
                    or (name.endswith(".bias") and "dense" in name)
                }
            else:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if name.endswith(".B")
                }

        elif any(key.startswith("model.") for key in model_state.keys()):
            if self.bias:
                params_to_send = {
                    name: tensor for name, tensor in model_state.items()
                    if (
                        name.endswith(".B")
                        or (name.endswith(".bias") and "self_attn" in name)
                        or (name.endswith(".bias") and "mlp" in name)
                    )
                }
            else:
                params_to_send = {
                name: tensor for name, tensor in model_state.items()
                if name.endswith(".B")
                }

        else:
            raise NotImplementedError("Unsupported model type for FFA-LoRA")

        # Minimal but critical fix: enforce deterministic order
        return [
            tensor.cpu().numpy()
            for _, tensor in sorted(params_to_send.items())]
