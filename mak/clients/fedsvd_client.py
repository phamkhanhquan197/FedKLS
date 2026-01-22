from mak.clients.base_client import BaseClient

class FedSVDClient(BaseClient):
    """FedSVD Client - supports both FedAvg and FFA modes.
    
    - FedAvg mode: Send both A and B matrices (like standard FedAvg on LoRA params)
    - FFA mode: Send only B matrices (A is frozen, like FFA-LoRA)
    
    The mode is determined from config_sim["fedsvd_config"]["mode"].
    """
    
    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, 
        kl_norm=None, dataset=None, apply_transforms=None, data_scheduler=None, 
        bias=None, rank_policy_map=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir, 
            dataset=dataset, apply_transforms=apply_transforms, 
            data_scheduler=data_scheduler, bias=bias, rank_policy_map=rank_policy_map
        )
        
        # Get FedSVD mode from config
        self.fedsvd_mode = config_sim.get("fedsvd_config", {}).get("mode", "fedavg")

    def __repr__(self) -> str:
        return f"FedSVD client (mode={self.fedsvd_mode})"

    def get_parameters(self, config):
        """
        Send parameters based on FedSVD mode:
        - FedAvg mode: Send both A and B (like standard LoRA)
        - FFA mode: Send only B (A is frozen)
        """
        if not self.config_sim["peft"]["enabled"]:
            # If PEFT is disabled, send full model parameters
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        model_state = self.model.state_dict()
        
        # FFA mode: Send only B matrices (like FFA-LoRA)
        if self.fedsvd_mode == "ffa":
            if any(key.startswith("distilbert.") for key in model_state.keys()):
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
            elif any(key.startswith("roberta.") for key in model_state.keys()):
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
            elif any(key.startswith("bert.") for key in model_state.keys()):
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
            elif self.config_sim["common"]["model"] in ["Resnet18", "Resnet34", "ResNet18Pretrained", "ResNet34Pretrained"]:
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items()
                        if name.endswith(".B") or (name.endswith(".bias") and "conv" in name)
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items()
                        if name.endswith(".B")
                    }
            else:
                raise NotImplementedError(f"FedSVD FFA mode: Unsupported model type {self.config_sim['common']['model']}")
        
        # FedAvg mode: Send both A and B matrices (standard LoRA approach)
        else:
            if any(key.startswith("distilbert.") for key in model_state.keys()):
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if "lin" in name
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if name.endswith(".B") or name.endswith(".A")
                    }
            elif any(key.startswith("roberta.") for key in model_state.keys()):
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if "self" in name or ("dense" in name and "classifier" not in name)
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if name.endswith(".B") or name.endswith(".A")
                    }
            elif any(key.startswith("bert.") for key in model_state.keys()):
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if "self" in name or "dense" in name
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if name.endswith(".B") or name.endswith(".A")
                    }
            elif any(key.startswith("model.") for key in model_state.keys()):
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if "self_attn" in name or "mlp" in name
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if name.endswith(".B") or name.endswith(".A")
                    }
            elif self.config_sim["common"]["model"] in ["Resnet18", "Resnet34", "ResNet18Pretrained", "ResNet34Pretrained"]:
                if self.bias:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if "conv" in name
                    }
                else:
                    params_to_send = {
                        name: tensor for name, tensor in model_state.items() 
                        if name.endswith(".B") or name.endswith(".A")
                    }
            else:
                raise ValueError(f"FedSVD FedAvg mode: PEFT parameter extraction not defined for model {self.config_sim['common']['model']}")

        # Return parameters in deterministic sorted order
        return [
            tensor.cpu().numpy()
            for _, tensor in sorted(params_to_send.items())
        ]
