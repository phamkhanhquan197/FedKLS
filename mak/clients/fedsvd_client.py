from mak.clients.base_client import BaseClient
from collections import OrderedDict
import torch

class FedSVDClient(BaseClient):
    """FedSVD Client - supports both FedAvg and FFA modes.
    
    Convention: A(r, in), B(out, r) matching PEFT exactly.
    Forward: ΔW = B @ A
    
    - FedAvg mode: Send delta of both A and B matrices
    - FFA mode: Freeze A, train B, send delta of B
    
    Matches 3rd-party fed-svd implementation:
    - Downlink: Full model (A+B)
    - Uplink: Delta (current - initial)
    - Server: Aggregate delta and apply to base model
    
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
        
        # Store initial state dict when receiving from server (for delta computation)
        self.init_state_dict = None
        
        # Track current round for SVD reinitialization detection
        self._current_round = 0

    def __repr__(self) -> str:
        return f"FedSVD client (mode={self.fedsvd_mode})"

    def _configure_trainable_parameters(self) -> None:
        """Freeze backbone; train only adapter params.

        This mirrors 3rd-party fed-svd+PEFT behavior (only LoRA parameters are
        trainable) and avoids Opacus DP failures caused by mixing unsupported
        trainable parameters.
        """
        trainable_names: set[str] = set()

        # Prefer explicit adapter module detection when available
        try:
            from mak.models.svd_model import SVDAdapter, ConvAdapter

            for module_name, module in self.model.named_modules():
                if isinstance(module, SVDAdapter):
                    trainable_names.add(f"{module_name}.A")
                    trainable_names.add(f"{module_name}.B")
                    if self.bias and getattr(module, "bias", None) is not None:
                        trainable_names.add(f"{module_name}.bias")
                elif isinstance(module, ConvAdapter):
                    trainable_names.add(f"{module_name}.A")
                    trainable_names.add(f"{module_name}.B")
        except Exception:
            # Fallback: name-based selection
            pass

        for name, param in self.model.named_parameters():
            # Fallback path: if we couldn't detect adapters, still try to only train A/B/bias
            if not trainable_names:
                is_adapter = name.endswith(".A") or name.endswith(".B")
                is_bias = self.bias and name.endswith(".bias")
                if not (is_adapter or is_bias):
                    param.requires_grad = False
                    continue
            else:
                if name not in trainable_names:
                    param.requires_grad = False
                    continue

            # Now within trainable set
            if self.fedsvd_mode == "ffa" and name.endswith(".A"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Safety net: if we ended up freezing everything, recover by unfreezing
        # adapter parameters by name. This prevents crashes in optimizer setup.
        num_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        if num_trainable == 0:
            has_adapter_params = any(
                n.endswith(".A") or n.endswith(".B") or n.endswith(".bias")
                for n, _p in self.model.named_parameters()
            )
            if not has_adapter_params:
                raise RuntimeError(
                    "FedSVD expected an SVD-adapted model with adapter parameters (.A/.B), "
                    "but none were found on the client model. Ensure client_model is the SVD/LoRA-adapted model "
                    "(not the base model) when using strategy=FedSVD."
                )

            # Unfreeze LoRA/SVD params as a fallback
            for n, p in self.model.named_parameters():
                if n.endswith(".B"):
                    p.requires_grad = True
                elif self.fedsvd_mode != "ffa" and n.endswith(".A"):
                    p.requires_grad = True
                elif self.bias and n.endswith(".bias"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
    
    def set_parameters(self, parameters):
        """Override to match 3rd-party Fed-SVD behavior exactly.
        
        3rd-party logic (misc/utils.py line 97-140):
        - FedAvg: Load all (A + B)
        - FFA (no SVD reinit): FILTER to load ONLY B, keep A unchanged
        - FFA (with SVD reinit): Load all (A + B reinit)
        
        W_res: Direct attribute → not transmitted → always unchanged
        """
        from mak.utils.general import set_params
        
        model_state = self.model.state_dict()
        
        # Check if SVD reinitialization happened
        recalculate_svd_period = self.config_sim.get("fedsvd_config", {}).get("recalculate_svd_period", 0)
        svd_warmup_steps = self.config_sim.get("fedsvd_config", {}).get("svd_warmup_steps", 0)
        is_svd_reinit = (recalculate_svd_period > 0 and 
                        self._current_round > svd_warmup_steps and
                        (self._current_round % recalculate_svd_period) == 0)
        
        # Matching 3rd-party logic (misc/utils.py line 113-119):
        # - if model_type == 'fedavg': receive_all = True
        # - elif args.recalculate_svd_period: receive_all = True  
        # - elif round == 1: receive_all = True (first initialization)
        # - else (ffa): filter to load only B
        
        if self.fedsvd_mode == "fedavg":
            # FedAvg mode: Always load all
            print(f"[FedSVDClient {self.client_id}] FedAvg mode - Loading all (A + B)")
            set_params(self.model, parameters, device=self.device, 
                      method=None, bias=self.bias)
        
        elif is_svd_reinit:
            # SVD reinit: Load all (even in FFA mode)
            print(f"[FedSVDClient {self.client_id}] Server re-calculated SVD - Loading all (A + B)")
            set_params(self.model, parameters, device=self.device, 
                      method=None, bias=self.bias)
        
        elif self._current_round == 1:
            # Round 1: First initialization - load all (A + B)
            print(f"[FedSVDClient {self.client_id}] Round 1 initialization - Loading all (A + B)")
            set_params(self.model, parameters, device=self.device, 
                      method=None, bias=self.bias)
        
        elif self.fedsvd_mode == "ffa":
            # FFA mode (no SVD reinit): FILTER to load only B
            print(f"[FedSVDClient {self.client_id}] FFA mode - Filtering to load ONLY B (keep A unchanged)")
            
            # Build mapping: parameter name → received array
            sorted_keys = sorted(model_state.keys())
            params_dict = dict(zip(sorted_keys, parameters))
            
            # Filter: Keep only B (matching 3rd-party logic)
            filtered_state = OrderedDict()
            for name in sorted_keys:
                if name.endswith(".B"):
                    # Load B from server
                    filtered_state[name] = torch.tensor(params_dict[name], device=self.device)
                    print(f"  ✅ Loading: {name}")
                elif self.bias and "bias" in name:
                    # Load bias if enabled
                    if any(kw in name for kw in ["lin", "self", "dense", "conv", "mlp", "self_attn"]):
                        filtered_state[name] = torch.tensor(params_dict[name], device=self.device)
                        print(f"  ✅ Loading: {name}")
                elif name.endswith(".A"):
                    # Skip A (keep unchanged)
                    print(f"  ⏭️  Skipping (keep unchanged): {name}")
            
            # Update model with filtered parameters
            model_state.update(filtered_state)
            self.model.load_state_dict(model_state, strict=False)
            print(f"  → Loaded {len(filtered_state)} parameters, kept {len(sorted_keys) - len(filtered_state)} unchanged")
        
        # Save initial state dict for delta computation
        self.init_state_dict = OrderedDict()
        for name, param in self.model.state_dict().items():
            if self._should_track_for_delta(name):
                self.init_state_dict[name] = param.clone().detach().cpu()

        # Freeze backbone and keep only adapter params trainable (3rd-party behavior)
        self._configure_trainable_parameters()

    def _should_track_for_delta(self, param_name):
        """Check if parameter should be tracked for delta computation.
        
        We track all LoRA/SVD parameters: A, B, and bias.
        """
        # Track if it ends with .A, .B
        if param_name.endswith(".A") or param_name.endswith(".B"):
            return True
        # Track bias if enabled
        if self.bias and "bias" in param_name:
            # Check if it's a LoRA-related bias
            if any(keyword in param_name for keyword in ["lin", "self", "dense", "conv", "mlp", "self_attn"]):
                return True
        return False
    
    def _should_send_parameter(self, param_name):
        """Determine if a parameter should be sent based on mode.
        
        - FedAvg mode: Send all LoRA parameters (A + B + bias)
        - FFA mode: Send only B matrices (+ bias), A is frozen
        """
        if self.fedsvd_mode == "ffa":
            # FFA mode: only send B matrices and bias (A is frozen)
            if param_name.endswith(".B"):
                return True
            if self.bias and "bias" in param_name:
                if any(keyword in param_name for keyword in ["lin", "self", "dense", "conv", "mlp", "self_attn"]):
                    return True
            return False
        else:
            # FedAvg mode: send all LoRA parameters
            return self._should_track_for_delta(param_name)

    def get_parameters(self, config=None):
        """Return delta (current - initial) of parameters based on mode.
        
        This matches 3rd-party fed-svd implementation:
        - Client sends: theta_diff = {k: (state_dict[k] - init_state_dict[k]) for k in keys}
        - Server aggregates deltas and applies to base model
        """
        if self.init_state_dict is None:
            raise ValueError(
                "init_state_dict is None. Make sure set_parameters() was called before get_parameters()."
            )
        
        current_state = self.model.state_dict()
        
        # Compute deltas for parameters that should be sent
        delta_dict = OrderedDict()
        for name in sorted(current_state.keys()):
            if self._should_send_parameter(name):
                if name not in self.init_state_dict:
                    raise KeyError(
                        f"Parameter {name} not found in init_state_dict. "
                        f"This should not happen - check _should_track_for_delta()."
                    )
                # Compute delta: current - initial
                delta = current_state[name].cpu() - self.init_state_dict[name]
                delta_dict[name] = delta
        
        # Debug: Log delta statistics
        if len(delta_dict) > 0:
            first_key = next(iter(delta_dict.keys()))
            first_delta = delta_dict[first_key]
            print(f"[FedSVDClient {self.client_id}] Sending {len(delta_dict)} deltas. "
                  f"First delta '{first_key[:50]}': shape={first_delta.shape}, "
                  f"mean={first_delta.mean():.6f}, std={first_delta.std():.6f}, "
                  f"max_abs={first_delta.abs().max():.6f}")
        
        # Return as list of numpy arrays (sorted by key for determinism)
        return [tensor.numpy() for tensor in delta_dict.values()]

    def fit(self, parameters, config):
        """Override to track round number for SVD reinitialization detection."""
        # Update round number from config (Flower passes "current_round")
        self._current_round = config.get("current_round", config.get("server_round", 0))
        
        # Call parent's fit method
        return super().fit(parameters, config)

    def evaluate(self, parameters, config):
        """Override to track round number for SVD reinitialization detection."""
        # Update round number from config before set_parameters is called
        self._current_round = config.get("current_round", config.get("server_round", config.get("round", 0)))
        
        # Call parent's evaluate method
        return super().evaluate(parameters, config)
