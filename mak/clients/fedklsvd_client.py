import torch
from flwr.common.logger import log
from logging import INFO
from collections import OrderedDict
from mak.clients.base_client import BaseClient

class FedKLSVDClient(BaseClient):
    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm
    ):
        super().__init__(client_id, model, trainset, valset, config_sim, device, save_dir)  
        self.kl_norm = kl_norm  # Normalized KL divergence for this client
        # log(INFO, f"Client {self.client_id}: Initialized with kl_norm = {self.kl_norm}")
        
    def __repr__(self) -> str:
        return "FedKLSVD client"
    
    def get_parameters(self, config):
        """Extracts the A, B, and bias parameters from the SVD model to send to the server."""
        self.model.eval()  # Ensure model is in eval mode
        params_dict = self.model.state_dict()
        ab_params = []
        for name, tensor in params_dict.items():
            if "lin" in name:
                ab_params.append(tensor.detach().cpu().numpy())
        # log(INFO, f"Client {self.client_id}: Parameters sent to server: {[name for name in params_dict.keys() if 'lin' in name]}")
        if not ab_params:
            log(INFO, f"Client {self.client_id}: No A, B, or bias parameters found to send!")
        return ab_params
    
    def set_parameters(self, parameters):
        """Sets the A, B, and bias parameters to the SVD model at the client."""
        # log(INFO, f"Client {self.client_id}: Insert global A, B, and bias parameters into SVD model")
        model_state = self.model.state_dict()
        ab_keys = [k for k in model_state.keys() if "lin" in k]

        #Check if the server sent the full SVD model (first round) or just only SVDAdapter parameters (subsequent rounds)
        if len(model_state) == len(parameters): #First round
            # log(INFO, f"Client {self.client_id}: Received full SVD model parameters with ({len(parameters)} layers)")
            params_dict = zip(model_state.keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        else: 
            # log(INFO, f"Client {self.client_id}: Updating SVDAdapter parameters (A, B, bias)")
            # Check for parameter mismatch
            if len(ab_keys) != len(parameters):
                log(INFO, f"Client {self.client_id}: Mismatch in parameter counts! Expected {len(ab_keys)}, received {len(parameters)}")
                raise ValueError("Parameter count mismatch between client and server")
            
            else:
                # Create state dict with only LoRA parameters
                ab_params  = OrderedDict()
                for key, array in zip(ab_keys, parameters):
                    ab_params [key] = torch.from_numpy(array)
                
                # Update model with LoRA parameters only
                model_state.update(ab_params )
                self.model.load_state_dict(model_state, strict=False)

    #Overwrite fit function from BaseClient to include kl_norm in the metrics
    def fit(self, parameters, config):
        # Call the parent class's fit method
        result = super().fit(parameters, config)
        # result is (parameters, num_examples, metrics)
        parameters, num_examples, metrics = result
        # Add kl_norm to metrics
        metrics["kl_norm"] = self.kl_norm
        return parameters, num_examples, metrics
    




