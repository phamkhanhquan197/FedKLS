import copy
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient


class PFedEditClient(BaseClient):
    """Client implementing PFedEdit-style private-layer selection.

    Usage: create with the same args as BaseClient plus `module_name_list` (list of candidate module names)
    and `num_layer` (how many private layers to keep).
    After local training, call `casual_trace(data_loader)` to compute the most-important private layers
    and set `self.layer_name` accordingly. The existing BaseClient will include `kept_indices` in metrics.
    """

    def __init__(
        self,
        client_id,
        model,
        trainset,
        valset,
        config_sim,
        device,
        save_dir,
        num_layer: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            client_id=client_id,
            model=model,
            trainset=trainset,
            valset=valset,
            config_sim=config_sim,
            device=device,
            save_dir=save_dir,
        )
        self.num_layer = num_layer
        self.module_name_list = self.get_model_list(model=model)
        self.previous_iter_model_weight = copy.deepcopy(self.model)

    def set_parameters(self, parameters):
        self.set_previous_local_weights()
        layers = self.casual_trace(self.valset)
        for k in range(len(layers)):
            print(f" round {round+1} user {i} repalced layer: {layers[k]}")
            self.model = self.recover_from_clean_model(self.model, layers[k])
        set_params(self.model, parameters)

    def set_previous_local_weights(self):
        # record previous local training weights
        for key, value in self.model.state_dict().items():
            self.previous_iter_model_weight.state_dict()[key].data.copy_(self.model.state_dict()[key])

    def recover_from_clean_model(self, model, module_name):
        module = self.get_module(module_name)
        setattr(model, module_name, module)
        return model

    def get_module(self, name):
        for n,m in self.previous_iter_model_weight.named_modules():
            if n == name:return m.to(self.device)      #same if using copy.deepcopy() or not
        raise LookupError(name)

    def hook_to_cpu(self):
        for key in list(self.model_hook.keys()):
            val = self.model_hook[key]
            if isinstance(val, tuple):
                # move tuple elements to cpu
                self.model_hook[key] = tuple(
                    v.detach().cpu() if hasattr(v, "detach") else v for v in val
                )
            elif val is not None and hasattr(val, "detach"):
                self.model_hook[key] = val.detach().cpu()

    def eval_model_with_hook(self, model: torch.nn.Module, test_loader: DataLoader, bias: List[float], recover: bool, recovered_name: str = None):
        model.to(self.device)
        model.eval()

        if recover:
            for name, module in self.previous_iter_model_weight.named_modules():
                if recovered_name in name:
                    setattr(model, name, module.to(self.device))
            gt_match_list = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, y_pred = model(x)
                bias.append(self.compute_st_bias(y_pred, y))
                if recover:
                    for a, b in zip(y_pred, y):
                        gt_match_list.append(True) if torch.argmax(a) == b else gt_match_list.append(False)
                del y_pred, logits
        model.to("cpu")
        if recover:
            return bias, gt_match_list
        else:
            return bias

    @staticmethod
    def compute_st_bias(prob: torch.Tensor, gt_label: torch.Tensor) -> float:
        gt_label = gt_label.detach().cpu()
        prob = prob.detach().cpu()
        bias = 0.0
        for i in range(prob.shape[0]):
            bias += prob[i][gt_label[i]]
        return float(bias / prob.shape[0])

    def casual_trace(self, data_loader: DataLoader):
        """Compute a simple scoring for candidate modules and set `self.layer_name` to top-k.

        This is a lightweight port of PFedEdit.casual_trace. It expects `self.module_name_list` to be populated
        with candidate module names. It uses `self.previous_iter_model_weight` and `self.model` to compute
        a bias-based effect score per candidate and selects the top `self.num_layer` modules.
        """
        if not self.module_name_list:
            return

        clean_local_bias = self.eval_model_with_hook(model=self.previous_iter_model_weight, test_loader=data_loader, recover=False, bias=[])

        total_effect = {}
        for i, name in enumerate(self.module_name_list):
            recovered_bias, gt_match_list = self.eval_model_with_hook(model=copy.deepcopy(self.model), test_loader=data_loader, recover=True, bias=[], recovered_name=name)
            # ratio differences
            total_effect[name] = [recovered_bias[x] / clean_local_bias[x] - 1 for x in range(len(clean_local_bias))]

            tf_list = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
            for val, mask in zip(total_effect[name], gt_match_list):
                if mask and val > 0:
                    tf_list["A"] += 1
                elif mask and val < 0:
                    tf_list["B"] += 1
                elif not mask and val > 0:
                    tf_list["C"] += 1
                elif not mask and val < 0:
                    tf_list["E"] += 1
            tf_list["D"] += sum(total_effect[name]) / len(clean_local_bias)
            total_effect[name] = tf_list

        # sort by tuple (A,B,C,D,E)
        def custom_sort(data):
            return (data[1]["A"], data[1]["B"], data[1]["C"], data[1]["D"], data[1]["E"])

        sorted_effect = sorted(total_effect.items(), key=custom_sort, reverse=True)
        chosen = [name for name, _ in sorted_effect[: self.num_layer]]
        self.layer_name = chosen
        return chosen

    # Helper to snapshot previous local weights
    def set_previous_local_weights(self):
        for key, value in self.model.state_dict().items():
            self.previous_iter_model_weight.state_dict()[key].data.copy_(self.model.state_dict()[key])

    @staticmethod
    def get_model_list(model):
        model_name = model.__class__.__name__
        if "vit" in model_name.lower():
            module_name_list = get_sub_ViT_module_name(model)
        elif "resnet18" in model_name.lower():
            module_name_list = get_sub_ResNet_module_name(model)
        elif "mlp" in model_name.lower():
            module_name_list = get_MLP_module_name(model)
        elif "vgg_11" in model_name.lower():
            module_name_list = get_sub_VGG_module_name(model)
        else:
            module_name_list = []
            print("No matching model found for pfededit_client module extraction.")
        return module_name_list
    
    @staticmethod
    def get_sub_ViT_module_name(model):
        name_list = []
        for i, _ in model.named_modules():
            if i == "model.conv_proj" or i == "model.encoder.ln" or i == "model.heads.head":
                name_list.append(i)
            elif len(i.split(".")) > 4 and "dropout" not in i:    #and "dropout" not in i
                if i.split(".")[-1]!= "mlp":name_list.append(i)
        return name_list

    @staticmethod
    def get_sub_ResNet_module_name(model):
        name_list = []
        for i, _ in model.named_modules():
            if len(i.split(".")) < 4:
                if i in ["backbone.avgpool", "backbone.conv1", "backbone.bn1", "backbone.relu", "backbone.maxpool"]:
                    name_list.append(i)
            else:
                name_list.append(i)
        return name_list

    @staticmethod
    def get_sub_VGG_module_name(model):
        name_list = []
        for i,_ in model.named_modules():
            if i not in ["", "network", "linear_layers"]:
                name_list.append(i)
        return name_list

    @staticmethod
    def get_MLP_module_name(model):
        name_list = []
        for x, _ in model.named_modules():
            if x != "": name_list.append(x)
        return name_list

    @staticmethod
    def get_top_VIT_module_name(model):
        name_list = []
        for i, _ in model.named_modules():
            if len(i.split(".")) <= 4:
                if i not in ["", "model", "model.encoder.dropout", "model.heads.head", "model.encoder.ln","model.encoder.layers"]:
                    name_list.append(i)
            elif i.split(".")[-1] == "mlp":
                name_list.append(i)
        return name_list