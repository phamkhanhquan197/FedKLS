import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.utils.general import set_params
from mak.utils.dataset_info import dataset_info


class FedASClient(BaseClient):
    """FedAS client adapter.

    Behavior:
    - Read prev model state ndarrays from FitIns.config["fedas.prev_state_ndarrays"] if present.
    - Compute an approximate FIM-trace on a small number of validation batches before local training.
    - Run the usual local training (reusing BaseClient.train).
    - Return FitRes with metrics including "fedas.fim_trace" and optionally echo the prev_state_version.
    """

    def compute_fim_trace(self, max_batches: int = 1) -> float:
        """Approximate the trace of the Fisher Information Matrix by summing squared gradients

        Uses the validation loader for a small number of batches.
        """
        self.model.eval()
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        criterion = torch.nn.NLLLoss()

        total_trace = 0.0
        batches = 0
        for batch in valloader:
            if batches >= max_batches:
                break
            self.model.zero_grad()
            if self.feature_key == "text" or self.feature_key == "content":
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(self.device), batch[y_label].to(self.device)
                preds = self.model(images)
                loss = criterion(preds, labels)

            loss.backward()

            # accumulate squared gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().cpu().numpy()
                    total_trace += float((g ** 2).sum())

        return total_trace

    def align_federated_parameters(self):
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        self.dataset.train()

        prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)

                for y, feat in zip(y, features):
                    prototypes[y].append(feat)

        mean_prototypes = [
            torch.stack(prototype).mean(dim=0) if prototype else None
            for prototype in prototypes
        ]

        alignment_optimizer = torch.optim.SGD(
            self.model.base.parameters(), lr=self.args.fedas.alignment_lr
        )

        for _ in range(self.args.fedas.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x, detach=False)
                loss = 0
                for label in y.unique().tolist():
                    if mean_prototypes[label] is not None:
                        loss += F.mse_loss(
                            features[y == label].mean(dim=0), mean_prototypes[label]
                        )

                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        self.prev_model.cpu()

    def fit(self, parameters, config):
        # Preserve current client model (local) and build prev_model from server-sent parameters
        local_model = copy.deepcopy(self.model)
        
        # Read prev state from server (ndarrays) if provided
        prev_state = None
        prev_version = None
        if isinstance(config, dict):
            prev_state = config.get("fedas.prev_state_ndarrays", None)
            prev_version = config.get("fedas.prev_state_version", None)

        # Store prev state metadata
        self.prev_state = prev_state
        self.prev_state_version = prev_version

        # If a prev_state was provided, align our local model base to the prev_model's prototypes
        fedas_conf = self.config_sim.get("fedas", {}) if isinstance(self.config_sim, dict) else {}
        alignment_lr = fedas_conf.get("alignment_lr", 0.0)
        alignment_epoch = fedas_conf.get("alignment_epoch", 0)
        if (prev_state is not None or parameters is not None) and alignment_lr > 0 and alignment_epoch > 0:
            try:
                # align the local_model towards prev_model prototypes, then use the aligned local as self.model
                self.align_federated_parameters(prev_model=prev_model, target_model=local_model, alignment_lr=alignment_lr, alignment_epoch=alignment_epoch)
                # replace current model with the aligned local model (do not overwrite with server global)
                self.model = local_model
            except Exception:
                # alignment is optional; failures should not crash the training
                pass

        # Proceed with normal training using parent's training helper
        # Build trainloader
        from torch.utils.data import DataLoader as _DL

        batch, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )
        trainloader = _DL(self.trainset, batch_size=batch, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Call the BaseClient.train implementation
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=self.optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
            scheduler=None,
        )

        # Compute approximate FIM-trace on a small subset of validation data
        try:
            fim_trace = self.compute_fim_trace(max_batches=1)
        except Exception:
            fim_trace = 1.0

        # After training return updated parameters and include the FIM weight in metrics
        metrics = {"client_id": self.client_id, "fedas.fim_trace": float(fim_trace)}
        if prev_version is not None:
            metrics["fedas.prev_state_version"] = prev_version

        return self.get_parameters({}), len(trainloader.dataset), metrics
