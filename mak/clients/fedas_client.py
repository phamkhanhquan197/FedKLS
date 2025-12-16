import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient


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
        criterion = self.get_loss(loss=self.config_sim["client"]["loss"]) if "client" in self.config_sim else None

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
                if criterion is None:
                    criterion = self.get_loss(loss=self.config_sim["client"]["loss"])
                preds = self.model(images)
                loss = criterion(preds, labels)

            loss.backward()

            # accumulate squared gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().cpu().numpy()
                    total_trace += float((g ** 2).sum())

            batches += 1

        # average over number of batches used
        if batches > 0:
            return float(total_trace / batches)
        return 0.0

    def fit(self, parameters, config):
        # Apply incoming parameters
        self.set_parameters(parameters)

        # Read prev state from server (ndarrays) if provided
        prev_state = None
        prev_version = None
        if isinstance(config, dict):
            prev_state = config.get("fedas.prev_state_ndarrays", None)
            prev_version = config.get("fedas.prev_state_version", None)

        # Store prev state locally for any client-side logic that needs it
        self.prev_state = prev_state
        self.prev_state_version = prev_version

        # Compute approximate FIM-trace on a small subset of validation data
        try:
            fim_trace = self.compute_fim_trace(max_batches=1)
        except Exception:
            fim_trace = 1.0

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

        # After training return updated parameters and include the FIM weight in metrics
        metrics = {"client_id": self.client_id, "fedas.fim_trace": float(fim_trace)}
        if prev_version is not None:
            metrics["fedas.prev_state_version"] = prev_version

        return self.get_parameters({}), len(trainloader.dataset), metrics
