from typing import Dict
from flwr.common import Scalar

from mak.clients.base_client import BaseClient


class PFedMoAPClient(BaseClient):
    def __repr__(self):
        return "PFedMoAPClient"

    def fit(self, parameters, config: Dict[str, Scalar]):
        # 1. load global prompt
        global_prompt = parameters[0]
        self.model.set_prompt(global_prompt)

        # 2. load non-local experts
        if config.get("has_experts", False):
            self.model.load_nonlocal_prompts(config["experts"])
        else:
            self.model.clear_nonlocal()

        # 3. local training
        trainloader = self.get_trainloader(config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=None,
            epochs=config["epochs"],
            device=self.device,
            config=config,
        )

        # 4. return updated prompt
        new_prompt = self.model.get_prompt()
        return (
            [new_prompt],
            len(trainloader.dataset),
            {},
        )
