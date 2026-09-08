from mak.models.clip_backbone import model
import numpy as np
import torch
import os
from .base_client import BaseClient
from torch.utils.data import DataLoader
from mak.clients.base_client import BaseClient
from mak.utils.helper import get_rank_candidates, get_optimizer
from logging import DEBUG, INFO
import json
class FedSpecClient(BaseClient):
    """FedSpec Client
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ===== Q-learning =====
        self.alpha = self.config_sim["fedspec_config"]["rl_agent"]["alpha"]
        self.gamma = self.config_sim["fedspec_config"]["rl_agent"]["gamma"]
        self.max_epsilon = self.config_sim["fedspec_config"]["rl_agent"]["max_epsilon"]
        self.min_epsilon = self.config_sim["fedspec_config"]["rl_agent"]["min_epsilon"]
        self.epsilon_decay = self.config_sim["fedspec_config"]["rl_agent"]["epsilon_decay"]

        # ===== Q-learning: detect state/action space & env =====
        max_rank = float("inf")
        for name, layer in self.model.named_parameters():
            if name.endswith(".A"):
                m, r = layer.shape
                max_rank = min(max_rank, m)
            elif name.endswith(".B"):
                r, n = layer.shape
                max_rank = min(max_rank, n)

        # Compute action/state candidates
        self.action_space, self.state_space = get_rank_candidates(max_rank)

        # Construct environment mapping state -> possible next states
        self.env_space = self.env_dict(self.state_space, self.action_space)

        # ===== Q-table path inside experiment folder =====
        if self.save_dir is None:
            raise ValueError("save_dir must be provided")

        self.qtable_dir = os.path.join(self.save_dir, "client_qtable")
        os.makedirs(self.qtable_dir, exist_ok=True)

        self.q_table_file = os.path.join(
            self.qtable_dir,
            f"client_{self.client_id}_q_table.json"
        )

        # Load or initialize Q-table
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "r") as f:
                self.q_table = json.load(f)
        else:
            self.q_table = [['x'] * len(self.action_space) for _ in range(len(self.state_space))]
            self.q_table = self.normalize_q_table(self.q_table, self.env_space)



    def __repr__(self) -> str:
        return "FedSpec client"
    
    def env_dict(self, state_space, action_space):
        env_space = {}
        for state in state_space:
            env = [state + action if state + action in state_space else "x" for action in action_space] #Checking all possible next state after perform 8 actions, if the next state does not exist in the distribution, marked as 'x'
            env_space[state] = env
        return env_space

    def normalize_q_table(self, q_table, env_space): #This function helps to convert the q-table under env_space format where updatable positions will have 0.0 values, otherwise, 'x'
        count = 0
        for _, next_state_list in env_space.items():
            next_state_list = [0.0 if i != 'x' else i for i in next_state_list]
            q_table[count] = next_state_list
            count += 1
        return q_table

    def greedy_policy(self, q_table, current_state_index, next_state_list):
        #Return the index has the highest Q-values in the Q-table (the current_state_index row)
        highest_Q_values_index = max((i for i, item in enumerate(q_table[current_state_index]) if item != 'x'), key=lambda i: self.q_table[current_state_index][i])
        next_state = next_state_list[highest_Q_values_index]
        #print(f"current_state: {current_state}, next_state-list: {next_state_list}, q_table: {q_table[current_state_index]}, highest_Q_values_index: {highest_Q_values_index}, next_state: {next_state}")
        return next_state

    def epsilon_greedy_policy(self, q_table, next_state_list, current_state_index, epsilon):
        rand_n = float(np.random.uniform(0, 1))

        if rand_n > epsilon: #If the current row of the Q-table is all-zeros elements => Random action
            #print("Highest Q-values action")
            next_state = self.greedy_policy(q_table, current_state_index, next_state_list) #return possible next_state : float with highest Q-values
            policy = "Highest"
        else:
            #print("Random action")
            next_state_list_without_x = [i for i in next_state_list if i != "x"] #Only consider these actions which different from "x"
            next_state = np.random.choice(next_state_list_without_x) #return possible next_state : float with random action (Except the "x" values)
            policy = "Random"
        return next_state, policy
    
        
    def fit(self, parameters, config):
        
        # ===== 0. LOAD MODEL =====
        self.set_parameters(parameters)

        # ===== Handle the dynamic dataset and multimodal collator for evaluation =====
        # Read dynamic data config
        dyn_cfg = self.config_sim.get("dynamic_data", {})
        enabled = dyn_cfg.get("enabled", False)
        mode = dyn_cfg.get("mode", "incremental")
        round_step = dyn_cfg.get("round_step", None)

        current_round = config.get("current_round", 0)

        # Decide whether to update dataset
        if enabled:
            # Always reload dataset to get round-specific indices and validation split
            # This ensures validation size changes when train size changes
            if self.data_scheduler is not None:
                self.reload_dataset(mode=mode, round_num=current_round)
            else:
                # Fallback to old approach
                if round_step is not None and (current_round % round_step == 0 or current_round == 1):
                    if mode == "reset":
                        self.reload_dataset(mode="replace", round_num=current_round)
                    elif mode == "incremental":
                        self.reload_dataset(mode="append", round_num=current_round)

        batch, epochs = (
            config["batch_size"],
            config["epochs"]
        )
        # Create a DataLoader for the training set
        # For multimodal: use collate_fn, enable num_workers for faster processing
        if self.clip_collator is not None:
            # Multimodal: use collator, enable workers
            trainloader = DataLoader(
                self.trainset,
                batch_size=batch,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if min(4, os.cpu_count() or 1) > 0 else False,
                prefetch_factor=2,
                collate_fn=self.clip_collator,
            )
        else:
            # Non-multimodal: check if local function (legacy)
            is_local_function = '<locals>' in self.apply_transforms.__qualname__ if self.apply_transforms else False
            num_workers = 0 if is_local_function else min(4, os.cpu_count() or 1)
            trainloader = DataLoader(
                self.trainset,
                batch_size=batch,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        #=======================================================================

        # ===== 1. TRAIN MODEL ON CURRENT RANK =====
        # Normal training process
        # Count the class distribution in the training set
        class_counts = self.count_class_distribution(trainloader)
  
        self.optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=self.optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )


        #To check how Q-Table changes after every round
        print(f"====> Q-TABLE {self.client_id}<====")
        print(f"  Action space: {self.action_space}")
        for i,j in zip(self.state_space,self.q_table):
            print(f"State: {i} - {j}")
        print("====> END OF Q-TABLE <====")

        # ===== 5. SELECT ACTION (EPSILON-GREEDY) =====
        current_round = config.get("current_round", 0)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * current_round) #epsilon value will be rounds

        current_rank_dict = config.get("rank_dict", None)
        current_state = current_rank_dict[self.client_id] #current rank
        current_state_index = list(self.env_space).index(current_state)
        next_state_list = self.env_space[current_state]

        next_state, policy = self.epsilon_greedy_policy(self.q_table, next_state_list, current_state_index, epsilon)
        action = next_state - current_state
        
        # print(f"Client {self.client_id} - Apply action: {action}, from State {current_state} to State {next_state}, Policy: {policy}, Epsilon: {epsilon:.4f}")
        params_to_send = self.get_parameters({})
        num_examples = len(trainloader.dataset)
        metrics = {"client_id": self.client_id, "class_distribution": class_counts, "rank": next_state, "kl_norm": self.kl_norm}

        return params_to_send, num_examples, metrics
    
    def evaluate(self, parameters, config):

        # ===== 0. LOAD MODEL WITH THE UPDATE RANKS=====
        self.set_parameters(parameters)

        # ===== Handle the dynamic dataset and multimodal collator for evaluation =====
        # Reload dataset to ensure validation size is updated for current round
        # This is necessary because evaluate() may be called after fit() in the same round
        # but with different dataset allocations
        dyn_cfg = self.config_sim.get("dynamic_data", {})
        enabled = dyn_cfg.get("enabled", False)
        if enabled and self.data_scheduler is not None:
            # Try to get current_round from config, fallback to "round" key or 0
            current_round = config.get("current_round", config.get("round", 0))
            self.reload_dataset(mode=dyn_cfg.get("mode", "incremental"), round_num=current_round)

        # Create validation DataLoader with collator for multimodal
        if self.clip_collator is not None:
            valloader = DataLoader(
                self.valset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if min(4, os.cpu_count() or 1) > 0 else False,
                prefetch_factor=2,
                collate_fn=self.clip_collator,
            )
        else:
            is_local_function = '<locals>' in self.apply_transforms.__qualname__ if self.apply_transforms else False
            num_workers = 0 if is_local_function else min(4, os.cpu_count() or 1)
            valloader = DataLoader(
                self.valset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

        # ===== EVALUATE THE MODEL =====
        class_counts = self.count_class_distribution(valloader)
        loss, accuracy, f1 = self.test(self.model, valloader, device=self.device)
        

        # ===== DEFINE THE REWARD SPACE =====
        ab_size = 0
        for name, param in self.model.state_dict().items():
            if name.endswith(".A") or name.endswith(".B"):
                ab_size += param.numel()
        ab_size /= 1e6  # MB
        reward = 1000*f1 - loss - ab_size

        # ==== UPDATE THE Q-TABLE ====
        
        current_rank_dict = config.get("current_rank_dict", None)
        current_state = current_rank_dict[self.client_id] #current rank
        current_state_index = list(self.env_space).index(current_state)

        next_rank_dict = config.get("next_rank_dict", None)
        next_state = next_rank_dict[self.client_id] #current rank
        next_state_index = list(self.env_space).index(next_state)
        next_state_list_without_x = [i for i in self.q_table[next_state_index] if i != "x"]

        action = next_state - current_state
        action_index = self.action_space.index(action)
        self.q_table[current_state_index][action_index] = round(self.q_table[current_state_index][action_index] + self.alpha * (reward + self.gamma * np.max(next_state_list_without_x) - self.q_table[current_state_index][action_index]), 3)
        # Save Q-table
        with open(self.q_table_file, "w") as f:
            json.dump(self.q_table, f)

        #To check how Q-Table changes after every round
        print(f"====> Q-TABLE {self.client_id}<====")
        print(f"  Action space: {self.action_space}")
        for i,j in zip(self.state_space,self.q_table):
            print(f"State: {i} - {j}")
        print("====> END OF Q-TABLE <====")
        


        # print(self.client_id, loss, f1, ab_size, reward, current_state, next_state)

        return float(loss), len(valloader.dataset), {"client_id": self.client_id, 
        "accuracy": float(accuracy), "f1_score": float(f1), "class_distribution": class_counts}



