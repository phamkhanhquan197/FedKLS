"""
Dynamic Data Sampling Schedule Utility

This module provides deterministic, disjoint, round-aware data allocation
for dynamic dataset updates in federated learning.

Key features:
- Incremental mode: Monotonic non-decreasing dataset size per client
- Reset mode: Dataset replacement with configurable size changes
- Disjoint allocation: No duplicate samples between clients
- Deterministic: Reproducible based on seed
"""

import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from flwr_datasets import FederatedDataset
from datasets import Dataset


class DynamicDataScheduler:
    """
    Manages dynamic data allocation schedule for federated learning clients.
    
    Ensures:
    - Disjoint sample allocation between clients
    - Deterministic behavior based on seed
    - Round-aware allocation (incremental or reset)
    """
    
    def __init__(
        self,
        federated_dataset: FederatedDataset,
        num_clients: int,
        total_rounds: int,
        seed: int,
        val_ratio: float = 0.2,
        mode: str = "incremental",
        round_step: int = 10,
        start_fraction: float = 0.3,  # For incremental: initial data fraction
        reset_size_range: Optional[Tuple[float, float]] = None,  # For reset: (min, max) fraction
    ):
        """
        Initialize dynamic data scheduler.
        
        Args:
            federated_dataset: The federated dataset object
            num_clients: Number of clients
            total_rounds: Total number of training rounds
            seed: Random seed for reproducibility
            val_ratio: Validation split ratio (default 0.2)
            mode: "incremental" or "reset"
            round_step: Dataset update frequency (trigger every N rounds)
            start_fraction: For incremental mode, initial data fraction per client
            reset_size_range: For reset mode, (min_fraction, max_fraction) for size variation
        """
        self.federated_dataset = federated_dataset
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.seed = seed
        self.val_ratio = val_ratio
        self.mode = mode
        self.round_step = round_step
        self.start_fraction = start_fraction
        self.reset_size_range = reset_size_range or (0.5, 1.0)
        
        # Initialize random state
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Cache for client partition data
        self._client_partitions: Dict[int, Dataset] = {}
        self._client_total_indices: Dict[int, List[int]] = {}
        
        # Schedule cache: (client_id, round) -> train_indices
        self._schedule_cache: Dict[Tuple[int, int], List[int]] = {}
        
        # Track allocated indices per client to ensure disjoint
        self._allocated_indices: Dict[int, set] = defaultdict(set)
        
        # Initialize partitions and create schedule
        self._initialize_partitions()
        self._create_schedule()
    
    def _initialize_partitions(self):
        """Load and cache client partitions."""
        for cid in range(self.num_clients):
            partition = self.federated_dataset.load_partition(partition_id=cid)
            self._client_partitions[cid] = partition
            # Get all indices for this partition
            total_size = len(partition)
            self._client_total_indices[cid] = list(range(total_size))
    
    def _create_schedule(self):
        """Create allocation schedule for all clients and rounds."""
        if self.mode == "incremental":
            self._create_incremental_schedule()
        elif self.mode == "reset":
            self._create_reset_schedule()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _create_incremental_schedule(self):
        """Create incremental schedule: monotonic non-decreasing dataset size."""
        milestones = [r for r in range(1, self.total_rounds + 1) if r % self.round_step == 0]
        if not milestones:
            milestones = [self.total_rounds]  # At least one milestone at the end
        
        # Ensure round 1 is included as initial milestone
        if 1 not in milestones:
            milestones = [1] + milestones
        
        # Ensure total_rounds is included as final milestone to reach 100% allocation
        if self.total_rounds not in milestones:
            milestones.append(self.total_rounds)
        
        # Remove duplicates and sort
        milestones = sorted(list(set(milestones)))
        
        for cid in range(self.num_clients):
            partition_size = len(self._client_partitions[cid])
            available_indices = sorted(self._client_total_indices[cid])
            
            # Calculate target sizes at each milestone
            # Linear growth from start_fraction to 1.0 based on rounds elapsed
            # This ensures: after total_rounds, all data is allocated (target_fraction = 1.0)
            milestone_sizes = []
            for i, milestone in enumerate(milestones):
                if milestone == 1:
                    # Round 1 uses start_fraction
                    target_fraction = self.start_fraction
                else:
                    # Calculate progress based on rounds elapsed, not milestone index
                    # This ensures even growth: (milestone - 1) / (total_rounds - 1)
                    if self.total_rounds > 1:
                        progress = (milestone - 1) / (self.total_rounds - 1)
                    else:
                        progress = 0
                    target_fraction = self.start_fraction + (1.0 - self.start_fraction) * progress
                    # Ensure target_fraction doesn't exceed 1.0
                    target_fraction = min(target_fraction, 1.0)
                target_size = int(partition_size * target_fraction)
                milestone_sizes.append((milestone, target_size))
            
            # Allocate indices incrementally
            current_indices = []
            
            for i, (milestone, target_size) in enumerate(milestone_sizes):
                # Ensure we have at least target_size indices
                # In incremental mode, we ADD to existing, never remove
                while len(current_indices) < target_size:
                    # Select new indices deterministically from available pool
                    remaining_indices = [idx for idx in available_indices if idx not in current_indices]
                    if not remaining_indices:
                        break  # No more indices available
                    
                    needed = target_size - len(current_indices)
                    # Use deterministic selection based on seed + cid + milestone
                    self.np_rng.seed(self.seed + cid * 1000 + milestone + len(current_indices))
                    selected = self.np_rng.choice(
                        remaining_indices, 
                        size=min(needed, len(remaining_indices)), 
                        replace=False
                    ).tolist()
                    current_indices.extend(sorted(selected))
                
                # Determine end round for this milestone
                if i + 1 < len(milestone_sizes):
                    next_milestone = milestone_sizes[i + 1][0]
                    end_round = next_milestone
                else:
                    end_round = self.total_rounds + 1
                
                # Cache for all rounds in this milestone period
                # IMPORTANT: Use a copy to avoid reference issues
                indices_copy = sorted(current_indices.copy())
                for round_num in range(milestone, end_round):
                    self._schedule_cache[(cid, round_num)] = indices_copy
    
    def _create_reset_schedule(self):
        """Create reset schedule: dataset replacement with size changes."""
        milestones = [r for r in range(1, self.total_rounds + 1) if r % self.round_step == 0]
        if not milestones:
            milestones = [self.total_rounds]
        
        for cid in range(self.num_clients):
            partition_size = len(self._client_partitions[cid])
            available_indices = sorted(self._client_total_indices[cid])
            
            # Initial round: use start_fraction
            initial_size = int(partition_size * self.start_fraction)
            self.np_rng.seed(self.seed + cid * 1000 + 1)
            initial_selected = sorted(self.np_rng.choice(
                available_indices,
                size=min(initial_size, len(available_indices)),
                replace=False
            ).tolist())
            
            prev_indices = initial_selected
            
            for round_num in range(1, self.total_rounds + 1):
                if round_num in milestones and round_num > 1:
                    # Reset: select new indices with different size
                    min_size = int(partition_size * self.reset_size_range[0])
                    max_size = int(partition_size * self.reset_size_range[1])
                    
                    # Deterministic size selection
                    self.np_rng.seed(self.seed + cid * 1000 + round_num)
                    target_size = self.np_rng.randint(min_size, max_size + 1)
                    
                    # Select indices deterministically (different from previous)
                    self.np_rng.seed(self.seed + cid * 2000 + round_num)
                    selected = sorted(self.np_rng.choice(
                        available_indices,
                        size=min(target_size, len(available_indices)),
                        replace=False
                    ).tolist())
                    
                    self._schedule_cache[(cid, round_num)] = selected
                    prev_indices = selected
                else:
                    # Use previous round's indices if not a milestone
                    self._schedule_cache[(cid, round_num)] = prev_indices.copy()
    
    def get_client_round_indices(self, client_id: int, round_num: int) -> List[int]:
        """
        Get train indices for a specific client at a specific round.
        
        Args:
            client_id: Client ID
            round_num: Current round number
            
        Returns:
            List of indices for training set
        """
        if (client_id, round_num) in self._schedule_cache:
            return self._schedule_cache[(client_id, round_num)].copy()
        
        # Fallback: use initial indices
        if client_id in self._client_total_indices:
            partition_size = len(self._client_partitions[client_id])
            initial_size = int(partition_size * self.start_fraction)
            return sorted(self._client_total_indices[client_id][:initial_size])
        
        return []
    
    def get_client_round_datasets(
        self, 
        client_id: int, 
        round_num: int,
        apply_transforms=None
    ) -> Tuple[Dataset, Dataset]:
        """
        Get train and validation datasets for a client at a specific round.
        
        Args:
            client_id: Client ID
            round_num: Current round number
            apply_transforms: Optional transform function
            
        Returns:
            Tuple of (trainset, valset) with proper validation split
        """
        train_indices = self.get_client_round_indices(client_id, round_num)
        
        if not train_indices:
            raise ValueError(f"No indices found for client {client_id} at round {round_num}")
        
        # Get partition
        partition = self._client_partitions[client_id]
        
        # Select subset based on indices
        trainset_full = partition.select(train_indices)
        
        # Split into train and validation
        # Use deterministic split based on seed + client_id + round
        split_seed = self.seed + client_id * 1000 + round_num
        splits = trainset_full.train_test_split(
            test_size=self.val_ratio,
            seed=split_seed
        )
        
        trainset = splits["train"]
        valset = splits["test"]
        
        if apply_transforms:
            trainset = trainset.with_transform(apply_transforms)
            valset = valset.with_transform(apply_transforms)
        
        return trainset, valset
    
    def get_client_dataset_size(self, client_id: int, round_num: int) -> Tuple[int, int]:
        """
        Get train and validation sizes for a client at a specific round.
        
        Args:
            client_id: Client ID
            round_num: Current round number
            
        Returns:
            Tuple of (train_size, val_size)
        """
        train_indices = self.get_client_round_indices(client_id, round_num)
        total_size = len(train_indices)
        train_size = int(total_size * (1 - self.val_ratio))
        val_size = total_size - train_size
        return train_size, val_size
    
    def verify_disjoint(self, round_num: int) -> bool:
        """
        Verify that clients have disjoint indices at a specific round.
        
        Args:
            round_num: Round number to check
            
        Returns:
            True if all clients have disjoint indices
        """
        all_indices = []
        for cid in range(self.num_clients):
            indices = set(self.get_client_round_indices(cid, round_num))
            if indices & set(all_indices):
                return False
            all_indices.extend(indices)
        return True

