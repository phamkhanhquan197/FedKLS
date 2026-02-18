"""
Dynamic Data Sampling Schedule Utility

This module provides deterministic, disjoint, round-aware data allocation
for dynamic dataset updates in federated learning.

Key features:
- Incremental mode: Monotonic non-decreasing dataset size per client
- Reset mode: Full dataset repartitioning with different Dirichlet distributions
- Disjoint allocation: No duplicate samples between clients
- Deterministic: Reproducible based on seed
"""

import random
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from flwr_datasets import FederatedDataset
from datasets import Dataset
from mak.utils.helper import get_partitioner
from mak.utils.dataset_info import dataset_info


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
        config_sim: dict,  # NEW: Config dict to recreate partitioner
        val_ratio: float = 0.2,
        mode: str = "incremental",
        round_step: int = 10,
        start_fraction: float = 0.3,  # For incremental: initial data fraction
    ):
        """
        Initialize dynamic data scheduler.
        
        Args:
            federated_dataset: The federated dataset object
            num_clients: Number of clients
            total_rounds: Total number of training rounds
            seed: Random seed for reproducibility
            config_sim: Config dict containing dataset info and partitioner settings
            val_ratio: Validation split ratio (default 0.2)
            mode: "incremental" or "reset"
            round_step: Dataset update frequency (trigger every N rounds)
            start_fraction: For incremental mode, initial data fraction per client
        """
        self.federated_dataset = federated_dataset
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.seed = seed
        self.config_sim = config_sim  # NEW: Store config for repartitioning
        self.val_ratio = val_ratio
        self.mode = mode
        self.round_step = round_step
        self.start_fraction = start_fraction
        
        # Initialize random state
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Cache for client partition data
        self._client_partitions: Dict[int, Dataset] = {}
        self._client_total_indices: Dict[int, List[int]] = {}
        
        # Schedule cache: (client_id, round) -> train_indices
        self._schedule_cache: Dict[Tuple[int, int], List[int]] = {}
        
        # NEW: Cache for repartitioned datasets per round (for reset mode)
        self._repartitioned_datasets: Dict[int, FederatedDataset] = {}
        
        # Track allocated indices per client to ensure disjoint
        self._allocated_indices: Dict[int, set] = defaultdict(set)
        
        # Cache for validation indices: (client_id, round) -> val_indices
        # This ensures consistency with train_test_split results
        self._validation_cache: Dict[Tuple[int, int], List[int]] = {}
        
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
            # Use set for O(1) lookup instead of O(n) list lookup (critical for large datasets)
            current_indices = set()
            
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
                    current_indices.update(selected)
                
                # Determine end round for this milestone
                if i + 1 < len(milestone_sizes):
                    next_milestone = milestone_sizes[i + 1][0]
                    end_round = next_milestone
                else:
                    end_round = self.total_rounds + 1
                
                # Cache for all rounds in this milestone period
                # Convert set to sorted list for caching
                indices_copy = sorted(list(current_indices))
                for round_num in range(milestone, end_round):
                    self._schedule_cache[(cid, round_num)] = indices_copy
    
    def _repartition_dataset(self, round_num: int) -> FederatedDataset:
        """
        Repartition the full dataset with a new seed to create different distribution.
        
        Args:
            round_num: Round number (used to generate different seed)
            
        Returns:
            New FederatedDataset with repartitioned data
        """
        # Round 1 should use the original seed to match the initial partition
        # Round > 1 uses different seed to create different distribution
        if round_num == 1:
            repartition_seed = self.seed  # Use original seed for round 1
        else:
            # Large multiplier to ensure different seeds for subsequent rounds
            repartition_seed = self.seed + round_num * 10000
        
        config_copy = self.config_sim.copy()
        config_copy["common"] = self.config_sim["common"].copy()
        config_copy["common"]["seed"] = repartition_seed
        
        # Get partitioner with new seed
        partitioner_dict = get_partitioner(config_sim=config_copy)
        
        # Get dataset name
        dataset_name = self.config_sim["common"]["dataset"]
        if dataset_name not in dataset_info.keys():
            available = list(dataset_info.keys())
            raise Exception(f"Dataset name should be among: {available}")
        
        # Create new FederatedDataset with repartitioned data
        fds = FederatedDataset(dataset=dataset_name, partitioners=partitioner_dict)
        
        return fds
    
    def _create_reset_schedule(self):
        """
        Create reset schedule: full dataset repartitioning at each milestone.
        At each milestone, the entire dataset is repartitioned with a different
        seed, ensuring all samples are distributed among clients following
        Dirichlet distribution.
        """
        milestones = [r for r in range(1, self.total_rounds + 1) if r % self.round_step == 0]
        if not milestones:
            milestones = [self.total_rounds]
        
        # Ensure round 1 is included
        if 1 not in milestones:
            milestones = [1] + milestones
        
        # Remove duplicates and sort
        milestones = sorted(list(set(milestones)))
        
        # For each milestone, repartition the dataset
        for milestone in milestones:
            # Repartition dataset with new seed
            repartitioned_fds = self._repartition_dataset(milestone)
            self._repartitioned_datasets[milestone] = repartitioned_fds
            
            # Load full partitions for all clients (100% allocation)
            for cid in range(self.num_clients):
                partition = repartitioned_fds.load_partition(partition_id=cid)
                # Use all indices in the partition (100% allocation)
                all_indices = list(range(len(partition)))
                self._schedule_cache[(cid, milestone)] = all_indices
        
        # Fill in rounds between milestones with previous milestone's data
        for cid in range(self.num_clients):
            prev_milestone = None
            for round_num in range(1, self.total_rounds + 1):
                if round_num in milestones:
                    prev_milestone = round_num
                elif prev_milestone is not None:
                    # Use previous milestone's indices
                    self._schedule_cache[(cid, round_num)] = self._schedule_cache[(cid, prev_milestone)].copy()
                else:
                    # Should not happen if round 1 is in milestones
                    raise ValueError(f"No milestone found before round {round_num}")
    
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
    
    def _extract_validation_indices(
        self,
        partition: Dataset,
        original_indices: List[int],
        val_dataset: Dataset,
        split_seed: int
    ) -> List[int]:
        """
        Extract validation indices from train_test_split result.
              
        Args:
            partition: Original partition dataset
            original_indices: Original indices that were split
            val_dataset: Validation dataset from train_test_split
            split_seed: Seed used for the split (not used, kept for compatibility)
            
        Returns:
            List of validation indices from original_indices
        """
        # This ensures consistency with train_test_split() which may round differently
        
        # Get actual validation size from val_dataset
        actual_val_size = len(val_dataset)
        
        # Use the same seed to reconstruct the shuffle order
        # This matches train_test_split() behavior
        np_rng = np.random.RandomState(split_seed)
        shuffled_indices = original_indices.copy()
        np_rng.shuffle(shuffled_indices)
        
        # Get validation indices (first actual_val_size after shuffle)
        # This matches what train_test_split() actually created
        val_indices = sorted(shuffled_indices[:actual_val_size])
        
        return val_indices
    
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
        
        # For reset mode, find the milestone dataset for this round
        if self.mode == "reset":
            # Find the milestone for this round
            milestone = None
            for m in sorted(self._repartitioned_datasets.keys()):
                if m <= round_num:
                    milestone = m
                else:
                    break
            
            if milestone is None:
                raise ValueError(f"No repartitioned dataset found for round {round_num}")
            
            # Load partition from repartitioned dataset
            repartitioned_fds = self._repartitioned_datasets[milestone]
            partition = repartitioned_fds.load_partition(partition_id=client_id)
            
            # In reset mode, use standard train/val split
            trainset_full = partition
            split_seed = self.seed + client_id * 1000 + round_num
            splits = trainset_full.train_test_split(
                test_size=self.val_ratio,
                seed=split_seed
            )
            trainset = splits["train"]
            valset = splits["test"]
        else:
            # For incremental mode, use original partition
            partition = self._client_partitions[client_id]
            
            # Use train_test_split for all rounds (consistent approach)
            # But ensure data already trained in previous rounds is NOT in validation set (prevent data leakage)
            if round_num == 1:
                # Round 1: Standard train/val split using train_test_split
                trainset_full = partition.select(train_indices)
                split_seed = self.seed + client_id * 1000 + round_num
                splits = trainset_full.train_test_split(
                    test_size=self.val_ratio,
                    seed=split_seed
                )
                trainset = splits["train"]
                valset = splits["test"]
                
                # Cache validation indices for round 1
                # Extract validation indices by comparing original and split datasets
                val_indices_round1 = self._extract_validation_indices(
                    partition, train_indices, splits["test"], split_seed
                )
                self._validation_cache[(client_id, round_num)] = val_indices_round1
            else:
                # Round N > 1: Only use new data for validation to prevent data leakage
                prev_train_indices = self.get_client_round_indices(client_id, round_num - 1)
                
                # Calculate new indices (data not seen in previous rounds)
                train_indices_set = set(train_indices)
                prev_train_indices_set = set(prev_train_indices)
                new_indices = sorted(list(train_indices_set - prev_train_indices_set))
                
                if len(new_indices) > 0:
                    # Use train_test_split on new indices (consistent with round 1)
                    new_indices_dataset = partition.select(new_indices)
                    split_seed = self.seed + client_id * 1000 + round_num
                    splits_new = new_indices_dataset.train_test_split(
                        test_size=self.val_ratio,
                        seed=split_seed
                    )
                    
                    # Extract validation indices from new data split
                    val_indices_new = self._extract_validation_indices(
                        partition, new_indices, splits_new["test"], split_seed
                    )
                    
                    # Get validation indices from all previous rounds (from cache or compute)
                    all_val_indices = set()
                    
                    # Round 1 validation (from cache or compute)
                    if (client_id, 1) in self._validation_cache:
                        round1_val_indices = self._validation_cache[(client_id, 1)]
                    else:
                        round1_train_indices = self.get_client_round_indices(client_id, 1)
                        round1_dataset = partition.select(round1_train_indices)
                        round1_split_seed = self.seed + client_id * 1000 + 1
                        round1_splits = round1_dataset.train_test_split(
                            test_size=self.val_ratio,
                            seed=round1_split_seed
                        )
                        round1_val_indices = self._extract_validation_indices(
                            partition, round1_train_indices, round1_splits["test"], round1_split_seed
                        )
                        self._validation_cache[(client_id, 1)] = round1_val_indices
                    all_val_indices.update(round1_val_indices)
                    
                    # Accumulate validation from all previous rounds (round 2 to round_num-1)
                    for prev_round in range(2, round_num):
                        prev_prev_train = self.get_client_round_indices(client_id, prev_round - 1)
                        prev_curr_train = self.get_client_round_indices(client_id, prev_round)
                        prev_prev_set = set(prev_prev_train)
                        prev_curr_set = set(prev_curr_train)
                        prev_new = sorted(list(prev_curr_set - prev_prev_set))
                        
                        if len(prev_new) > 0:
                            # Get from cache or compute
                            if (client_id, prev_round) in self._validation_cache:
                                prev_val_indices = self._validation_cache[(client_id, prev_round)]
                            else:
                                prev_new_dataset = partition.select(prev_new)
                                prev_split_seed = self.seed + client_id * 1000 + prev_round
                                prev_splits = prev_new_dataset.train_test_split(
                                    test_size=self.val_ratio,
                                    seed=prev_split_seed
                                )
                                prev_val_indices = self._extract_validation_indices(
                                    partition, prev_new, prev_splits["test"], prev_split_seed
                                )
                                self._validation_cache[(client_id, prev_round)] = prev_val_indices
                            all_val_indices.update(prev_val_indices)
                    
                    # Add new validation indices for current round
                    all_val_indices.update(val_indices_new)
                    self._validation_cache[(client_id, round_num)] = val_indices_new
                    
                    # Training set: all train_indices EXCEPT validation indices
                    train_indices_final = sorted(list(train_indices_set - all_val_indices))
                    
                    # Validation set: accumulated from all rounds (round 1 to round_num)
                    val_indices_final = sorted(list(all_val_indices))
                    
                    trainset = partition.select(train_indices_final)
                    valset = partition.select(val_indices_final)
                else:
                    # No new data: keep validation set from previous rounds
                    # Get accumulated validation indices from all previous rounds (round 1 to round_num-1)
                    all_val_indices = set()
                    
                    # Round 1 validation (from cache or compute)
                    if (client_id, 1) in self._validation_cache:
                        round1_val_indices = self._validation_cache[(client_id, 1)]
                    else:
                        round1_train_indices = self.get_client_round_indices(client_id, 1)
                        round1_dataset = partition.select(round1_train_indices)
                        round1_split_seed = self.seed + client_id * 1000 + 1
                        round1_splits = round1_dataset.train_test_split(
                            test_size=self.val_ratio,
                            seed=round1_split_seed
                        )
                        round1_val_indices = self._extract_validation_indices(
                            partition, round1_train_indices, round1_splits["test"], round1_split_seed
                        )
                        self._validation_cache[(client_id, 1)] = round1_val_indices
                    all_val_indices.update(round1_val_indices)
                    
                    # Accumulate validation from all previous rounds
                    for prev_round in range(2, round_num):
                        prev_prev_train = self.get_client_round_indices(client_id, prev_round - 1)
                        prev_curr_train = self.get_client_round_indices(client_id, prev_round)
                        prev_prev_set = set(prev_prev_train)
                        prev_curr_set = set(prev_curr_train)
                        prev_new = sorted(list(prev_curr_set - prev_prev_set))
                        
                        if len(prev_new) > 0:
                            # Get from cache or compute
                            if (client_id, prev_round) in self._validation_cache:
                                prev_val_indices = self._validation_cache[(client_id, prev_round)]
                            else:
                                prev_new_dataset = partition.select(prev_new)
                                prev_split_seed = self.seed + client_id * 1000 + prev_round
                                prev_splits = prev_new_dataset.train_test_split(
                                    test_size=self.val_ratio,
                                    seed=prev_split_seed
                                )
                                prev_val_indices = self._extract_validation_indices(
                                    partition, prev_new, prev_splits["test"], prev_split_seed
                                )
                                self._validation_cache[(client_id, prev_round)] = prev_val_indices
                            all_val_indices.update(prev_val_indices)
                    
                    # Training set: all train_indices EXCEPT validation indices
                    train_indices_final = sorted(list(train_indices_set - all_val_indices))
                    
                    # Validation set: accumulated from all previous rounds (round 1 to round_num-1, no new data this round)
                    val_indices_final = sorted(list(all_val_indices))
                    
                    trainset = partition.select(train_indices_final)
                    valset = partition.select(val_indices_final)
        
        if apply_transforms:
            trainset = trainset.with_transform(apply_transforms)
            valset = valset.with_transform(apply_transforms)
        
        return trainset, valset
