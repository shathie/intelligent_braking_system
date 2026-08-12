"""
Stratified batch sampling for balanced class representation in each batch.

When class distribution is imbalanced, standard random sampling can result in batches
where rare classes are severely underrepresented. This sampler ensures every batch
contains a balanced mix of classes, improving training stability and convergence.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Dict, Optional, Callable


class StratifiedBatchSampler(Sampler):
    """Create batches with balanced class representation.
    
    This sampler ensures that each batch contains approximately equal representation
    from all classes, preventing batches from being dominated by majority classes.
    
    Example:
        >>> dataset = SomeDataset()  # 1000 samples, 10 classes with severe imbalance
        >>> class_ids = [dataset[i]['class_id'] for i in range(len(dataset))]
        >>> sampler = StratifiedBatchSampler(
        ...     class_ids=class_ids,
        ...     batch_size=32,
        ...     num_batches=50,
        ...     shuffle=True
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        class_ids: np.ndarray,
        batch_size: int,
        num_batches: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """Initialize stratified batch sampler.
        
        Args:
            class_ids: array of class labels for all samples, shape (num_samples,)
            batch_size: number of samples per batch
            num_batches: number of batches to generate. If None, uses num_samples / batch_size
            drop_last: whether to drop incomplete batches
            shuffle: whether to shuffle samples within each class and batch order
            seed: random seed for reproducibility
        """
        self.class_ids = np.array(class_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Group indices by class
        self.class_indices: Dict[int, List[int]] = {}
        for class_id in np.unique(self.class_ids):
            self.class_indices[class_id] = np.where(self.class_ids == class_id)[0].tolist()
        
        self.num_classes = len(self.class_indices)
        
        # Calculate batch composition
        self.samples_per_class_per_batch = max(1, self.batch_size // self.num_classes)
        
        # Determine number of batches
        if num_batches is None:
            self.num_batches = len(self.class_ids) // self.batch_size
            if not self.drop_last and len(self.class_ids) % self.batch_size != 0:
                self.num_batches += 1
        else:
            self.num_batches = num_batches
        
        # Statistics
        self.class_counts = {
            class_id: len(indices)
            for class_id, indices in self.class_indices.items()
        }
    
    def __iter__(self):
        """Generate batches with balanced class distribution."""
        batches = []
        
        # Create pointers for cycling through each class's indices
        class_pointers = {class_id: 0 for class_id in self.class_indices.keys()}
        
        for _ in range(self.num_batches):
            batch = []
            
            # Iterate through all classes
            for class_id in sorted(self.class_indices.keys()):
                indices = self.class_indices[class_id]
                
                if self.shuffle:
                    indices = np.random.permutation(indices).tolist()
                
                # Cycle pointer if we've exhausted this class
                if class_pointers[class_id] + self.samples_per_class_per_batch > len(indices):
                    class_pointers[class_id] = 0
                    if self.shuffle:
                        indices = np.random.permutation(indices).tolist()
                
                # Take samples from this class
                start = class_pointers[class_id]
                end = min(start + self.samples_per_class_per_batch, len(indices))
                batch.extend(indices[start:end])
                
                # Update pointer
                class_pointers[class_id] = end
            
            # Fill remaining slots with random samples to reach batch_size
            if len(batch) < self.batch_size:
                all_indices = np.concatenate([np.array(self.class_indices[c]) for c in sorted(self.class_indices.keys())])
                remaining = self.batch_size - len(batch)
                extras = np.random.choice(all_indices, size=remaining, replace=True)
                batch.extend(extras)
            
            # Truncate to exact batch size
            batch = batch[:self.batch_size]
            
            # Optionally shuffle within batch
            if self.shuffle:
                np.random.shuffle(batch)
            
            batches.append(batch)
        
        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Yield individual indices
        for batch in batches:
            for idx in batch:
                yield int(idx)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.num_batches * self.batch_size
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return self.batch_size
    
    def get_class_distribution_in_batch(self) -> Dict[int, int]:
        """Return expected number of samples per class in each batch."""
        return {
            class_id: self.samples_per_class_per_batch
            for class_id in self.class_indices.keys()
        }
    
    def print_stats(self):
        """Print statistics about class distribution."""
        print(f"StratifiedBatchSampler Statistics:")
        print(f"  Total samples: {len(self.class_ids)}")
        print(f"  Total classes: {self.num_classes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of batches: {self.num_batches}")
        print(f"  Samples per class per batch: {self.samples_per_class_per_batch}")
        print(f"  Total iterations: {len(self)}")
        print(f"\n  Class counts:")
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            pct = 100.0 * count / len(self.class_ids)
            print(f"    Class {class_id:2d}: {count:6d} samples ({pct:5.2f}%)")


class AdaptiveStratifiedSampler(StratifiedBatchSampler):
    """Enhanced stratified sampler that adapts composition based on training progress.
    
    This sampler starts with balanced batches but gradually increases minority class
    representation as training progresses (curriculum learning approach).
    """
    
    def __init__(
        self,
        class_ids: np.ndarray,
        batch_size: int,
        num_batches: Optional[int] = None,
        minority_threshold: float = 0.05,
        max_minority_multiplier: float = 3.0,
        schedule: str = 'linear',
        **kwargs
    ):
        """Initialize adaptive stratified sampler.
        
        Args:
            class_ids: array of class labels
            batch_size: samples per batch
            num_batches: number of batches
            minority_threshold: classes with <5% samples are considered minority
            max_minority_multiplier: max how much to boost minority classes (3x = 3× more samples)
            schedule: how to scale multiplier over training ('linear', 'cosine', 'step')
        """
        super().__init__(class_ids, batch_size, num_batches, **kwargs)
        
        self.minority_threshold = minority_threshold
        self.max_minority_multiplier = max_minority_multiplier
        self.schedule = schedule
        
        # Identify minority vs majority classes
        self.minority_classes = set()
        self.majority_classes = set()
        
        for class_id, count in self.class_counts.items():
            pct = count / len(self.class_ids)
            if pct < minority_threshold:
                self.minority_classes.add(class_id)
            else:
                self.majority_classes.add(class_id)
        
        print(f"Minority classes (< {100*minority_threshold}%): {sorted(self.minority_classes)}")
        print(f"Majority classes: {sorted(self.majority_classes)}")
    
    def get_multiplier_at_step(self, current_step: int, total_steps: int) -> Dict[int, float]:
        """Get per-class multiplier at current training step.
        
        Args:
            current_step: current training step (0-indexed)
            total_steps: total number of training steps
        
        Returns:
            dict mapping class_id to sampling multiplier
        """
        # Compute progress (0.0 to 1.0)
        progress = current_step / max(1, total_steps - 1)
        
        # Compute minority class multiplier based on schedule
        if self.schedule == 'linear':
            minority_mult = 1.0 + (self.max_minority_multiplier - 1.0) * progress
        elif self.schedule == 'cosine':
            # Slow start, fast ramp-up in middle, plateau at end
            from math import cos, pi
            minority_mult = 1.0 + (self.max_minority_multiplier - 1.0) * (1.0 - cos(progress * pi)) / 2.0
        elif self.schedule == 'step':
            # Step-wise increase
            if progress < 0.5:
                minority_mult = 1.5
            elif progress < 0.8:
                minority_mult = 2.25
            else:
                minority_mult = self.max_minority_multiplier
        else:
            minority_mult = self.max_minority_multiplier
        
        # Build multiplier dict
        multipliers = {}
        for class_id in self.class_indices.keys():
            if class_id in self.minority_classes:
                multipliers[class_id] = minority_mult
            else:
                # Slightly reduce majority class to maintain batch size
                n_minority = len(self.minority_classes)
                n_majority = len(self.majority_classes)
                boost = (minority_mult - 1.0) * n_minority / max(1, n_majority)
                multipliers[class_id] = max(1.0, 1.0 - boost * 0.5)
        
        return multipliers


class ImbalancedDatasetSampler(Sampler):
    """Simple weighted sampler based on inverse class frequency.
    
    This is simpler than StratifiedBatchSampler but less effective at ensuring
    balanced batches.
    """
    
    def __init__(self, class_ids: np.ndarray, num_samples: Optional[int] = None, seed: Optional[int] = None):
        """Initialize imbalanced dataset sampler.
        
        Args:
            class_ids: array of class labels
            num_samples: number of samples to draw (default: len(class_ids))
            seed: random seed
        """
        self.class_ids = np.array(class_ids)
        self.num_samples = num_samples or len(self.class_ids)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Compute per-class weights (inverse frequency)
        unique_classes, class_counts = np.unique(self.class_ids, return_counts=True)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        
        # Assign weight to each sample
        self.weights = np.zeros(len(self.class_ids))
        for i, class_id in enumerate(self.class_ids):
            class_idx = np.where(unique_classes == class_id)[0][0]
            self.weights[i] = class_weights[class_idx]
        
        self.weights = torch.as_tensor(self.weights, dtype=torch.double)
    
    def __iter__(self):
        """Generate weighted random samples."""
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples


# Example usage
if __name__ == "__main__":
    print("Stratified Batch Sampler Example")
    print("="*50)
    
    # Simulate imbalanced dataset with 10 classes
    np.random.seed(42)
    num_samples = 1000
    num_classes = 10
    
    # Create imbalanced class distribution
    class_ids = np.concatenate([
        np.full(500, 0),   # Majority class
        np.full(200, 1),
        np.full(100, 2),
        np.full(50, 3),
        np.full(30, 4),
        np.full(20, 5),
        np.full(20, 6),
        np.full(20, 7),
        np.full(20, 8),
        np.full(40, 9),    # Minority class
    ])
    
    assert len(class_ids) == num_samples
    
    # Create sampler
    sampler = StratifiedBatchSampler(
        class_ids=class_ids,
        batch_size=32,
        num_batches=50,
        shuffle=True,
        seed=42
    )
    
    # Print statistics
    sampler.print_stats()
    print()
    
    # Check class distribution in first few batches
    print("Class distribution in first 3 batches:")
    print("-"*50)
    
    batch_size = 32
    batch_num = 0
    current_batch = []
    batch_classes = []
    
    for idx in sampler:
        current_batch.append(idx)
        batch_classes.append(class_ids[idx])
        
        if len(current_batch) == batch_size:
            unique, counts = np.unique(batch_classes, return_counts=True)
            print(f"Batch {batch_num}:")
            for c, cnt in zip(unique, counts):
                print(f"  Class {c}: {cnt:2d} samples")
            
            batch_num += 1
            current_batch = []
            batch_classes = []
            
            if batch_num >= 3:
                break
    
    print("\n✓ Example complete")
