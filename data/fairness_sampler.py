"""
Fairness-aware sampling for balanced training across skin tones.
"""

import numpy as np
from torch.utils.data import Sampler, Dataset
from typing import List, Optional, Dict
from collections import defaultdict


class FairnessSampler(Sampler):
    """
    Sampler that ensures balanced sampling across skin tone groups.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        skin_tone_labels: Optional[List[int]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Initialize fairness sampler.
        
        Args:
            dataset: Dataset to sample from
            skin_tone_labels: List of skin tone labels (Fitzpatrick scale I-VI)
                If None, will try to extract from dataset
            num_samples: Number of samples per epoch
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        
        # Get skin tone labels
        if skin_tone_labels is None:
            self.skin_tone_labels = self._extract_skin_tones()
        else:
            self.skin_tone_labels = skin_tone_labels
        
        # Group indices by skin tone
        self.skin_tone_groups = defaultdict(list)
        for idx, skin_tone in enumerate(self.skin_tone_labels):
            self.skin_tone_groups[skin_tone].append(idx)
        
        # Calculate number of samples per group for balanced sampling
        self.num_groups = len(self.skin_tone_groups)
        if num_samples is None:
            num_samples = len(dataset)
        
        self.num_samples = num_samples
        self.samples_per_group = num_samples // self.num_groups
    
    def _extract_skin_tones(self) -> List[int]:
        """Extract skin tone labels from dataset if available."""
        skin_tones = []
        
        # Try to get skin tone from dataset
        if hasattr(self.dataset, 'skin_tones'):
            return self.dataset.skin_tones
        
        # If not available, assign random (for testing)
        # In practice, this should be provided or extracted from metadata
        return [np.random.randint(1, 7) for _ in range(len(self.dataset))]
    
    def __iter__(self):
        """Generate balanced sample indices."""
        indices = []
        
        # Sample from each group
        for skin_tone, group_indices in self.skin_tone_groups.items():
            if self.replacement:
                # Sample with replacement
                sampled = np.random.choice(
                    group_indices,
                    size=self.samples_per_group,
                    replace=True,
                )
            else:
                # Sample without replacement (may need to repeat if not enough)
                if len(group_indices) >= self.samples_per_group:
                    sampled = np.random.choice(
                        group_indices,
                        size=self.samples_per_group,
                        replace=False,
                    )
                else:
                    # Repeat indices if not enough samples
                    sampled = np.random.choice(
                        group_indices,
                        size=self.samples_per_group,
                        replace=True,
                    )
            
            indices.extend(sampled.tolist())
        
        # Shuffle to mix groups
        np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples


class StratifiedFairnessSampler(Sampler):
    """
    Stratified sampler that balances both skin tone and class labels.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        skin_tone_labels: Optional[List[int]] = None,
        class_labels: Optional[List[int]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Initialize stratified fairness sampler.
        
        Args:
            dataset: Dataset to sample from
            skin_tone_labels: Skin tone labels (Fitzpatrick I-VI)
            class_labels: Class labels (0=benign, 1=malignant)
            num_samples: Number of samples per epoch
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        
        # Get labels
        if skin_tone_labels is None:
            skin_tone_labels = self._extract_skin_tones()
        if class_labels is None:
            class_labels = self._extract_class_labels()
        
        self.skin_tone_labels = skin_tone_labels
        self.class_labels = class_labels
        
        # Group indices by (skin_tone, class) combination
        self.strata = defaultdict(list)
        for idx, (skin_tone, class_label) in enumerate(
            zip(skin_tone_labels, class_labels)
        ):
            self.strata[(skin_tone, class_label)].append(idx)
        
        # Calculate samples per stratum
        self.num_strata = len(self.strata)
        if num_samples is None:
            num_samples = len(dataset)
        
        self.num_samples = num_samples
        self.samples_per_stratum = num_samples // self.num_strata
    
    def _extract_skin_tones(self) -> List[int]:
        """Extract skin tone labels."""
        if hasattr(self.dataset, 'skin_tones'):
            return self.dataset.skin_tones
        return [np.random.randint(1, 7) for _ in range(len(self.dataset))]
    
    def _extract_class_labels(self) -> List[int]:
        """Extract class labels."""
        if hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        return [0] * len(self.dataset)  # Default to benign
    
    def __iter__(self):
        """Generate stratified balanced sample indices."""
        indices = []
        
        # Sample from each stratum
        for (skin_tone, class_label), stratum_indices in self.strata.items():
            if self.replacement:
                sampled = np.random.choice(
                    stratum_indices,
                    size=self.samples_per_stratum,
                    replace=True,
                )
            else:
                if len(stratum_indices) >= self.samples_per_stratum:
                    sampled = np.random.choice(
                        stratum_indices,
                        size=self.samples_per_stratum,
                        replace=False,
                    )
                else:
                    sampled = np.random.choice(
                        stratum_indices,
                        size=self.samples_per_stratum,
                        replace=True,
                    )
            
            indices.extend(sampled.tolist())
        
        # Shuffle
        np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples

