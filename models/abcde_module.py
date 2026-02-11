"""
ABCDE module for extracting and scoring ABCDE criteria.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ABCDEFeatureExtractor(nn.Module):
    """
    Feature extractor for a specific ABCDE criterion.
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 256):
        """
        Initialize ABCDE feature extractor.
        
        Args:
            input_dim: Input feature dimension
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and attention map.
        
        Args:
            x: Input features (B, input_dim, H, W)
        
        Returns:
            Tuple of:
                - features: (B, feature_dim, H, W)
                - attention_map: (B, 1, H, W)
        """
        features = self.conv_layers(x)
        attention_map = self.attention(features)
        
        # Apply attention
        features = features * attention_map
        
        return features, attention_map


class ABCDEModule(nn.Module):
    """
    ABCDE module that extracts features for each criterion and outputs scores.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize ABCDE module.
        
        Args:
            input_dim: Input feature dimension
            feature_dim: Feature dimension for each criterion
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature extractors for each ABCDE criterion
        self.asymmetry_extractor = ABCDEFeatureExtractor(input_dim, feature_dim)
        self.border_extractor = ABCDEFeatureExtractor(input_dim, feature_dim)
        self.color_extractor = ABCDEFeatureExtractor(input_dim, feature_dim)
        self.diameter_extractor = ABCDEFeatureExtractor(input_dim, feature_dim)
        self.evolution_extractor = ABCDEFeatureExtractor(input_dim, feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Scoring heads
        self.asymmetry_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.border_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.diameter_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.evolution_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features (B, input_dim, H, W)
        
        Returns:
            Dictionary with:
                - abcde_scores: (B, 5) - Scores for A, B, C, D, E
                - attention_maps: Dict of attention maps for each criterion
                - features: Dict of extracted features for each criterion
        """
        # Extract features for each criterion
        asym_features, asym_att = self.asymmetry_extractor(features)
        border_features, border_att = self.border_extractor(features)
        color_features, color_att = self.color_extractor(features)
        diam_features, diam_att = self.diameter_extractor(features)
        evol_features, evol_att = self.evolution_extractor(features)
        
        # Global pooling
        asym_pooled = self.global_pool(asym_features).squeeze(-1).squeeze(-1)  # (B, feature_dim)
        border_pooled = self.global_pool(border_features).squeeze(-1).squeeze(-1)
        color_pooled = self.global_pool(color_features).squeeze(-1).squeeze(-1)
        diam_pooled = self.global_pool(diam_features).squeeze(-1).squeeze(-1)
        evol_pooled = self.global_pool(evol_features).squeeze(-1).squeeze(-1)
        
        # Score each criterion
        asym_score = self.asymmetry_head(asym_pooled).squeeze(-1)  # (B,)
        border_score = self.border_head(border_pooled).squeeze(-1)
        color_score = self.color_head(color_pooled).squeeze(-1)
        diam_score = self.diameter_head(diam_pooled).squeeze(-1)
        evol_score = self.evolution_head(evol_pooled).squeeze(-1)
        
        # Concatenate scores
        abcde_scores = torch.stack(
            [asym_score, border_score, color_score, diam_score, evol_score],
            dim=1,
        )  # (B, 5)
        
        return {
            'abcde_scores': abcde_scores,
            'attention_maps': {
                'asymmetry': asym_att,
                'border': border_att,
                'color': color_att,
                'diameter': diam_att,
                'evolution': evol_att,
            },
            'features': {
                'asymmetry': asym_features,
                'border': border_features,
                'color': color_features,
                'diameter': diam_features,
                'evolution': evol_features,
            },
        }

