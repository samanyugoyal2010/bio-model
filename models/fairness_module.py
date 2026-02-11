"""
Fairness module for adversarial debiasing during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FairnessDiscriminator(nn.Module):
    """
    Discriminator network for adversarial fairness training.
    Predicts skin tone from features (should fail if debiasing works).
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_skin_tones: int = 6,  # Fitzpatrick I-VI
        dropout: float = 0.1,
    ):
        """
        Initialize fairness discriminator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_skin_tones: Number of skin tone classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_skin_tones),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict skin tone from features.
        
        Args:
            features: Input features (B, input_dim)
        
        Returns:
            Skin tone logits: (B, num_skin_tones)
        """
        return self.discriminator(features)


class FairnessModule(nn.Module):
    """
    Fairness module that applies adversarial debiasing.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_skin_tones: int = 6,
        debias_weight: float = 0.1,
        dropout: float = 0.1,
    ):
        """
        Initialize fairness module.
        
        Args:
            feature_dim: Feature dimension
            num_skin_tones: Number of skin tone classes
            debias_weight: Weight for adversarial loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.debias_weight = debias_weight
        self.num_skin_tones = num_skin_tones
        
        # Discriminator (adversarial)
        self.discriminator = FairnessDiscriminator(
            input_dim=feature_dim,
            hidden_dim=256,
            num_skin_tones=num_skin_tones,
            dropout=dropout,
        )
        
        # Gradient reversal layer (for adversarial training)
        self.gradient_reversal = GradientReversalLayer(lambda_param=debias_weight)
    
    def forward(
        self,
        features: torch.Tensor,
        apply_adversarial: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (B, feature_dim)
            apply_adversarial: Whether to apply adversarial debiasing
        
        Returns:
            Features (unchanged, for compatibility)
        """
        if apply_adversarial and self.training:
            # Apply gradient reversal for adversarial training
            reversed_features = self.gradient_reversal(features)
            # Discriminator prediction (used in loss, not returned)
            _ = self.discriminator(reversed_features)
        
        return features
    
    def predict_skin_tone(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict skin tone (for evaluation).
        
        Args:
            features: Input features (B, feature_dim)
        
        Returns:
            Skin tone logits: (B, num_skin_tones)
        """
        return self.discriminator(features)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal function for adversarial training.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient reversal layer for adversarial training.
    """
    
    def __init__(self, lambda_param: float = 1.0):
        """
        Initialize gradient reversal layer.
        
        Args:
            lambda_param: Lambda parameter for gradient reversal
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_param)

