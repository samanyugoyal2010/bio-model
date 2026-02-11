"""
Multi-modal fusion module for combining image and clinical features.
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention_modules import CrossModalAttention


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module combining image and clinical features.
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        clinical_dim: int = 512,
        fusion_dim: int = 512,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-modal fusion module.
        
        Args:
            image_dim: Image feature dimension
            clinical_dim: Clinical feature dimension
            fusion_dim: Fusion dimension
            num_attention_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Projections
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.clinical_proj = nn.Linear(clinical_dim, fusion_dim)
        
        # Cross-attention layers
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(
                image_dim=fusion_dim,
                clinical_dim=fusion_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_attention_layers)
        ])
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image_features: Image features (B, image_dim) or (B, N, image_dim)
            clinical_features: Clinical features (B, clinical_dim) or None
        
        Returns:
            Fused features: (B, fusion_dim) or (B, N, fusion_dim)
        """
        # Project image features
        image_proj = self.image_proj(image_features)
        
        if clinical_features is not None:
            # Project clinical features
            clinical_proj = self.clinical_proj(clinical_features)
            
            # Apply cross-attention layers
            fused = image_proj
            for attention_layer in self.attention_layers:
                fused = attention_layer(fused, clinical_proj)
            
            # Concatenate and fuse
            if fused.dim() == 2:
                # Both are 2D
                concatenated = torch.cat([fused, clinical_proj], dim=-1)
            else:
                # Image is 3D, clinical is 2D
                clinical_expanded = clinical_proj.unsqueeze(1).expand_as(fused)
                concatenated = torch.cat([fused, clinical_expanded], dim=-1)
            
            output = self.fusion_layer(concatenated)
        else:
            # No clinical features, just use image features
            output = image_proj
        
        return output

