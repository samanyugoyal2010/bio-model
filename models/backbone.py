"""
Hybrid CNN-Transformer backbone for skin cancer detection.
Combines EfficientNet-V2 and Vision Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import timm
from transformers import ViTModel, ViTConfig


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNet-V2 backbone for feature extraction.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        num_features: int = 1280,
    ):
        """
        Initialize EfficientNet-V2 backbone.
        
        Args:
            model_name: EfficientNet-V2 variant name
            pretrained: Whether to use pretrained weights
            num_features: Number of output features
        """
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='',  # Keep spatial dimensions
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model.forward_features(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Projection layer
        self.projection = nn.Conv2d(
            self.feature_dim,
            num_features,
            kernel_size=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Features: (B, num_features, H', W')
        """
        features = self.model.forward_features(x)
        features = self.projection(features)
        return features


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for feature extraction.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        pretrained: bool = True,
    ):
        """
        Initialize ViT backbone.
        
        Args:
            image_size: Input image size
            patch_size: Patch size
            dim: Embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
            emb_dropout: Embedding dropout rate
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        
        # Create ViT config
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=mlp_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        
        # Load pretrained model or create new
        if pretrained and image_size == 224:  # Standard pretrained size
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            # Adjust if dimensions don't match
            if self.model.config.hidden_size != dim:
                self.model = ViTModel(config)
        else:
            self.model = ViTModel(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Tuple of:
                - cls_token: (B, dim) - CLS token embedding
                - patch_tokens: (B, num_patches, dim) - Patch embeddings
        """
        outputs = self.model(pixel_values=x)
        
        # Extract CLS token and patch tokens
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        patch_tokens = outputs.last_hidden_state[:, 1:]  # Patch tokens
        
        return cls_token, patch_tokens


class HybridBackbone(nn.Module):
    """
    Hybrid backbone combining EfficientNet-V2 and Vision Transformer.
    """
    
    def __init__(
        self,
        efficientnet_variant: str = "efficientnet_v2_s",
        image_size: int = 512,
        patch_size: int = 16,
        vit_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        fusion_dim: int = 512,
        pretrained: bool = True,
    ):
        """
        Initialize hybrid backbone.
        
        Args:
            efficientnet_variant: EfficientNet-V2 variant name
            image_size: Input image size
            patch_size: ViT patch size
            vit_dim: ViT embedding dimension
            vit_depth: ViT depth
            vit_heads: ViT attention heads
            fusion_dim: Fusion dimension
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # CNN backbone (EfficientNet-V2)
        self.cnn_backbone = EfficientNetV2Backbone(
            model_name=efficientnet_variant,
            pretrained=pretrained,
            num_features=fusion_dim,
        )
        
        # ViT backbone
        self.vit_backbone = ViTBackbone(
            image_size=image_size,
            patch_size=patch_size,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            pretrained=pretrained,
        )
        
        # Fusion layers
        self.cnn_projection = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1)
        
        # Project ViT features to fusion dimension
        self.vit_projection = nn.Linear(vit_dim, fusion_dim)
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True,
        )
        
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Tuple of:
                - cnn_features: (B, fusion_dim, H', W') - CNN features
                - vit_cls: (B, fusion_dim) - ViT CLS token
                - vit_patches: (B, num_patches, fusion_dim) - ViT patch tokens
        """
        # CNN features
        cnn_features = self.cnn_backbone(x)  # (B, fusion_dim, H', W')
        cnn_features = self.cnn_projection(cnn_features)
        
        # ViT features
        vit_cls, vit_patches = self.vit_backbone(x)  # (B, vit_dim), (B, num_patches, vit_dim)
        vit_cls = self.vit_projection(vit_cls)  # (B, fusion_dim)
        vit_patches = self.vit_projection(vit_patches)  # (B, num_patches, fusion_dim)
        
        # Cross-attention fusion
        # Reshape CNN features to sequence
        B, C, H, W = cnn_features.shape
        cnn_sequence = cnn_features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, fusion_dim)
        
        # Apply cross-attention: CNN queries, ViT patches as key/value
        fused_features, _ = self.cross_attention(
            query=cnn_sequence,
            key=vit_patches,
            value=vit_patches,
        )
        
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_dropout(fused_features)
        
        # Reshape back to spatial
        cnn_features = fused_features.permute(0, 2, 1).view(B, C, H, W)
        
        return cnn_features, vit_cls, vit_patches

