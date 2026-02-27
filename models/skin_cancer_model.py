"""
Main skin cancer detection model combining all components.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import HybridBackbone
from .abcde_module import ABCDEModule
from .attention_modules import MultiHeadABCDEAttention
from .fusion_module import MultiModalFusion
from .fairness_module import FairnessModule
from .uncertainty_module import UncertaintyModule

from core import ClinicalFeatureEncoder


class SkinCancerModel(nn.Module):
    """
    Main model for facial skin cancer detection with ABCDE scoring.
    """
    
    def __init__(
        self,
        # Backbone config
        efficientnet_variant: str = "efficientnet_v2_s",
        image_size: int = 512,
        patch_size: int = 16,
        vit_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        # ABCDE config
        abcde_enabled: bool = True,
        abcde_feature_dim: int = 256,
        # Fusion config
        fusion_enabled: bool = True,
        clinical_features_dim: int = 10,
        fusion_dim: int = 512,
        # Fairness config
        fairness_enabled: bool = True,
        num_skin_tones: int = 6,
        debias_weight: float = 0.1,
        # Uncertainty config
        uncertainty_enabled: bool = True,
        num_samples: int = 10,
        # Output config
        num_classes: int = 2,
        lesion_types: int = 7,
        pretrained: bool = True,
    ):
        """
        Initialize main model.
        """
        super().__init__()
        
        self.abcde_enabled = abcde_enabled
        self.fusion_enabled = fusion_enabled
        self.fairness_enabled = fairness_enabled
        self.uncertainty_enabled = uncertainty_enabled
        
        # Backbone
        self.backbone = HybridBackbone(
            efficientnet_variant=efficientnet_variant,
            image_size=image_size,
            patch_size=patch_size,
            vit_dim=vit_dim,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            fusion_dim=fusion_dim,
            pretrained=pretrained,
        )
        
        # ABCDE Module
        if abcde_enabled:
            self.abcde_module = ABCDEModule(
                input_dim=fusion_dim,
                feature_dim=abcde_feature_dim,
                num_heads=8,
            )
        
        # Multi-head ABCDE attention
        if abcde_enabled:
            self.abcde_attention = MultiHeadABCDEAttention(
                embed_dim=fusion_dim,
                num_heads=5,  # One for each ABCDE criterion
                head_dim=64,
            )
        
        # Clinical feature encoder
        if fusion_enabled:
            self.clinical_encoder = ClinicalFeatureEncoder(
                feature_dims={
                    'age': 1,
                    'gender': 2,
                    'location': 6,
                    'history': 2,
                    'family_history': 2,
                },
                embedding_dim=64,
                hidden_dim=256,
                output_dim=fusion_dim,
            )
        
        # Multi-modal fusion
        if fusion_enabled:
            self.fusion = MultiModalFusion(
                image_dim=fusion_dim,
                clinical_dim=fusion_dim,
                fusion_dim=fusion_dim,
                num_attention_layers=2,
                num_heads=8,
            )
        
        # Fairness module
        if fairness_enabled:
            self.fairness_module = FairnessModule(
                feature_dim=fusion_dim,
                num_skin_tones=num_skin_tones,
                debias_weight=debias_weight,
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_classes),
        )
        
        # Lesion type head (optional)
        self.lesion_type_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, lesion_types),
        )
        
        # Uncertainty module
        if uncertainty_enabled:
            self.uncertainty_module = UncertaintyModule(
                input_dim=fusion_dim,
                hidden_dim=256,
                output_dim=1,
                num_samples=num_samples,
            )
    
    def forward(
        self,
        image: torch.Tensor,
        clinical_features: Optional[Dict[str, torch.Tensor]] = None,
        return_uncertainty: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image: Input image (B, 3, H, W)
            clinical_features: Dictionary of clinical features (optional)
            return_uncertainty: Whether to compute uncertainty
        
        Returns:
            Dictionary with:
                - logits: Classification logits (B, num_classes)
                - abcde_scores: ABCDE scores (B, 5) if enabled
                - attention_maps: Attention maps if enabled
                - uncertainty: Uncertainty estimates if enabled
        """
        # Backbone
        cnn_features, vit_cls, vit_patches = self.backbone(image)
        # cnn_features: (B, fusion_dim, H', W')
        # vit_cls: (B, fusion_dim)
        # vit_patches: (B, num_patches, fusion_dim)
        
        # ABCDE feature extraction
        abcde_outputs = None
        if self.abcde_enabled:
            abcde_outputs = self.abcde_module(cnn_features)
            abcde_scores = abcde_outputs['abcde_scores']
            abcde_feat_dict = abcde_outputs['features']
        else:
            abcde_scores = None
            abcde_feat_dict = None
        
        # Reshape CNN features for attention
        B, C, H, W = cnn_features.shape
        cnn_sequence = cnn_features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, fusion_dim)
        
        # ABCDE attention
        if self.abcde_enabled and abcde_feat_dict:
            attended_features = self.abcde_attention(cnn_sequence, abcde_feat_dict)
            # Use CLS token or mean pooling
            pooled_features = attended_features.mean(dim=1)  # (B, fusion_dim)
        else:
            pooled_features = self.global_pool(cnn_features).squeeze(-1).squeeze(-1)  # (B, fusion_dim)
        
        # Combine with ViT CLS token
        pooled_features = (pooled_features + vit_cls) / 2  # (B, fusion_dim)
        
        # Clinical feature encoding
        clinical_encoded = None
        if self.fusion_enabled and clinical_features is not None:
            clinical_encoded = self.clinical_encoder(clinical_features)  # (B, fusion_dim)
        
        # Multi-modal fusion
        if self.fusion_enabled and clinical_encoded is not None:
            fused_features = self.fusion(pooled_features, clinical_encoded)
        else:
            fused_features = pooled_features
        
        # Fairness module (adversarial debiasing)
        if self.fairness_enabled:
            fused_features = self.fairness_module(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)
        lesion_type_logits = self.lesion_type_head(fused_features)  # (B, lesion_types)
        
        # Uncertainty estimation
        uncertainty_outputs = None
        if self.uncertainty_enabled and return_uncertainty:
            mean, epistemic, aleatoric = self.uncertainty_module(fused_features)
            uncertainty_outputs = {
                'mean': mean,
                'epistemic': epistemic,
                'aleatoric': aleatoric,
                'total': epistemic + aleatoric,
            }
        
        # Prepare output
        output = {
            'logits': logits,
            'lesion_type_logits': lesion_type_logits,
            'features': fused_features,
        }
        
        if abcde_scores is not None:
            output['abcde_scores'] = abcde_scores
        
        if abcde_outputs and 'attention_maps' in abcde_outputs:
            output['attention_maps'] = abcde_outputs['attention_maps']
        
        if uncertainty_outputs:
            output['uncertainty'] = uncertainty_outputs
        
        return output

