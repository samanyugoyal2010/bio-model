"""
Multi-head attention modules for ABCDE features and cross-modal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadABCDEAttention(nn.Module):
    """
    Multi-head attention with separate heads for each ABCDE feature.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head ABCDE attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads (should be 5 for ABCDE)
            head_dim: Dimension per head
            dropout: Dropout rate
        """
        super().__init__()
        
        assert num_heads == 5, "Should have 5 heads for ABCDE"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Separate attention heads for each ABCDE criterion
        self.query_projs = nn.ModuleList([
            nn.Linear(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.value_projs = nn.ModuleList([
            nn.Linear(embed_dim, head_dim) for _ in range(num_heads)
        ])
        
        self.output_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        abcde_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Base features (B, N, embed_dim)
            abcde_features: Dictionary with ABCDE feature tensors
        
        Returns:
            Attended features: (B, N, embed_dim)
        """
        B, N, _ = features.shape
        
        # Process each ABCDE head
        head_outputs = []
        
        abcde_keys = ['asymmetry', 'border', 'color', 'diameter', 'evolution']
        
        for idx, key in enumerate(abcde_keys):
            abcde_feat = abcde_features[key]  # (B, feature_dim, H, W)
            
            # Reshape to sequence
            B_f, C, H, W = abcde_feat.shape
            abcde_seq = abcde_feat.view(B_f, C, H * W).permute(0, 2, 1)  # (B, H*W, feature_dim)
            
            # Project
            Q = self.query_projs[idx](features)  # (B, N, head_dim)
            K = self.key_projs[idx](abcde_seq)  # (B, H*W, head_dim)
            V = self.value_projs[idx](abcde_seq)  # (B, H*W, head_dim)
            
            # Scaled dot-product attention
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention
            head_output = torch.bmm(attn_weights, V)  # (B, N, head_dim)
            head_outputs.append(head_output)
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, N, num_heads * head_dim)
        
        # Output projection
        output = self.output_proj(concatenated)
        output = self.dropout(output)
        output = self.norm(output + features)  # Residual connection
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing image and clinical features.
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        clinical_dim: int = 512,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            image_dim: Image feature dimension
            clinical_dim: Clinical feature dimension
            fusion_dim: Fusion dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Projections
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.clinical_proj = nn.Linear(clinical_dim, fusion_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image_features: Image features (B, N, image_dim) or (B, image_dim)
            clinical_features: Clinical features (B, clinical_dim)
        
        Returns:
            Fused features: (B, N, fusion_dim) or (B, fusion_dim)
        """
        # Project
        image_proj = self.image_proj(image_features)  # (B, N, fusion_dim) or (B, fusion_dim)
        clinical_proj = self.clinical_proj(clinical_features)  # (B, fusion_dim)
        
        # Ensure clinical features have sequence dimension
        if clinical_proj.dim() == 2:
            clinical_proj = clinical_proj.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Cross-attention: image queries, clinical key/value
        if image_proj.dim() == 2:
            image_proj = image_proj.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Self-attention on image features with clinical context
        attended, _ = self.attention(
            query=image_proj,
            key=torch.cat([image_proj, clinical_proj], dim=1),
            value=torch.cat([image_proj, clinical_proj], dim=1),
        )
        
        # Residual and norm
        attended = self.norm1(attended + image_proj)
        
        # Feed-forward
        output = self.ffn(attended)
        output = self.norm2(output + attended)
        
        # Remove sequence dimension if input was 2D
        if image_features.dim() == 2:
            output = output.squeeze(1)
        
        return output

