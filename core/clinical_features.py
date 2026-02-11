"""
Clinical feature encoding for multi-modal fusion.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class ClinicalFeatureEncoder(nn.Module):
    """
    Encoder for clinical metadata features.
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ):
        """
        Initialize clinical feature encoder.
        
        Args:
            feature_dims: Dictionary mapping feature names to their dimensions
                e.g., {'age': 1, 'gender': 2, 'location': 6}
            embedding_dim: Embedding dimension for categorical features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        self.numeric_features = []
        self.categorical_features = []
        
        total_dim = 0
        
        for feature_name, dim in feature_dims.items():
            if dim == 1:
                # Numeric feature
                self.numeric_features.append(feature_name)
                total_dim += 1
            else:
                # Categorical feature
                self.categorical_features.append(feature_name)
                self.embeddings[feature_name] = nn.Embedding(dim, embedding_dim)
                total_dim += embedding_dim
        
        # MLP to process concatenated features
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode clinical features.
        
        Args:
            features: Dictionary mapping feature names to tensors
                - Numeric features: (batch_size, 1)
                - Categorical features: (batch_size,) with integer values
        
        Returns:
            Encoded features: (batch_size, output_dim)
        """
        encoded_features = []
        
        # Process numeric features
        for feature_name in self.numeric_features:
            if feature_name in features:
                encoded_features.append(features[feature_name])
        
        # Process categorical features
        for feature_name in self.categorical_features:
            if feature_name in features:
                embedded = self.embeddings[feature_name](features[feature_name].long())
                encoded_features.append(embedded)
        
        # Concatenate all features
        if encoded_features:
            concatenated = torch.cat(encoded_features, dim=-1)
        else:
            # If no features, return zeros
            batch_size = next(iter(features.values())).shape[0]
            concatenated = torch.zeros(batch_size, 0, device=next(iter(features.values())).device)
        
        # Pass through MLP
        output = self.mlp(concatenated)
        
        return output


def create_default_clinical_encoder(output_dim: int = 512) -> ClinicalFeatureEncoder:
    """
    Create default clinical feature encoder with common features.
    
    Args:
        output_dim: Output dimension
    
    Returns:
        ClinicalFeatureEncoder instance
    """
    feature_dims = {
        'age': 1,           # Numeric
        'gender': 2,        # Categorical: 0=male, 1=female
        'location': 6,      # Categorical: face regions
        'history': 2,      # Categorical: 0=no history, 1=history
        'family_history': 2,  # Categorical: 0=no, 1=yes
    }
    
    return ClinicalFeatureEncoder(
        feature_dims=feature_dims,
        embedding_dim=64,
        hidden_dim=256,
        output_dim=output_dim,
    )


def encode_clinical_features(
    features: Dict[str, np.ndarray],
    encoder: ClinicalFeatureEncoder,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Encode clinical features from numpy arrays.
    
    Args:
        features: Dictionary mapping feature names to numpy arrays
        encoder: ClinicalFeatureEncoder instance
        device: Device to run on
    
    Returns:
        Encoded features tensor
    """
    # Convert to tensors
    tensor_features = {}
    for key, value in features.items():
        tensor = torch.from_numpy(np.array(value)).to(device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)  # Add feature dimension for numeric
        tensor_features[key] = tensor
    
    # Encode
    with torch.no_grad():
        encoded = encoder(tensor_features)
    
    return encoded

