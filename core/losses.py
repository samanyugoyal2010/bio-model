"""
All loss functions for skin cancer detection.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class ABCDELoss(nn.Module):
    """Loss function for ABCDE scoring."""
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute ABCDE loss."""
        if targets is None:
            return {
                'total_loss': torch.tensor(0.0, device=predictions.device),
                'individual_losses': {
                    'A': torch.tensor(0.0, device=predictions.device),
                    'B': torch.tensor(0.0, device=predictions.device),
                    'C': torch.tensor(0.0, device=predictions.device),
                    'D': torch.tensor(0.0, device=predictions.device),
                    'E': torch.tensor(0.0, device=predictions.device),
                },
            }
        
        criterion_names = ['A', 'B', 'C', 'D', 'E']
        individual_losses = {}
        
        for idx, name in enumerate(criterion_names):
            pred_criterion = predictions[:, idx]
            target_criterion = targets[:, idx]
            loss = self.mse_loss(pred_criterion, target_criterion)
            individual_losses[name] = loss
        
        total_loss = sum(individual_losses.values()) * self.weight
        
        return {
            'total_loss': total_loss,
            'individual_losses': individual_losses,
        }


class CombinedABCDELoss(nn.Module):
    """Combined loss for ABCDE scoring with classification."""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        abcde_weight: float = 0.5,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.abcde_weight = abcde_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.abcde_loss = ABCDELoss(weight=1.0, reduction=reduction)
    
    def forward(
        self,
        classification_logits: torch.Tensor,
        classification_targets: torch.Tensor,
        abcde_predictions: Optional[torch.Tensor] = None,
        abcde_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        cls_loss = self.ce_loss(classification_logits, classification_targets)
        abcde_loss_dict = self.abcde_loss(abcde_predictions, abcde_targets)
        abcde_loss = abcde_loss_dict['total_loss']
        
        total_loss = (
            self.classification_weight * cls_loss +
            self.abcde_weight * abcde_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'abcde_loss': abcde_loss,
            'abcde_individual': abcde_loss_dict['individual_losses'],
        }


class FairnessLoss(nn.Module):
    """Adversarial fairness loss."""
    
    def __init__(self, weight: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        skin_tone_logits: torch.Tensor,
        skin_tone_targets: torch.Tensor,
        adversarial: bool = True,
    ) -> torch.Tensor:
        """Compute fairness loss."""
        loss = self.ce_loss(skin_tone_logits, skin_tone_targets)
        
        if adversarial:
            loss = -loss * self.weight
        else:
            loss = loss * self.weight
        
        return loss


class UncertaintyLoss(nn.Module):
    """Loss function that incorporates uncertainty."""
    
    def __init__(self, kl_weight: float = 0.01, reduction: str = 'mean'):
        super().__init__()
        self.kl_weight = kl_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epistemic_uncertainty: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor,
        kl_divergence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty-aware loss."""
        precision = 1.0 / (aleatoric_uncertainty + 1e-8)
        weighted_loss = precision * (predictions - targets) ** 2
        
        if self.reduction == 'mean':
            prediction_loss = weighted_loss.mean()
        elif self.reduction == 'sum':
            prediction_loss = weighted_loss.sum()
        else:
            prediction_loss = weighted_loss
        
        kl_loss = torch.tensor(0.0, device=predictions.device)
        if kl_divergence is not None:
            kl_loss = kl_divergence * self.kl_weight
        
        total_loss = prediction_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'kl_loss': kl_loss,
            'epistemic_uncertainty': epistemic_uncertainty.mean(),
            'aleatoric_uncertainty': aleatoric_uncertainty.mean(),
        }

