"""
Comprehensive evaluation pipeline.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from .metrics import (
    compute_classification_metrics,
    compute_abcde_metrics,
    compute_uncertainty_metrics,
    compute_fairness_metrics,
)


class Evaluator:
    """Comprehensive evaluator for skin cancer detection model."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        compute_fairness: bool = False,
    ) -> Dict:
        """Evaluate model on dataset."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_abcde_scores = []
        all_abcde_targets = []
        all_uncertainties = []
        all_groups = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                
                outputs = self.model(
                    image=image,
                    clinical_features=batch.get('clinical_features'),
                )
                
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(label.cpu().numpy())
                all_probabilities.append(probabilities[:, 1].cpu().numpy())
                
                if 'abcde_scores' in outputs:
                    all_abcde_scores.append(outputs['abcde_scores'].cpu().numpy())
                
                if 'abcde_targets' in batch:
                    all_abcde_targets.append(batch['abcde_targets'].cpu().numpy())
                
                if 'uncertainty' in outputs:
                    uncertainty = outputs['uncertainty']
                    if 'total' in uncertainty:
                        all_uncertainties.append(uncertainty['total'].cpu().numpy())
                
                if compute_fairness and 'skin_tone' in batch:
                    all_groups.append(batch['skin_tone'].cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_probabilities = np.concatenate(all_probabilities)
        
        metrics = {}
        metrics['classification'] = compute_classification_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        if all_abcde_scores:
            all_abcde_scores = np.concatenate(all_abcde_scores)
            abcde_targets = None
            if all_abcde_targets:
                abcde_targets = np.concatenate(all_abcde_targets)
            metrics['abcde'] = compute_abcde_metrics(all_abcde_scores, abcde_targets)
        
        if all_uncertainties:
            all_uncertainties = np.concatenate(all_uncertainties)
            metrics['uncertainty'] = compute_uncertainty_metrics(
                all_probabilities, all_targets, all_uncertainties
            )
        
        if compute_fairness and all_groups:
            all_groups = np.concatenate(all_groups)
            metrics['fairness'] = compute_fairness_metrics(
                all_predictions, all_targets, all_groups, all_probabilities
            )
        
        return metrics

