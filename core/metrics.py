"""
All evaluation metrics including classification, ABCDE, uncertainty, and fairness.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Dict, Optional


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics."""
    accuracy = accuracy_score(targets, predictions)
    metrics = {'accuracy': float(accuracy)}
    
    if len(np.unique(targets)) == 2:
        precision = precision_score(targets, predictions, average='binary', zero_division=0)
        recall = recall_score(targets, predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, predictions, average='binary', zero_division=0)
        
        metrics.update({
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),
            'f1_score': float(f1),
        })
        
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        metrics.update({
            'ppv': float(ppv),
            'npv': float(npv),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        })
        
        if probabilities is not None:
            if probabilities.ndim > 1:
                probabilities = probabilities[:, 1]
            try:
                auc = roc_auc_score(targets, probabilities)
                metrics['auc_roc'] = float(auc)
            except ValueError:
                metrics['auc_roc'] = 0.0
    
    return metrics


def compute_abcde_metrics(
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute ABCDE scoring metrics."""
    metrics = {}
    if targets is None:
        return metrics
    
    criterion_names = ['A', 'B', 'C', 'D', 'E']
    for idx, name in enumerate(criterion_names):
        pred_criterion = predictions[:, idx]
        target_criterion = targets[:, idx]
        
        mse = np.mean((pred_criterion - target_criterion) ** 2)
        mae = np.mean(np.abs(pred_criterion - target_criterion))
        
        if np.std(pred_criterion) > 0 and np.std(target_criterion) > 0:
            correlation = np.corrcoef(pred_criterion, target_criterion)[0, 1]
        else:
            correlation = 0.0
        
        metrics[f'{name}_mse'] = float(mse)
        metrics[f'{name}_mae'] = float(mae)
        metrics[f'{name}_correlation'] = float(correlation)
    
    overall_mse = np.mean((predictions - targets) ** 2)
    overall_mae = np.mean(np.abs(predictions - targets))
    
    metrics.update({
        'overall_mse': float(overall_mse),
        'overall_mae': float(overall_mae),
    })
    
    return metrics


def compute_uncertainty_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
) -> Dict[str, float]:
    """Compute uncertainty metrics."""
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    brier_score = np.mean((predictions - targets) ** 2)
    epsilon = 1e-8
    nll = -np.mean(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
    
    return {
        'ece': float(ece),
        'brier_score': float(brier_score),
        'nll': float(nll),
        'mean_uncertainty': float(uncertainties.mean()),
    }


def compute_fairness_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute fairness metrics across groups."""
    unique_groups = np.unique(groups)
    metrics = {}
    group_metrics = {}
    
    for group in unique_groups:
        group_mask = groups == group
        group_preds = predictions[group_mask]
        group_targets = targets[group_mask]
        
        accuracy = (group_preds == group_targets).mean()
        
        if group_targets.sum() > 0:
            sensitivity = (group_preds[group_targets == 1] == 1).mean()
        else:
            sensitivity = 0.0
        
        if (group_targets == 0).sum() > 0:
            specificity = (group_preds[group_targets == 0] == 0).mean()
        else:
            specificity = 0.0
        
        auc = 0.0
        if probabilities is not None and len(np.unique(group_targets)) == 2:
            group_probs = probabilities[group_mask]
            try:
                auc = roc_auc_score(group_targets, group_probs)
            except ValueError:
                auc = 0.0
        
        group_metrics[group] = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'auc': float(auc),
        }
    
    metrics['group_metrics'] = group_metrics
    
    sensitivities = [group_metrics[g]['sensitivity'] for g in unique_groups]
    specificities = [group_metrics[g]['specificity'] for g in unique_groups]
    
    metrics['equalized_odds_tpr'] = float(np.std(sensitivities))
    metrics['equalized_odds_fpr'] = float(np.std(1 - np.array(specificities)))
    
    positive_rates = []
    for group in unique_groups:
        group_mask = groups == group
        positive_rate = predictions[group_mask].mean()
        positive_rates.append(positive_rate)
    
    metrics['demographic_parity'] = float(np.std(positive_rates))
    
    if probabilities is not None:
        calibration_scores = []
        for group in unique_groups:
            group_mask = groups == group
            group_probs = probabilities[group_mask]
            group_targets = targets[group_mask]
            
            if len(group_probs) > 0:
                calibration = np.abs(group_probs.mean() - group_targets.mean())
                calibration_scores.append(calibration)
        
        metrics['calibration_difference'] = float(np.std(calibration_scores)) if calibration_scores else 0.0
    
    return metrics

