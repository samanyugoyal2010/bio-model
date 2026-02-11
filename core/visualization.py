"""
Visualization utilities for results, attention maps, and ABCDE scores.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
import seaborn as sns


def plot_attention_maps(
    image: np.ndarray,
    attention_maps: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Plot attention maps overlaid on image.
    
    Args:
        image: Input image (H, W, 3)
        attention_maps: Dictionary mapping attention names to maps (H, W)
        save_path: Path to save figure
        figsize: Figure size
    """
    num_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_maps + 1, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention maps
    for idx, (name, att_map) in enumerate(attention_maps.items(), 1):
        # Resize attention map to image size if needed
        if att_map.shape != image.shape[:2]:
            from scipy.ndimage import zoom
            zoom_factors = (
                image.shape[0] / att_map.shape[0],
                image.shape[1] / att_map.shape[1],
            )
            att_map = zoom(att_map, zoom_factors)
        
        # Overlay attention on image
        axes[idx].imshow(image)
        im = axes[idx].imshow(
            att_map,
            alpha=0.5,
            cmap='jet',
            interpolation='bilinear',
        )
        axes[idx].set_title(f'{name} Attention')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_abcde_scores(
    abcde_scores: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot ABCDE scores as bar chart.
    
    Args:
        abcde_scores: Dictionary with 'A', 'B', 'C', 'D', 'E' scores
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    criteria = ['Asymmetry', 'Border', 'Color', 'Diameter', 'Evolution']
    scores = [
        abcde_scores.get('A', 0.0),
        abcde_scores.get('B', 0.0),
        abcde_scores.get('C', 0.0),
        abcde_scores.get('D', 0.0),
        abcde_scores.get('E', 0.0),
    ]
    
    bars = ax.bar(criteria, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('ABCDE Rule Scores', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_uncertainty(
    image: np.ndarray,
    prediction: float,
    uncertainty: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot prediction with uncertainty visualization.
    
    Args:
        image: Input image
        prediction: Prediction probability (0-1)
        uncertainty: Uncertainty value (0-1)
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].barh(['Benign', 'Malignant'], [1 - prediction, prediction], color=['green', 'red'])
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Prediction: {prediction:.3f}')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Uncertainty
    uncertainty_color = plt.cm.RdYlGn_r(uncertainty)  # Red for high uncertainty, green for low
    axes[2].barh(['Uncertainty'], [uncertainty], color=uncertainty_color)
    axes[2].set_xlim([0, 1])
    axes[2].set_xlabel('Uncertainty')
    axes[2].set_title(f'Uncertainty: {uncertainty:.3f}')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Plot training history (loss, metrics over epochs).
    
    Args:
        history: Dictionary with metric names and lists of values
        save_path: Path to save figure
        figsize: Figure size
    """
    num_metrics = len(history)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(history.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_fairness_analysis(
    fairness_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot fairness metrics across different groups.
    
    Args:
        fairness_metrics: Dictionary mapping metric names to group values
            e.g., {'accuracy': {'group1': 0.9, 'group2': 0.85}, ...}
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(fairness_metrics), figsize=figsize)
    
    if len(fairness_metrics) == 1:
        axes = [axes]
    
    for idx, (metric_name, group_values) in enumerate(fairness_metrics.items()):
        groups = list(group_values.keys())
        values = list(group_values.values())
        
        bars = axes[idx].bar(groups, values, color=plt.cm.Set3(range(len(groups))))
        axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

