"""
Core module containing training, losses, evaluation, explainability, and utilities.
"""

from .trainer import Trainer
from .losses import ABCDELoss, CombinedABCDELoss, FairnessLoss, UncertaintyLoss
from .metrics import (
    compute_classification_metrics,
    compute_abcde_metrics,
    compute_uncertainty_metrics,
    compute_fairness_metrics,
)
from .evaluator import Evaluator
from .explainability import GradCAM, ABCDEExplainer, UncertaintyVisualizer
from .face_utils import (
    FaceUtils,
    FacialRegion,
    FitzpatrickScale,
    SkinToneClassifier,
    extract_facial_regions,
    analyze_symmetry,
    estimate_lesion_diameter,
    classify_fitzpatrick,
)
from .clinical_features import ClinicalFeatureEncoder, create_default_clinical_encoder
from .visualization import (
    plot_attention_maps,
    plot_abcde_scores,
    plot_uncertainty,
    plot_training_history,
    plot_fairness_analysis,
)

__all__ = [
    # Trainer
    'Trainer',
    # Losses
    'ABCDELoss',
    'CombinedABCDELoss',
    'FairnessLoss',
    'UncertaintyLoss',
    # Metrics
    'compute_classification_metrics',
    'compute_abcde_metrics',
    'compute_uncertainty_metrics',
    'compute_fairness_metrics',
    # Evaluator
    'Evaluator',
    # Explainability
    'GradCAM',
    'ABCDEExplainer',
    'UncertaintyVisualizer',
    # Face utils
    'FaceUtils',
    'FacialRegion',
    'FitzpatrickScale',
    'SkinToneClassifier',
    'extract_facial_regions',
    'analyze_symmetry',
    'estimate_lesion_diameter',
    'classify_fitzpatrick',
    # Clinical features
    'ClinicalFeatureEncoder',
    'create_default_clinical_encoder',
    # Visualization
    'plot_attention_maps',
    'plot_abcde_scores',
    'plot_uncertainty',
    'plot_training_history',
    'plot_fairness_analysis',
]

