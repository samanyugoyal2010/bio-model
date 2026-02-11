"""
Fairness analysis script.
"""

import argparse
import yaml
import torch
import json
import pandas as pd
from pathlib import Path

from models.skin_cancer_model import SkinCancerModel
from data.dataset import SkinCancerDataset
from data.augmentation import get_augmentation
from core import Evaluator, plot_fairness_analysis


def main():
    parser = argparse.ArgumentParser(description='Analyze fairness of model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='./fairness_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['hardware']['device']
    
    # Create model
    model_config = config['model']
    model = SkinCancerModel(
        efficientnet_variant=model_config['backbone']['efficientnet_variant'],
        image_size=config['data']['image_size'],
        abcde_enabled=model_config['abcde']['enabled'],
        fusion_enabled=model_config['fusion']['enabled'],
        fairness_enabled=model_config['fairness']['enabled'],
        uncertainty_enabled=model_config['uncertainty']['enabled'],
        num_classes=model_config['outputs']['classification'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset
    test_dataset = SkinCancerDataset(
        data_dir=args.test_data,
        image_size=config['data']['image_size'],
        transform=get_augmentation(
            image_size=config['data']['image_size'],
            is_training=False,
        ),
        is_training=False,
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
    )
    
    # Evaluate with fairness metrics
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader, compute_fairness=True)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'fairness_metrics.json', 'w') as f:
        json.dump(metrics.get('fairness', {}), f, indent=2)
    
    # Plot fairness analysis
    if 'fairness' in metrics and 'group_metrics' in metrics['fairness']:
        group_metrics = metrics['fairness']['group_metrics']
        
        # Prepare data for plotting
        fairness_metrics_plot = {
            'accuracy': {str(k): v['accuracy'] for k, v in group_metrics.items()},
            'sensitivity': {str(k): v['sensitivity'] for k, v in group_metrics.items()},
            'specificity': {str(k): v['specificity'] for k, v in group_metrics.items()},
        }
        
        plot_fairness_analysis(
            fairness_metrics_plot,
            save_path=str(output_dir / 'fairness_analysis.png'),
        )
    
    print("Fairness analysis complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()

