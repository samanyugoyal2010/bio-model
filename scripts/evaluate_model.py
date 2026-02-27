"""
Evaluation script.
"""

import argparse
import yaml
import torch
import json
from pathlib import Path

from models.skin_cancer_model import SkinCancerModel
from data.dataset import SkinCancerDataset
from data.augmentation import get_augmentation
from core import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate skin cancer detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--compute_fairness', action='store_true', help='Compute fairness metrics')
    
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
        num_classes=model_config['num_classes'],
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
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader, compute_fairness=args.compute_fairness)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation complete!")
    print(f"Results saved to {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()

