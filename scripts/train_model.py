"""
Training script.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from models.skin_cancer_model import SkinCancerModel
from data.dataset import SkinCancerDataset, MultiModalSkinCancerDataset
from data.augmentation import get_augmentation
from core import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
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
    
    # Create datasets
    train_dataset = SkinCancerDataset(
        data_dir=config['data']['processed_data_dir'] + '/train',
        image_size=config['data']['image_size'],
        transform=get_augmentation(
            image_size=config['data']['image_size'],
            is_training=True,
        ),
        is_training=True,
    )
    
    val_dataset = SkinCancerDataset(
        data_dir=config['data']['processed_data_dir'] + '/val',
        image_size=config['data']['image_size'],
        transform=get_augmentation(
            image_size=config['data']['image_size'],
            is_training=False,
        ),
        is_training=False,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    # Create optimizer
    training_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        **training_config['optimizer_params'],
    )
    
    # Create scheduler
    scheduler = None
    if training_config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=training_config['scheduler_params']['min_lr'],
        )
    
    # Create trainer
    discriminator_optimizer = None
    if model_config['fairness']['enabled']:
        discriminator_optimizer = torch.optim.Adam(
            model.fairness_module.discriminator.parameters(),
            lr=training_config['learning_rate'],
        )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=f"{config['paths']['checkpoints']}/{args.experiment_name}",
        classification_weight=training_config['loss_weights']['classification'],
        abcde_weight=training_config['loss_weights']['abcde'],
        fairness_weight=training_config['loss_weights']['fairness'],
        use_abcde=model_config['abcde']['enabled'],
        use_fairness=model_config['fairness']['enabled'],
        discriminator_optimizer=discriminator_optimizer,
    )
    
    # Resume from checkpoint
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Training loop
    num_epochs = training_config['num_epochs']
    best_val_loss = float('inf')
    patience = training_config['early_stopping']['patience']
    patience_counter = 0
    
    for epoch in range(num_epochs):
        trainer.current_epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Early stopping
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_checkpoint('best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint']['save_frequency'] == 0:
            trainer.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
    
    print("Training complete!")


if __name__ == '__main__':
    main()

