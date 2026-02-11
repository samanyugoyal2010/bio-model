"""
Unified trainer with support for ABCDE and fairness training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os

from .losses import CombinedABCDELoss, FairnessLoss


class Trainer:
    """
    Unified trainer supporting ABCDE scoring and adversarial fairness training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        # Loss weights
        classification_weight: float = 1.0,
        abcde_weight: float = 0.5,
        fairness_weight: float = 0.1,
        # Feature flags
        use_abcde: bool = True,
        use_fairness: bool = False,
        # Fairness-specific
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save checkpoints
            classification_weight: Weight for classification loss
            abcde_weight: Weight for ABCDE loss
            fairness_weight: Weight for fairness loss
            use_abcde: Whether to use ABCDE loss
            use_fairness: Whether to use fairness training
            discriminator_optimizer: Optimizer for fairness discriminator (if use_fairness)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        self.use_abcde = use_abcde
        self.use_fairness = use_fairness
        self.discriminator_optimizer = discriminator_optimizer
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss criterion
        self.criterion = CombinedABCDELoss(
            classification_weight=classification_weight,
            abcde_weight=abcde_weight if use_abcde else 0.0,
        )
        
        if use_fairness:
            self.fairness_criterion = FairnessLoss(weight=fairness_weight)
        else:
            self.fairness_criterion = None
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch in pbar:
            batch = self._move_to_device(batch)
            loss_dict = self._train_step(batch)
            
            loss = loss_dict['total_loss']
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            
            for batch in pbar:
                batch = self._move_to_device(batch)
                loss_dict = self._validate_step(batch)
                
                loss = loss_dict['total_loss']
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}
    
    def _train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Training step."""
        # Forward pass
        outputs = self.model(
            image=batch['image'],
            clinical_features=batch.get('clinical_features'),
        )
        
        # Train discriminator (if fairness enabled)
        if self.use_fairness and self.discriminator_optimizer:
            if hasattr(self.model, 'fairness_module') and self.model.fairness_module:
                fairness_features = outputs.get('features', None)
                
                if fairness_features is not None and 'skin_tone' in batch:
                    self.discriminator_optimizer.zero_grad()
                    skin_tone_logits = self.model.fairness_module.predict_skin_tone(fairness_features)
                    fairness_loss = self.fairness_criterion(
                        skin_tone_logits,
                        batch['skin_tone'],
                        adversarial=False,
                    )
                    fairness_loss.backward()
                    self.discriminator_optimizer.step()
        
        # Compute main loss
        loss_dict = self.criterion(
            classification_logits=outputs['logits'],
            classification_targets=batch['label'],
            abcde_predictions=outputs.get('abcde_scores') if self.use_abcde else None,
            abcde_targets=batch.get('abcde_targets') if self.use_abcde else None,
        )
        
        # Add fairness loss (adversarial)
        if self.use_fairness and self.fairness_criterion:
            if hasattr(self.model, 'fairness_module') and self.model.fairness_module:
                fairness_features = outputs.get('features', None)
                if fairness_features is not None and 'skin_tone' in batch:
                    skin_tone_logits = self.model.fairness_module.predict_skin_tone(fairness_features)
                    adversarial_fairness_loss = self.fairness_criterion(
                        skin_tone_logits,
                        batch['skin_tone'],
                        adversarial=True,
                    )
                    loss_dict['total_loss'] = loss_dict['total_loss'] + adversarial_fairness_loss
                    loss_dict['fairness_loss'] = adversarial_fairness_loss
        
        return loss_dict
    
    def _validate_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Validation step."""
        return self._train_step(batch)
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {filepath}")

