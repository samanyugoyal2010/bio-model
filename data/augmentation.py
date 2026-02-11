"""
Face-aware data augmentation for facial skin cancer images.
Preserves facial structure while augmenting images.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceAwareAugmentation:
    """
    Face-aware augmentation that preserves facial structure.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        is_training: bool = True,
        probability: float = 0.8,
        lesion_preserving: bool = True,
    ):
        """
        Initialize face-aware augmentation.
        
        Args:
            image_size: Target image size
            is_training: Whether augmentation is for training
            probability: Probability of applying augmentation
            lesion_preserving: Whether to preserve lesion characteristics
        """
        self.image_size = image_size
        self.is_training = is_training
        self.probability = probability
        self.lesion_preserving = lesion_preserving
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline."""
        if not self.is_training:
            # Validation/test: only resize and normalize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        
        # Training: apply augmentations
        transforms = [
            # Geometric augmentations (face-aware)
            A.HorizontalFlip(p=0.5),  # Faces are generally symmetric
            A.Rotate(limit=15, p=0.3),  # Small rotations only
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3,
            ),
            
            # Color augmentations (lesion-preserving)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise and blur (moderate)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            
            # Resize and normalize
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
        
        return A.Compose(transforms, p=self.probability)
    
    def __call__(self, image: np.ndarray, **kwargs) -> Dict:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with augmented image
        """
        return self.transform(image=image, **kwargs)
    
    def apply_region_specific_augmentation(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        face_region: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Apply region-specific augmentations based on facial landmarks.
        Different augmentation strategies for different facial regions.
        
        Args:
            image: Input image
            landmarks: Facial landmarks (68 points)
            face_region: Face bounding box
        
        Returns:
            Augmented image
        """
        if landmarks is None:
            return self.transform(image=image)['image']
        
        # Identify facial regions
        # Nose region (points 27-35)
        # Cheek regions (points 1-16 for left, 17-26 for right)
        # Forehead (estimated from top of face)
        
        # For now, apply standard augmentation
        # In future, could implement region-specific strategies
        return self.transform(image=image)['image']


def get_augmentation(
    image_size: int = 512,
    is_training: bool = True,
    probability: float = 0.8,
) -> A.Compose:
    """
    Get augmentation pipeline.
    
    Args:
        image_size: Target image size
        is_training: Whether for training
        probability: Augmentation probability
    
    Returns:
        Albumentations compose object
    """
    aug = FaceAwareAugmentation(
        image_size=image_size,
        is_training=is_training,
        probability=probability,
    )
    return aug.transform

