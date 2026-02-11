"""
PyTorch dataset classes for skin cancer detection.
Supports single-modal (image only) and multi-modal (image + clinical data) datasets.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

from .preprocessing import FacePreprocessor
from .augmentation import FaceAwareAugmentation


class SkinCancerDataset(Dataset):
    """
    Base dataset class for skin cancer images.
    Supports facial landmark detection and face-aware preprocessing.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        image_size: int = 512,
        transform: Optional[callable] = None,
        face_detection: bool = True,
        facial_landmarks: bool = True,
        is_training: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            metadata_file: CSV file with metadata (image_id, label, etc.)
            image_size: Target image size
            transform: Optional transform pipeline
            face_detection: Whether to detect faces
            facial_landmarks: Whether to extract facial landmarks
            is_training: Whether dataset is for training (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training
        
        # Initialize face preprocessor
        self.face_preprocessor = None
        if face_detection or facial_landmarks:
            self.face_preprocessor = FacePreprocessor(
                detect_faces=face_detection,
                extract_landmarks=facial_landmarks,
            )
        
        # Load metadata
        self.metadata = None
        self.image_paths = []
        self.labels = []
        self.lesion_types = []
        
        if metadata_file:
            self._load_metadata(metadata_file)
        else:
            self._load_from_directory()
    
    def _load_metadata(self, metadata_file: Union[str, Path]):
        """Load dataset from metadata CSV file."""
        df = pd.read_csv(metadata_file)
        
        for _, row in df.iterrows():
            image_id = row.get('image_id', row.get('id', ''))
            image_path = self.data_dir / f"{image_id}.jpg"
            
            if not image_path.exists():
                image_path = self.data_dir / f"{image_id}.png"
            
            if image_path.exists():
                self.image_paths.append(image_path)
                
                # Labels: 0 = benign, 1 = malignant
                label = row.get('label', row.get('target', 0))
                if isinstance(label, str):
                    label = 1 if label.lower() in ['malignant', 'melanoma', 'cancer'] else 0
                self.labels.append(int(label))
                
                # Lesion type (optional)
                lesion_type = row.get('lesion_type', row.get('dx', 0))
                self.lesion_types.append(lesion_type)
    
    def _load_from_directory(self):
        """Load dataset from directory structure (images in subdirectories by class)."""
        benign_dir = self.data_dir / "benign"
        malignant_dir = self.data_dir / "malignant"
        
        if benign_dir.exists():
            for img_path in benign_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(0)
                self.lesion_types.append("benign")
            for img_path in benign_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(0)
                self.lesion_types.append("benign")
        
        if malignant_dir.exists():
            for img_path in malignant_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(1)
                self.lesion_types.append("malignant")
            for img_path in malignant_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(1)
                self.lesion_types.append("malignant")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            Dictionary with:
                - image: Preprocessed image tensor
                - label: Binary label (0=benign, 1=malignant)
                - lesion_type: Lesion type (optional)
                - landmarks: Facial landmarks (if available)
                - face_region: Face region coordinates (if available)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        lesion_type = self.lesion_types[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Face preprocessing
        landmarks = None
        face_region = None
        if self.face_preprocessor:
            result = self.face_preprocessor.process(image)
            image = result['image']
            landmarks = result.get('landmarks')
            face_region = result.get('face_region')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # Already a tensor from transform
            pass
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'lesion_type': lesion_type,
        }
        
        if landmarks is not None:
            sample['landmarks'] = torch.tensor(landmarks, dtype=torch.float32)
        
        if face_region is not None:
            sample['face_region'] = torch.tensor(face_region, dtype=torch.float32)
        
        return sample


class MultiModalSkinCancerDataset(SkinCancerDataset):
    """
    Multi-modal dataset that includes clinical metadata along with images.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Union[str, Path],
        image_size: int = 512,
        transform: Optional[callable] = None,
        face_detection: bool = True,
        facial_landmarks: bool = True,
        is_training: bool = True,
        clinical_features: Optional[List[str]] = None,
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            clinical_features: List of clinical feature column names to include
                (e.g., ['age', 'gender', 'location', 'history'])
        """
        super().__init__(
            data_dir=data_dir,
            metadata_file=metadata_file,
            image_size=image_size,
            transform=transform,
            face_detection=face_detection,
            facial_landmarks=facial_landmarks,
            is_training=is_training,
        )
        
        self.clinical_features = clinical_features or ['age', 'gender', 'location']
        self.clinical_data = []
        self._load_clinical_data(metadata_file)
    
    def _load_clinical_data(self, metadata_file: Union[str, Path]):
        """Load clinical metadata."""
        df = pd.read_csv(metadata_file)
        
        for _, row in df.iterrows():
            clinical_dict = {}
            for feature in self.clinical_features:
                value = row.get(feature, 0.0)
                
                # Handle categorical features
                if feature == 'gender':
                    value = 0.0 if str(value).lower() in ['male', 'm', '0'] else 1.0
                elif feature == 'location':
                    # Encode location as numeric (simplified)
                    location_map = {
                        'face': 0.0, 'cheek': 0.1, 'nose': 0.2, 'forehead': 0.3,
                        'chin': 0.4, 'other': 0.5
                    }
                    value = location_map.get(str(value).lower(), 0.5)
                else:
                    # Numeric features (age, etc.)
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0.0
                
                clinical_dict[feature] = value
            
            self.clinical_data.append(clinical_dict)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with clinical features."""
        sample = super().__getitem__(idx)
        
        # Add clinical features
        clinical_dict = self.clinical_data[idx]
        clinical_tensor = torch.tensor(
            [clinical_dict[feat] for feat in self.clinical_features],
            dtype=torch.float32
        )
        sample['clinical_features'] = clinical_tensor
        
        return sample

