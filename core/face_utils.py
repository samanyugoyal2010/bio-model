"""
Facial utilities including landmark detection, region extraction, and skin tone classification.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from enum import Enum, IntEnum


class FacialRegion(Enum):
    """Facial region types."""
    FOREHEAD = "forehead"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    NOSE = "nose"
    CHIN = "chin"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    MOUTH = "mouth"
    OTHER = "other"


class FitzpatrickScale(IntEnum):
    """Fitzpatrick skin type scale (I-VI)."""
    TYPE_I = 1
    TYPE_II = 2
    TYPE_III = 3
    TYPE_IV = 4
    TYPE_V = 5
    TYPE_VI = 6


class FaceUtils:
    """Utility class for facial analysis."""
    
    LANDMARK_INDICES = {
        'jaw': list(range(0, 17)),
        'right_eyebrow': list(range(17, 22)),
        'left_eyebrow': list(range(22, 27)),
        'nose': list(range(27, 36)),
        'right_eye': list(range(36, 42)),
        'left_eye': list(range(42, 48)),
        'mouth': list(range(48, 68)),
    }
    
    @staticmethod
    def get_region_from_landmarks(landmarks: np.ndarray, region: FacialRegion) -> np.ndarray:
        """Extract region-specific landmarks."""
        if region == FacialRegion.NOSE:
            return landmarks[27:36]
        elif region == FacialRegion.LEFT_EYE:
            return landmarks[42:48]
        elif region == FacialRegion.RIGHT_EYE:
            return landmarks[36:42]
        elif region == FacialRegion.MOUTH:
            return landmarks[48:68]
        elif region == FacialRegion.LEFT_CHEEK:
            return np.vstack([landmarks[0:3], landmarks[17:22]])
        elif region == FacialRegion.RIGHT_CHEEK:
            return np.vstack([landmarks[14:17], landmarks[22:27]])
        elif region == FacialRegion.FOREHEAD:
            return np.vstack([landmarks[17:27]])
        elif region == FacialRegion.CHIN:
            return landmarks[6:11]
        else:
            return landmarks
    
    @staticmethod
    def get_region_bounding_box(region_landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box for a region."""
        x_min = int(region_landmarks[:, 0].min())
        y_min = int(region_landmarks[:, 1].min())
        x_max = int(region_landmarks[:, 0].max())
        y_max = int(region_landmarks[:, 1].max())
        return (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def classify_lesion_region(landmarks: np.ndarray, lesion_center: Tuple[float, float]) -> FacialRegion:
        """Classify which facial region a lesion is in."""
        x, y = lesion_center
        
        nose_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.NOSE))
        left_eye_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.LEFT_EYE))
        right_eye_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.RIGHT_EYE))
        mouth_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.MOUTH))
        left_cheek_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.LEFT_CHEEK))
        right_cheek_bbox = FaceUtils.get_region_bounding_box(FaceUtils.get_region_from_landmarks(landmarks, FacialRegion.RIGHT_CHEEK))
        
        def point_in_bbox(point, bbox):
            x1, y1, x2, y2 = bbox
            return x1 <= point[0] <= x2 and y1 <= point[1] <= y2
        
        if point_in_bbox((x, y), nose_bbox):
            return FacialRegion.NOSE
        elif point_in_bbox((x, y), left_eye_bbox):
            return FacialRegion.LEFT_EYE
        elif point_in_bbox((x, y), right_eye_bbox):
            return FacialRegion.RIGHT_EYE
        elif point_in_bbox((x, y), mouth_bbox):
            return FacialRegion.MOUTH
        elif point_in_bbox((x, y), left_cheek_bbox):
            return FacialRegion.LEFT_CHEEK
        elif point_in_bbox((x, y), right_cheek_bbox):
            return FacialRegion.RIGHT_CHEEK
        else:
            return FacialRegion.OTHER


def extract_facial_regions(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[FacialRegion, np.ndarray]:
    """Extract all facial regions from landmarks."""
    regions = {}
    for region in FacialRegion:
        if region != FacialRegion.OTHER:
            region_landmarks = FaceUtils.get_region_from_landmarks(landmarks, region)
            regions[region] = region_landmarks
    return regions


def analyze_symmetry(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
    """Analyze facial symmetry for asymmetry detection."""
    h, w = image_shape[:2]
    normalized = landmarks.copy()
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    
    left_eye_center = normalized[36:42].mean(axis=0)
    right_eye_center = normalized[42:48].mean(axis=0)
    nose_tip = normalized[30]
    left_mouth = normalized[48]
    right_mouth = normalized[54]
    
    eye_symmetry = np.abs(left_eye_center[0] - (1.0 - right_eye_center[0]))
    mouth_symmetry = np.abs(left_mouth[0] - (1.0 - right_mouth[0]))
    eye_nose_alignment = np.abs((left_eye_center[1] + right_eye_center[1]) / 2 - nose_tip[1])
    
    asymmetry_score = (eye_symmetry + mouth_symmetry + eye_nose_alignment) / 3.0
    symmetry_score = 1.0 - asymmetry_score
    
    return {
        'symmetry_score': float(symmetry_score),
        'asymmetry_score': float(asymmetry_score),
        'eye_symmetry': float(eye_symmetry),
        'mouth_symmetry': float(mouth_symmetry),
        'eye_nose_alignment': float(eye_nose_alignment),
    }


def estimate_lesion_diameter(landmarks: np.ndarray, lesion_bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
    """Estimate lesion diameter normalized by facial landmarks."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = lesion_bbox
    lesion_width = (x2 - x1) / w
    lesion_height = (y2 - y1) / h
    lesion_diameter = np.sqrt(lesion_width ** 2 + lesion_height ** 2)
    
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    inter_ocular_distance = np.linalg.norm(left_eye - right_eye) / w
    normalized_diameter = lesion_diameter / (inter_ocular_distance + 1e-6)
    
    return float(normalized_diameter)


class SkinToneClassifier:
    """Classifier for skin tone using Fitzpatrick scale."""
    
    def __init__(self):
        self.fitzpatrick_ranges = {
            FitzpatrickScale.TYPE_I: {'L': (85, 100), 'a': (-5, 5), 'b': (5, 15)},
            FitzpatrickScale.TYPE_II: {'L': (75, 85), 'a': (-3, 7), 'b': (8, 18)},
            FitzpatrickScale.TYPE_III: {'L': (65, 75), 'a': (0, 10), 'b': (12, 22)},
            FitzpatrickScale.TYPE_IV: {'L': (55, 65), 'a': (5, 15), 'b': (15, 25)},
            FitzpatrickScale.TYPE_V: {'L': (40, 55), 'a': (8, 18), 'b': (18, 28)},
            FitzpatrickScale.TYPE_VI: {'L': (20, 40), 'a': (10, 20), 'b': (20, 30)},
        }
    
    def extract_skin_region(self, image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract skin region from face."""
        if landmarks is None:
            return np.ones((image.shape[0], image.shape[1]), dtype=bool)
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        left_cheek_points = np.array([landmarks[1:4], landmarks[17:22]]).reshape(-1, 2).astype(int)
        right_cheek_points = np.array([landmarks[14:17], landmarks[22:27]]).reshape(-1, 2).astype(int)
        
        cv2.fillPoly(mask, [left_cheek_points], True)
        cv2.fillPoly(mask, [right_cheek_points], True)
        
        return mask
    
    def analyze_skin_color(self, image: np.ndarray, skin_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Analyze skin color in LAB color space."""
        if skin_mask is None:
            skin_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
        
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        skin_pixels = lab_image[skin_mask]
        
        if len(skin_pixels) == 0:
            return {'L': 50.0, 'a': 0.0, 'b': 0.0}
        
        return {
            'L': float(skin_pixels[:, 0].mean()),
            'a': float(skin_pixels[:, 1].mean()),
            'b': float(skin_pixels[:, 2].mean()),
        }
    
    def classify_fitzpatrick(self, image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Tuple[FitzpatrickScale, float, Dict]:
        """Classify skin tone using Fitzpatrick scale."""
        skin_mask = self.extract_skin_region(image, landmarks)
        color_stats = self.analyze_skin_color(image, skin_mask)
        
        scores = {}
        for fitz_type, ranges in self.fitzpatrick_ranges.items():
            L_center = (ranges['L'][0] + ranges['L'][1]) / 2
            a_center = (ranges['a'][0] + ranges['a'][1]) / 2
            b_center = (ranges['b'][0] + ranges['b'][1]) / 2
            
            L_dist = abs(color_stats['L'] - L_center) / (ranges['L'][1] - ranges['L'][0] + 1e-6)
            a_dist = abs(color_stats['a'] - a_center) / (ranges['a'][1] - ranges['a'][0] + 1e-6)
            b_dist = abs(color_stats['b'] - b_center) / (ranges['b'][1] - ranges['b'][0] + 1e-6)
            
            score = 1.0 / (1.0 + L_dist + a_dist + b_dist)
            scores[fitz_type] = score
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence, color_stats


def classify_fitzpatrick(image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Tuple[int, float]:
    """Convenience function to classify skin tone."""
    classifier = SkinToneClassifier()
    fitz_type, confidence, _ = classifier.classify_fitzpatrick(image, landmarks)
    return int(fitz_type), confidence

