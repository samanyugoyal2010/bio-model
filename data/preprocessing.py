"""
Face detection and preprocessing for facial skin cancer images.
Includes facial landmark detection and face region extraction.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import dlib
from pathlib import Path


class FacePreprocessor:
    """
    Preprocessor for facial images with landmark detection.
    """
    
    def __init__(
        self,
        detect_faces: bool = True,
        extract_landmarks: bool = True,
        landmark_model_path: Optional[str] = None,
        target_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initialize face preprocessor.
        
        Args:
            detect_faces: Whether to detect and crop faces
            extract_landmarks: Whether to extract 68-point facial landmarks
            landmark_model_path: Path to dlib shape predictor model
            target_size: Target image size (width, height)
        """
        self.detect_faces = detect_faces
        self.extract_landmarks = extract_landmarks
        self.target_size = target_size
        
        # Initialize face detector
        self.face_detector = None
        self.landmark_predictor = None
        
        if detect_faces or extract_landmarks:
            # Try to load dlib models
            try:
                if landmark_model_path:
                    model_path = Path(landmark_model_path)
                else:
                    # Try default locations
                    model_path = Path("./models/shape_predictor_68_face_landmarks.dat")
                    if not model_path.exists():
                        model_path = Path("./shape_predictor_68_face_landmarks.dat")
                
                if model_path.exists():
                    self.face_detector = dlib.get_frontal_face_detector()
                    self.landmark_predictor = dlib.shape_predictor(str(model_path))
                else:
                    print(f"Warning: Landmark model not found at {model_path}")
                    print("Face detection will use OpenCV Haar cascades instead.")
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
            except Exception as e:
                print(f"Warning: Could not load dlib models: {e}")
                print("Falling back to OpenCV face detection.")
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
    
    def detect_face_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Return largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            return (x, y, x + w, y + h)
        
        return None
    
    def detect_face_dlib(self, image: np.ndarray) -> Optional[dlib.rectangle]:
        """Detect face using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray, 1)
        
        if len(faces) > 0:
            # Return largest face
            return max(faces, key=lambda rect: rect.width() * rect.height())
        
        return None
    
    def extract_landmarks_dlib(
        self, image: np.ndarray, face_rect: dlib.rectangle
    ) -> np.ndarray:
        """Extract 68-point facial landmarks using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        landmarks = self.landmark_predictor(gray, face_rect)
        
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return points
    
    def normalize_landmarks(
        self, landmarks: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Normalize landmarks to [0, 1] range."""
        h, w = image_shape[:2]
        normalized = landmarks.copy()
        normalized[:, 0] /= w
        normalized[:, 1] /= h
        return normalized
    
    def crop_face(
        self, image: np.ndarray, face_region: Tuple[int, int, int, int], padding: float = 0.2
    ) -> np.ndarray:
        """Crop face region with padding."""
        x1, y1, x2, y2 = face_region
        h, w = image.shape[:2]
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        pad_w = int(width * padding)
        pad_h = int(height * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def process(self, image: np.ndarray) -> Dict:
        """
        Process image: detect face, extract landmarks, crop and resize.
        
        Args:
            image: Input image as numpy array (RGB)
        
        Returns:
            Dictionary with:
                - image: Processed image
                - landmarks: Facial landmarks (68 points, normalized) or None
                - face_region: Face bounding box (x1, y1, x2, y2) or None
        """
        result = {
            'image': image,
            'landmarks': None,
            'face_region': None,
        }
        
        if not self.detect_faces and not self.extract_landmarks:
            # Just resize
            result['image'] = self.resize_image(image)
            return result
        
        # Detect face
        face_rect = None
        face_bbox = None
        
        if isinstance(self.face_detector, dlib.fhog_object_detector):
            # dlib detector
            face_rect = self.detect_face_dlib(image)
            if face_rect:
                face_bbox = (
                    face_rect.left(), face_rect.top(),
                    face_rect.right(), face_rect.bottom()
                )
        else:
            # OpenCV detector
            face_bbox = self.detect_face_opencv(image)
        
        if face_bbox:
            result['face_region'] = face_bbox
            
            # Extract landmarks
            if self.extract_landmarks and self.landmark_predictor and face_rect:
                landmarks = self.extract_landmarks_dlib(image, face_rect)
                result['landmarks'] = self.normalize_landmarks(
                    landmarks, image.shape
                )
            
            # Crop face if requested
            if self.detect_faces:
                image = self.crop_face(image, face_bbox)
        
        # Resize to target size
        result['image'] = self.resize_image(image)
        
        return result
    
    def analyze_facial_symmetry(
        self, landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze facial symmetry for asymmetry detection.
        
        Args:
            landmarks: 68-point facial landmarks (normalized)
        
        Returns:
            Dictionary with symmetry metrics
        """
        if landmarks is None or len(landmarks) != 68:
            return {'symmetry_score': 0.0, 'asymmetry_score': 1.0}
        
        # Key facial points for symmetry analysis
        # Left and right eye corners, nose tip, mouth corners
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        
        # Calculate symmetry by comparing left and right features
        eye_symmetry = np.abs(left_eye[0] - (1.0 - right_eye[0]))
        mouth_symmetry = np.abs(left_mouth[0] - (1.0 - right_mouth[0]))
        
        # Overall symmetry score (lower is more symmetric)
        asymmetry_score = (eye_symmetry + mouth_symmetry) / 2.0
        symmetry_score = 1.0 - asymmetry_score
        
        return {
            'symmetry_score': symmetry_score,
            'asymmetry_score': asymmetry_score,
            'eye_symmetry': eye_symmetry,
            'mouth_symmetry': mouth_symmetry,
        }

