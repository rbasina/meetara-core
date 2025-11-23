"""
Facial Emotion Detection Tool for Meetara Core.

This tool provides facial emotion recognition using offline models like deepface,
mediapipe, and FER.
"""
import io
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logger import agent_logger


class FaceEmotionInput(BaseModel):
    """Input schema for facial emotion detection tool."""
    image_data: bytes = Field(description="Image data in bytes for facial emotion detection")


class FaceEmotionTool(BaseTool):
    """Tool for detecting emotions from facial expressions."""
    
    name: str = "face_emotion_tool"
    description: str = "Detect emotions from facial expressions using offline models"
    args_schema: type = FaceEmotionInput
    
    def __init__(self):
        super().__init__()
        self._emotion_model = None
        self._face_detector = None
        self._initialize_models()
    
    @property
    def emotion_model(self):
        return self._emotion_model
    
    @property
    def face_detector(self):
        return self._face_detector
    
    def _initialize_models(self):
        """Initialize facial emotion detection models."""
        try:
            # Try to initialize deepface
            from deepface import DeepFace
            
            self._emotion_model = "deepface"
            agent_logger.info("DeepFace emotion model initialized")
            
        except ImportError:
            agent_logger.warning("DeepFace not available, trying FER")
            try:
                from fer import FER
                self._emotion_model = FER(mtcnn=True)
                agent_logger.info("FER emotion model initialized")
                
            except ImportError:
                agent_logger.warning("FER not available, using fallback")
                self._emotion_model = None
        
        # Initialize face detector (OpenCV)
        try:
            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            agent_logger.info("Face detector initialized")
            
        except Exception as e:
            agent_logger.error(f"Failed to initialize face detector: {e}")
            self._face_detector = None
    
    def _run(self, image_data: bytes) -> Dict[str, Any]:
        """Run facial emotion detection on image data."""
        try:
            if not self.emotion_model:
                return self._fallback_emotion_detection(image_data)
            
            # Convert bytes to image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "error": "Invalid image data",
                    "faces_detected": 0
                }
            
            # Detect faces first
            faces = self._detect_faces(image)
            
            if not faces:
                return {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "error": "No faces detected",
                    "faces_detected": 0
                }
            
            # Analyze emotion for the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            emotion_result = self._analyze_emotion(image, largest_face)
            
            agent_logger.info(f"Detected emotion: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.3f})")
            
            return {
                **emotion_result,
                "faces_detected": len(faces),
                "model": self.emotion_model if isinstance(self.emotion_model, str) else "fer"
            }
            
        except Exception as e:
            agent_logger.error(f"Error in facial emotion detection: {e}")
            return self._fallback_emotion_detection(image_data)
    
    async def _arun(self, image_data: bytes) -> Dict[str, Any]:
        """Async run of facial emotion detection."""
        return self._run(image_data)
    
    def _detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in the image."""
        if self.face_detector is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist()
    
    def _analyze_emotion(self, image: np.ndarray, face_bbox: list) -> Dict[str, Any]:
        """Analyze emotion for a specific face region."""
        x, y, w, h = face_bbox
        
        if isinstance(self.emotion_model, str) and self.emotion_model == "deepface":
            return self._analyze_with_deepface(image, face_bbox)
        else:
            return self._analyze_with_fer(image, face_bbox)
    
    def _analyze_with_deepface(self, image: np.ndarray, face_bbox: list) -> Dict[str, Any]:
        """Analyze emotion using DeepFace."""
        try:
            from deepface import DeepFace
            
            # Create temporary file for image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, image)
                temp_file_path = temp_file.name
            
            try:
                # Analyze emotion
                result = DeepFace.analyze(
                    temp_file_path,
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                # Extract emotion and confidence
                if isinstance(result, list):
                    result = result[0]
                
                emotion = result['dominant_emotion']
                emotions = result['emotion']
                confidence = emotions[emotion] / 100.0
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "emotions": emotions
                }
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
                
        except Exception as e:
            agent_logger.error(f"Error in DeepFace analysis: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _analyze_with_fer(self, image: np.ndarray, face_bbox: list) -> Dict[str, Any]:
        """Analyze emotion using FER."""
        try:
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            
            # Analyze emotion
            result = self.emotion_model.predict(face_roi)
            
            if result:
                emotions = result[0]['emotions']
                emotion = max(emotions.items(), key=lambda x: x[1])[0]
                confidence = emotions[emotion] / 100.0
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "emotions": emotions
                }
            else:
                return {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "error": "No emotion detected"
                }
                
        except Exception as e:
            agent_logger.error(f"Error in FER analysis: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _fallback_emotion_detection(self, image_data: bytes) -> Dict[str, Any]:
        """Fallback emotion detection using basic image analysis."""
        try:
            # Convert bytes to image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "error": "Invalid image data",
                    "model": "fallback"
                }
            
            # Basic image analysis for emotion detection
            features = self._extract_image_features(image)
            emotion = self._classify_emotion_from_image_features(features)
            
            agent_logger.info(f"Fallback emotion detection: {emotion}")
            
            return {
                "emotion": emotion,
                "confidence": 0.5,  # Lower confidence for fallback
                "features": features,
                "model": "fallback"
            }
            
        except Exception as e:
            agent_logger.error(f"Error in fallback emotion detection: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "error": str(e),
                "model": "fallback"
            }
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic image features for emotion classification."""
        features = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            features["brightness"] = np.mean(gray)
            
            # Contrast
            features["contrast"] = np.std(gray)
            
            # Edge density (using Sobel)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features["edge_density"] = np.mean(edge_magnitude)
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            features["saturation"] = np.mean(hsv[:, :, 1])
            features["value"] = np.mean(hsv[:, :, 2])
            
        except Exception as e:
            agent_logger.warning(f"Error extracting image features: {e}")
            features = {
                "brightness": 128.0,
                "contrast": 50.0,
                "edge_density": 0.0,
                "saturation": 128.0,
                "value": 128.0
            }
        
        return features
    
    def _classify_emotion_from_image_features(self, features: Dict[str, float]) -> str:
        """Classify emotion based on image features."""
        brightness = features.get("brightness", 128.0)
        contrast = features.get("contrast", 50.0)
        edge_density = features.get("edge_density", 0.0)
        saturation = features.get("saturation", 128.0)
        
        # High brightness + high saturation = happy
        if brightness > 150 and saturation > 150:
            return "happy"
        
        # Low brightness + low contrast = sad
        elif brightness < 100 and contrast < 30:
            return "sad"
        
        # High edge density = angry/anxious
        elif edge_density > 50:
            return "angry"
        
        # High contrast + medium brightness = anxious
        elif contrast > 80 and 100 < brightness < 150:
            return "anxious"
        
        # Default to neutral
        else:
            return "neutral"
    
    def get_supported_emotions(self) -> Dict[str, Any]:
        """Get supported emotions and their descriptions."""
        return {
            "emotions": {
                "happy": "Positive, joyful, excited",
                "sad": "Negative, sorrowful, depressed", 
                "angry": "Hostile, frustrated, irritated",
                "anxious": "Worried, nervous, stressed",
                "neutral": "Calm, balanced, indifferent"
            },
            "model": self.emotion_model if isinstance(self.emotion_model, str) else "fer"
        } 