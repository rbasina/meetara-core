"""
Emotion Detection Tool for Meetara Core.

This tool provides speech emotion recognition using offline models like SpeechBrain.
"""
import io
import tempfile
import numpy as np
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logger import agent_logger


class EmotionInput(BaseModel):
    """Input schema for emotion detection tool."""
    audio_data: bytes = Field(description="Audio data in bytes for emotion detection")


class EmotionTool(BaseTool):
    """Tool for detecting emotions from speech audio."""
    
    name: str = "emotion_tool"
    description: str = "Detect emotions from speech audio using offline models"
    args_schema: type = EmotionInput
    
    def __init__(self):
        super().__init__()
        self._emotion_model = None
        self._initialize_model()
    
    @property
    def emotion_model(self):
        return self._emotion_model
    
    def _initialize_model(self):
        """Initialize the emotion detection model."""
        try:
            # Try to initialize SpeechBrain emotion recognition
            from speechbrain.pretrained import EncoderClassifier
            
            # Use a simple emotion classification model
            # In production, you might want to use a more sophisticated model
            self._emotion_model = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="models/emotion/speechbrain_emotion"
            )
            agent_logger.info("Speech emotion model initialized")
            
        except ImportError:
            agent_logger.warning("SpeechBrain not available, using fallback emotion detection")
            self._emotion_model = None
        except Exception as e:
            agent_logger.error(f"Failed to initialize emotion model: {e}")
            self._emotion_model = None
    
    def _run(self, audio_data: bytes) -> Dict[str, Any]:
        """Run emotion detection on audio data."""
        try:
            if not self.emotion_model:
                return self._fallback_emotion_detection(audio_data)
            
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Predict emotion
                out_prob, score, index, text_lab = self.emotion_model.classify_file(temp_file_path)
                
                # Get emotion label and confidence
                emotion = text_lab[0]
                confidence = float(out_prob.max())
                
                agent_logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.3f})")
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "probabilities": out_prob.tolist(),
                    "model": "speechbrain"
                }
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
                
        except Exception as e:
            agent_logger.error(f"Error in emotion detection: {e}")
            return self._fallback_emotion_detection(audio_data)
    
    async def _arun(self, audio_data: bytes) -> Dict[str, Any]:
        """Async run of emotion detection."""
        return self._run(audio_data)
    
    def _fallback_emotion_detection(self, audio_data: bytes) -> Dict[str, Any]:
        """Fallback emotion detection using audio features."""
        try:
            import librosa
            
            # Load audio data
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=None)
            
            # Extract basic audio features
            features = self._extract_audio_features(audio_array, sr)
            
            # Simple rule-based emotion detection
            emotion = self._classify_emotion_from_features(features)
            
            agent_logger.info(f"Fallback emotion detection: {emotion}")
            
            return {
                "emotion": emotion,
                "confidence": 0.6,  # Lower confidence for fallback
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
    
    def _extract_audio_features(self, audio_array: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features for emotion classification."""
        features = {}
        
        try:
            # Energy (RMS)
            features["energy"] = np.sqrt(np.mean(audio_array**2))
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
            features["spectral_centroid"] = np.mean(spectral_centroids)
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sr)[0]
            features["spectral_rolloff"] = np.mean(spectral_rolloff)
            
            # Zero crossing rate (noisiness)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]
            features["zero_crossing_rate"] = np.mean(zero_crossing_rate)
            
            # MFCC (mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs)
            features["mfcc_std"] = np.std(mfccs)
            
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 90)]
            features["pitch_mean"] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            
        except Exception as e:
            agent_logger.warning(f"Error extracting audio features: {e}")
            # Set default values
            features = {
                "energy": 0.0,
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0,
                "mfcc_mean": 0.0,
                "mfcc_std": 0.0,
                "pitch_mean": 0.0
            }
        
        return features
    
    def _classify_emotion_from_features(self, features: Dict[str, float]) -> str:
        """Classify emotion based on audio features."""
        # Simple rule-based classification
        energy = features.get("energy", 0.0)
        spectral_centroid = features.get("spectral_centroid", 0.0)
        zero_crossing_rate = features.get("zero_crossing_rate", 0.0)
        pitch_mean = features.get("pitch_mean", 0.0)
        
        # High energy + high pitch = happy/excited
        if energy > 0.1 and pitch_mean > 200:
            return "happy"
        
        # Low energy + low pitch = sad
        elif energy < 0.05 and pitch_mean < 150:
            return "sad"
        
        # High zero crossing rate = angry/anxious
        elif zero_crossing_rate > 0.1:
            return "angry"
        
        # High spectral centroid = anxious
        elif spectral_centroid > 2000:
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
            "model": "speechbrain" if self.emotion_model else "fallback"
        } 