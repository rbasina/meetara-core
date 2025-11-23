"""
Emotion Detection API endpoints for Meetara Core.

This module provides endpoints for detecting emotions from facial expressions
and speech audio using offline models.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from app.core.logger import api_logger
from app.core.security import privacy_enforcer
from app.agent.tools import EmotionTool, FaceEmotionTool


router = APIRouter(prefix="/emotion", tags=["emotion"])


class EmotionResponse(BaseModel):
    """Response model for emotion detection."""
    emotion: str
    confidence: float
    model: str
    additional_info: Optional[Dict[str, Any]] = None


@router.post("/detect-face-emotion", response_model=EmotionResponse)
async def detect_face_emotion(
    image: UploadFile = File(..., description="Image file for facial emotion detection")
):
    """Detect emotions from facial expressions in an image."""
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image data
        image_data = await image.read()
        
        # Validate file size
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        # Initialize face emotion tool
        face_emotion_tool = FaceEmotionTool()
        
        # Detect emotion
        result = await face_emotion_tool.ainvoke({"image_data": image_data})
        
        # Log detection (without sensitive data)
        privacy_enforcer.safe_log(
            "Face emotion detection completed",
            {
                "emotion": result.get("emotion"),
                "confidence": result.get("confidence"),
                "model": result.get("model"),
                "faces_detected": result.get("faces_detected", 0)
            }
        )
        
        api_logger.info(f"Face emotion detected: {result.get('emotion')} (confidence: {result.get('confidence', 0):.3f})")
        
        return EmotionResponse(
            emotion=result.get("emotion", "neutral"),
            confidence=result.get("confidence", 0.0),
            model=result.get("model", "unknown"),
            additional_info={
                "faces_detected": result.get("faces_detected", 0),
                "error": result.get("error")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error in face emotion detection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect facial emotions. Please try again."
        )


@router.post("/detect-speech-emotion", response_model=EmotionResponse)
async def detect_speech_emotion(
    audio: UploadFile = File(..., description="Audio file for speech emotion detection")
):
    """Detect emotions from speech audio."""
    try:
        # Validate file type
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file (WAV, MP3, etc.)"
            )
        
        # Read audio data
        audio_data = await audio.read()
        
        # Validate file size
        if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit for audio
            raise HTTPException(
                status_code=400,
                detail="Audio file too large. Maximum size is 50MB."
            )
        
        # Initialize speech emotion tool
        emotion_tool = EmotionTool()
        
        # Detect emotion
        result = await emotion_tool.ainvoke({"audio_data": audio_data})
        
        # Log detection (without sensitive data)
        privacy_enforcer.safe_log(
            "Speech emotion detection completed",
            {
                "emotion": result.get("emotion"),
                "confidence": result.get("confidence"),
                "model": result.get("model")
            }
        )
        
        api_logger.info(f"Speech emotion detected: {result.get('emotion')} (confidence: {result.get('confidence', 0):.3f})")
        
        return EmotionResponse(
            emotion=result.get("emotion", "neutral"),
            confidence=result.get("confidence", 0.0),
            model=result.get("model", "unknown"),
            additional_info={
                "segments": result.get("segments", 0),
                "error": result.get("error")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error in speech emotion detection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect speech emotions. Please try again."
        )


@router.post("/detect-multi-emotion")
async def detect_multi_emotion(
    image: Optional[UploadFile] = File(None, description="Image file for facial emotion"),
    audio: Optional[UploadFile] = File(None, description="Audio file for speech emotion")
):
    """Detect emotions from multiple sources (facial and speech)."""
    try:
        results = {
            "face_emotion": None,
            "speech_emotion": None,
            "combined_emotion": None,
            "confidence": 0.0
        }
        
        # Detect facial emotion if image provided
        if image:
            face_result = await detect_face_emotion(image)
            results["face_emotion"] = {
                "emotion": face_result.emotion,
                "confidence": face_result.confidence,
                "model": face_result.model
            }
        
        # Detect speech emotion if audio provided
        if audio:
            speech_result = await detect_speech_emotion(audio)
            results["speech_emotion"] = {
                "emotion": speech_result.emotion,
                "confidence": speech_result.confidence,
                "model": speech_result.model
            }
        
        # Combine emotions if both sources available
        if results["face_emotion"] and results["speech_emotion"]:
            emotions = [
                (results["face_emotion"]["emotion"], results["face_emotion"]["confidence"]),
                (results["speech_emotion"]["emotion"], results["speech_emotion"]["confidence"])
            ]
            
            # Simple voting mechanism
            emotion_counts = {}
            for emotion, confidence in emotions:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += confidence
                else:
                    emotion_counts[emotion] = confidence
            
            # Get most confident emotion
            combined_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            results["combined_emotion"] = combined_emotion[0]
            results["confidence"] = combined_emotion[1] / 2.0  # Average confidence
        
        elif results["face_emotion"]:
            results["combined_emotion"] = results["face_emotion"]["emotion"]
            results["confidence"] = results["face_emotion"]["confidence"]
        
        elif results["speech_emotion"]:
            results["combined_emotion"] = results["speech_emotion"]["emotion"]
            results["confidence"] = results["speech_emotion"]["confidence"]
        
        else:
            results["combined_emotion"] = "neutral"
            results["confidence"] = 0.0
        
        api_logger.info(f"Multi-emotion detection completed: {results['combined_emotion']}")
        
        return results
        
    except Exception as e:
        api_logger.error(f"Error in multi-emotion detection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect emotions from multiple sources."
        )


@router.get("/supported-emotions")
async def get_supported_emotions():
    """Get information about supported emotions and detection models."""
    try:
        face_tool = FaceEmotionTool()
        speech_tool = EmotionTool()
        
        face_emotions = face_tool.get_supported_emotions()
        speech_emotions = speech_tool.get_supported_emotions()
        
        return {
            "facial_emotions": face_emotions,
            "speech_emotions": speech_emotions,
            "supported_models": {
                "facial": ["deepface", "fer", "fallback"],
                "speech": ["speechbrain", "fallback"]
            }
        }
        
    except Exception as e:
        api_logger.error(f"Error getting supported emotions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve supported emotions information"
        ) 