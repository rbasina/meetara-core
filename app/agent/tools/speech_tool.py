"""
Speech Tool for Meetara Core.

This tool provides speech-to-text (STT) and text-to-speech (TTS) capabilities
using offline models like faster-whisper and edge-tts.
"""
import io
import tempfile
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logger import agent_logger


class SpeechToTextInput(BaseModel):
    """Input schema for speech-to-text tool."""
    audio_data: bytes = Field(description="Audio data in bytes")
    language: str = Field(default="en", description="Language code for transcription")


class TextToSpeechInput(BaseModel):
    """Input schema for text-to-speech tool."""
    text: str = Field(description="Text to convert to speech")
    voice: str = Field(default="en-US-JennyNeural", description="Voice to use for synthesis")
    speed: float = Field(default=1.0, description="Speech speed (0.5 to 2.0)")


class SpeechTool(BaseTool):
    """Tool for speech-to-text and text-to-speech processing."""
    
    name: str = "speech_tool"
    description: str = "Convert speech to text or text to speech using offline models"
    args_schema: type = SpeechToTextInput
    
    def __init__(self):
        super().__init__()
        self._stt_model = None
        self._tts_voice = settings.tts_voice
        self._initialize_models()
    
    @property
    def stt_model(self):
        return self._stt_model
    
    @property
    def tts_voice(self):
        return self._tts_voice
    
    def _initialize_models(self):
        """Initialize speech processing models."""
        try:
            # Initialize STT model (faster-whisper)
            from faster_whisper import WhisperModel
            self._stt_model = WhisperModel(
                settings.stt_model,
                device="cpu",
                compute_type="int8"
            )
            agent_logger.info("Speech-to-text model initialized")
            
        except ImportError:
            agent_logger.warning("faster-whisper not available, STT will be limited")
            self._stt_model = None
        except Exception as e:
            agent_logger.error(f"Failed to initialize STT model: {e}")
            self._stt_model = None
    
    def _run(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """Run speech-to-text processing."""
        try:
            if not self.stt_model:
                return {
                    "transcript": "",
                    "language": language,
                    "error": "Speech-to-text model not available",
                    "confidence": 0.0
                }
            
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Transcribe audio
                segments, info = self.stt_model.transcribe(
                    temp_file_path,
                    language=language,
                    beam_size=5
                )
                
                # Collect transcript
                transcript = ""
                confidence_scores = []
                
                for segment in segments:
                    transcript += segment.text + " "
                    confidence_scores.append(segment.avg_logprob)
                
                # Calculate average confidence
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
                agent_logger.info(f"Transcribed audio: {transcript[:100]}...")
                
                return {
                    "transcript": transcript.strip(),
                    "language": info.language,
                    "confidence": avg_confidence,
                    "segments": len(confidence_scores)
                }
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
                
        except Exception as e:
            agent_logger.error(f"Error in speech-to-text: {e}")
            return {
                "transcript": "",
                "language": language,
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _arun(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """Async run of speech-to-text processing."""
        return self._run(audio_data, language)
    
    async def text_to_speech(self, text: str, voice: str = None, speed: float = 1.0) -> Dict[str, Any]:
        """Convert text to speech."""
        try:
            if not voice:
                voice = self._tts_voice
            
            # Use edge-tts for TTS
            import edge_tts
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            try:
                # Generate speech
                communicate = edge_tts.Communicate(text, voice, rate=f"{speed:+.1f}%")
                await communicate.save(temp_file_path)
                
                # Read the generated audio
                with open(temp_file_path, "rb") as f:
                    audio_data = f.read()
                
                agent_logger.info(f"Generated speech for text: {text[:50]}...")
                
                return {
                    "audio_data": audio_data,
                    "voice": voice,
                    "speed": speed,
                    "text_length": len(text),
                    "audio_format": "mp3"
                }
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
                
        except ImportError:
            agent_logger.warning("edge-tts not available, TTS will be limited")
            return {
                "audio_data": None,
                "voice": voice,
                "error": "Text-to-speech not available",
                "text": text
            }
        except Exception as e:
            agent_logger.error(f"Error in text-to-speech: {e}")
            return {
                "audio_data": None,
                "voice": voice,
                "error": str(e),
                "text": text
            }
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available TTS voices."""
        try:
            import edge_tts
            
            # This would require async context, so we'll return a subset
            common_voices = {
                "en-US-JennyNeural": "Jenny (Female, US English)",
                "en-US-GuyNeural": "Guy (Male, US English)",
                "en-GB-SoniaNeural": "Sonia (Female, British English)",
                "en-GB-RyanNeural": "Ryan (Male, British English)",
                "es-ES-ElviraNeural": "Elvira (Female, Spanish)",
                "fr-FR-DeniseNeural": "Denise (Female, French)",
                "de-DE-KatjaNeural": "Katja (Female, German)",
                "it-IT-IsabellaNeural": "Isabella (Female, Italian)",
                "pt-BR-FranciscaNeural": "Francisca (Female, Portuguese)",
                "ru-RU-SvetlanaNeural": "Svetlana (Female, Russian)"
            }
            
            return {
                "voices": common_voices,
                "default_voice": self.tts_voice
            }
            
        except ImportError:
            return {
                "voices": {},
                "error": "edge-tts not available"
            }
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get supported languages for STT."""
        return {
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi", "te", "ta", "bn"],
            "default_language": "en"
        } 