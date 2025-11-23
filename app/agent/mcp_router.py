"""
MCP (Multi-Component Planner) Router for Meetara Core.

This module orchestrates the flow between RAG retrieval, emotion detection,
and response generation to provide contextually appropriate responses.
"""
from typing import Dict, List, Any, Optional
from langchain_core.documents import Document
from app.core.logger import agent_logger
from app.rag.domain_retrievers import get_domain_retriever
from app.agent.tools import EmotionTool, FaceEmotionTool


class MCPRouter:
    """Multi-Component Planner Router for orchestrating AI responses."""
    
    def __init__(self):
        self.emotion_tool = EmotionTool()
        self.face_emotion_tool = FaceEmotionTool()
        agent_logger.info("MCPRouter initialized")
    
    async def process_with_emotion_context(
        self,
        query: str,
        domain: str,
        detected_emotion: Optional[str] = None,
        face_emotion: Optional[str] = None,
        speech_emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process query with emotion-aware context and RAG retrieval."""
        try:
            # Get domain-specific context
            domain_context = await self._get_domain_context(query, domain)
            
            # Determine emotional context
            emotion_context = await self._analyze_emotion_context(
                detected_emotion, face_emotion, speech_emotion
            )
            
            # Combine contexts for response generation
            combined_context = {
                "query": query,
                "domain": domain,
                "domain_context": domain_context,
                "emotion_context": emotion_context,
                "response_style": self._determine_response_style(emotion_context)
            }
            
            agent_logger.info(f"Processed query with emotion context for domain: {domain}")
            return combined_context
            
        except Exception as e:
            agent_logger.error(f"Error in MCP routing: {e}")
            return {
                "query": query,
                "domain": domain,
                "error": str(e),
                "response_style": "neutral"
            }
    
    async def _get_domain_context(self, query: str, domain: str) -> List[Document]:
        """Retrieve relevant context from the domain's vector store."""
        try:
            retriever = get_domain_retriever(domain)
            context_docs = retriever.similarity_search(query, k=3)
            
            agent_logger.info(f"Retrieved {len(context_docs)} context documents for domain: {domain}")
            return context_docs
            
        except Exception as e:
            agent_logger.warning(f"Failed to get domain context for {domain}: {e}")
            return []
    
    async def _analyze_emotion_context(
        self,
        detected_emotion: Optional[str] = None,
        face_emotion: Optional[str] = None,
        speech_emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze and combine emotion detection results."""
        emotions = []
        
        if detected_emotion:
            emotions.append(("detected", detected_emotion))
        if face_emotion:
            emotions.append(("face", face_emotion))
        if speech_emotion:
            emotions.append(("speech", speech_emotion))
        
        # Determine primary emotion
        primary_emotion = self._determine_primary_emotion(emotions)
        
        # Get emotion-specific response guidelines
        response_guidelines = self._get_emotion_guidelines(primary_emotion)
        
        return {
            "primary_emotion": primary_emotion,
            "emotion_sources": emotions,
            "response_guidelines": response_guidelines,
            "confidence": len(emotions) / 3.0  # Simple confidence based on sources
        }
    
    def _determine_primary_emotion(self, emotions: List[tuple]) -> str:
        """Determine the primary emotion from multiple sources."""
        if not emotions:
            return "neutral"
        
        # Simple voting mechanism
        emotion_counts = {}
        for source, emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most common emotion
        if emotion_counts:
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        return "neutral"
    
    def _get_emotion_guidelines(self, emotion: str) -> Dict[str, str]:
        """Get response guidelines based on detected emotion."""
        guidelines = {
            "happy": {
                "tone": "enthusiastic and positive",
                "approach": "build on the positive energy",
                "style": "encouraging and celebratory"
            },
            "sad": {
                "tone": "gentle and supportive",
                "approach": "offer comfort and understanding",
                "style": "empathetic and caring"
            },
            "angry": {
                "tone": "calm and measured",
                "approach": "de-escalate and provide solutions",
                "style": "patient and problem-solving"
            },
            "anxious": {
                "tone": "reassuring and calm",
                "approach": "provide clarity and structure",
                "style": "supportive and grounding"
            },
            "neutral": {
                "tone": "professional and helpful",
                "approach": "provide clear information",
                "style": "informative and direct"
            }
        }
        
        return guidelines.get(emotion, guidelines["neutral"])
    
    def _determine_response_style(self, emotion_context: Dict[str, Any]) -> str:
        """Determine the appropriate response style based on emotion."""
        primary_emotion = emotion_context.get("primary_emotion", "neutral")
        confidence = emotion_context.get("confidence", 0.0)
        
        # High confidence in emotion detection
        if confidence > 0.6:
            return f"emotion_aware_{primary_emotion}"
        
        # Medium confidence
        elif confidence > 0.3:
            return f"cautious_{primary_emotion}"
        
        # Low confidence or no emotion detected
        else:
            return "neutral"
    
    async def detect_emotions_from_multiple_sources(
        self,
        audio_data: Optional[bytes] = None,
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Detect emotions from multiple sources (speech and facial)."""
        results = {
            "speech_emotion": None,
            "face_emotion": None,
            "combined_emotion": None,
            "confidence": 0.0
        }
        
        try:
            # Detect speech emotion
            if audio_data:
                speech_result = await self.emotion_tool.ainvoke({
                    "audio_data": audio_data
                })
                results["speech_emotion"] = speech_result.get("emotion")
            
            # Detect facial emotion
            if image_data:
                face_result = await self.face_emotion_tool.ainvoke({
                    "image_data": image_data
                })
                results["face_emotion"] = face_result.get("emotion")
            
            # Combine emotions
            emotions = []
            if results["speech_emotion"]:
                emotions.append(results["speech_emotion"])
            if results["face_emotion"]:
                emotions.append(results["face_emotion"])
            
            if emotions:
                results["combined_emotion"] = self._determine_primary_emotion([
                    ("source", emotion) for emotion in emotions
                ])
                results["confidence"] = len(emotions) / 2.0
            
            agent_logger.info(f"Detected emotions: {results}")
            return results
            
        except Exception as e:
            agent_logger.error(f"Error detecting emotions: {e}")
            return results
    
    def get_routing_info(self) -> Dict[str, Any]:
        """Get information about the MCP routing capabilities."""
        return {
            "capabilities": [
                "domain_specific_retrieval",
                "emotion_aware_responses",
                "multi_source_emotion_detection",
                "contextual_response_adaptation"
            ],
            "supported_emotions": [
                "happy", "sad", "angry", "anxious", "neutral"
            ],
            "response_styles": [
                "emotion_aware_happy",
                "emotion_aware_sad", 
                "emotion_aware_angry",
                "emotion_aware_anxious",
                "cautious_*",
                "neutral"
            ]
        }


def create_mcp_router() -> MCPRouter:
    """Factory function to create an MCPRouter instance."""
    return MCPRouter()


# Global MCP router instance
_mcp_router: Optional[MCPRouter] = None


def get_mcp_router() -> MCPRouter:
    """Get the global MCPRouter instance."""
    global _mcp_router
    if _mcp_router is None:
        _mcp_router = create_mcp_router()
    return _mcp_router 