"""
Model Management API for Meetara Core.

Provides endpoints for:
- Listing available models
- Getting model information
- Switching active model
- Model health checks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.config_loader import config_loader
from app.core.gguf_llm_processor import get_meetara_gguf_processor
from app.core.logger import agent_logger

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelInfo(BaseModel):
    """Model information response."""
    id: str
    name: str
    description: str
    size: float
    parameters: str
    tier: str
    default: bool
    recommended_for: List[str]
    loaded: bool = False


class ModelListResponse(BaseModel):
    """Response for listing all models."""
    models: List[ModelInfo]
    current_model: str
    auto_select_enabled: bool


class ModelSwitchRequest(BaseModel):
    """Request to switch active model."""
    model_id: str


class ModelSwitchResponse(BaseModel):
    """Response after switching model."""
    success: bool
    message: str
    current_model: str


@router.get("/", response_model=ModelListResponse)
async def list_models():
    """Get list of all available models.
    
    Returns information about all configured models including:
    - Model ID and display name
    - Size and parameters
    - Recommended domains
    - Whether currently loaded
    """
    try:
        ui_models = config_loader.get_models_for_ui()
        processor = get_meetara_gguf_processor()
        
        # Check which models are currently loaded
        loaded_models = list(processor.models.keys()) if processor else []
        
        models = []
        for model in ui_models:
            # Determine if this model is loaded
            model_loaded = False
            if model['id'] == 'meetara-1.7b' and 'instruct' in loaded_models:
                model_loaded = True
            elif model['id'] == 'meetara-4b-thinking' and 'thinking' in loaded_models:
                model_loaded = True
            
            models.append(ModelInfo(
                id=model['id'],
                name=model['name'],
                description=model['description'],
                size=model['size'],
                parameters=model['parameters'],
                tier=model['tier'],
                default=model['default'],
                recommended_for=model['recommended_for'],
                loaded=model_loaded
            ))
        
        # Get current model setting
        current_model = config_loader.get_default_model()
        
        # Check if auto-select is enabled
        selection_strategy = config_loader.model_config.get('selection_strategy', {})
        auto_select = selection_strategy.get('auto_select', True)
        
        return ModelListResponse(
            models=models,
            current_model=current_model,
            auto_select_enabled=auto_select
        )
        
    except Exception as e:
        agent_logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier (e.g., 'meetara-1.7b')
    """
    try:
        model_config = config_loader.get_model_config(model_id)
        
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        processor = get_meetara_gguf_processor()
        loaded_models = list(processor.models.keys()) if processor else []
        
        # Check if loaded
        model_loaded = False
        if model_id == 'meetara-1.7b' and 'instruct' in loaded_models:
            model_loaded = True
        elif model_id == 'meetara-4b-thinking' and 'thinking' in loaded_models:
            model_loaded = True
        
        return ModelInfo(
            id=model_id,
            name=model_config.get('display_name', model_id),
            description=model_config.get('description', ''),
            size=model_config.get('size_gb', 0),
            parameters=model_config.get('parameters', ''),
            tier=model_config.get('tier', 'balanced'),
            default=model_config.get('default', False),
            recommended_for=model_config.get('recommended_for', []),
            loaded=model_loaded
        )
        
    except HTTPException:
        raise
    except Exception as e:
        agent_logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    """Switch the active model.
    
    This will set the default model for future requests.
    Note: The model will be loaded lazily on first use.
    
    Args:
        request: ModelSwitchRequest with model_id
    """
    try:
        model_id = request.model_id
        model_config = config_loader.get_model_config(model_id)
        
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # Update the runtime config (this doesn't persist to file)
        # The processor will use this on next request
        processor = get_meetara_gguf_processor()
        
        if not processor:
            raise HTTPException(status_code=503, detail="Model processor not available")
        
        # Get HF info for the model
        hf_info = config_loader.get_model_hf_info(model_id)
        
        if hf_info['repo_id'] and hf_info['filename']:
            # Update processor config to use this model
            processor.config.meetara_hf_model_id = hf_info['repo_id']
            processor.config.meetara_hf_model_file = hf_info['filename']
            
            agent_logger.info(f"Switched to model: {model_id} ({hf_info['repo_id']})")
            
            return ModelSwitchResponse(
                success=True,
                message=f"Switched to {model_config.get('display_name', model_id)}. Model will load on next query.",
                current_model=model_id
            )
        else:
            raise HTTPException(status_code=400, detail="Model configuration incomplete")
        
    except HTTPException:
        raise
    except Exception as e:
        agent_logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommended/{domain}")
async def get_recommended_model(domain: str):
    """Get the recommended model for a specific domain.
    
    Args:
        domain: Domain name (e.g., 'healthcare', 'programming')
        
    Returns:
        Recommended model ID and info
    """
    try:
        recommended_model_id = config_loader.get_model_for_domain(domain)
        model_config = config_loader.get_model_config(recommended_model_id)
        
        return {
            "domain": domain,
            "recommended_model": recommended_model_id,
            "model_name": model_config.get('display_name', recommended_model_id),
            "reason": f"Recommended for {model_config.get('tier', 'general')} tier domains"
        }
        
    except Exception as e:
        agent_logger.error(f"Error getting recommended model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def models_health():
    """Check health status of model system.
    
    Returns information about:
    - Models currently loaded in memory
    - Available model paths
    - Memory usage estimates
    """
    try:
        processor = get_meetara_gguf_processor()
        
        if not processor:
            return {
                "status": "not_initialized",
                "models_loaded": 0,
                "available_models": []
            }
        
        model_info = processor.get_model_info()
        health = processor.health_check()
        
        return {
            "status": health.get("status", "unknown"),
            "models_loaded": health.get("models_loaded", 0),
            "loaded_model_types": list(processor.models.keys()),
            "available_model_paths": model_info.get("available_models", []),
            "models_path_exists": health.get("models_path_exists", False),
            "meetara_models_enabled": model_info.get("meetara_models_enabled", False)
        }
        
    except Exception as e:
        agent_logger.error(f"Error checking model health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

