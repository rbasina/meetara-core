"""
YAML Loading Utilities for Meetara Core.
Centralized YAML file loading with error handling.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from app.core.logger import agent_logger


def load_yaml_file(file_path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML file with error handling.
    
    Args:
        file_path: Path to YAML file
        default: Default value to return if file doesn't exist or fails to load
        
    Returns:
        Loaded YAML data as dictionary, or default if file doesn't exist/fails
    """
    if default is None:
        default = {}
    
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data is not None else default
        else:
            agent_logger.warning(f"YAML file not found: {file_path}")
            return default
    except Exception as e:
        agent_logger.error(f"Error loading YAML file {file_path}: {e}")
        return default


def safe_load_yaml(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load YAML file, returning None on error.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded YAML data or None if error
    """
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None
    except Exception as e:
        agent_logger.error(f"Error loading YAML file {file_path}: {e}")
        return None

