"""
Centralized logging configuration for Meetara Core.
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from app.core.config import settings


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """Configure centralized logging for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    # Error handler for uncaught exceptions
    logger.add(
        sys.stderr,
        format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="ERROR",
        colorize=True
    )


# Initialize logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file
)


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)


# Module-specific loggers
api_logger = get_logger("api")
agent_logger = get_logger("agent")
rag_logger = get_logger("rag")
emotion_logger = get_logger("emotion")
security_logger = get_logger("security")


def is_verbose() -> bool:
    """Check if verbose logging is enabled (for detailed debug output)."""
    return settings.verbose_logging