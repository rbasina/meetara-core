"""
Security and privacy enforcement for Meetara Core.
"""
import re
import hashlib
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import aiofiles
from pydantic import BaseModel, ValidationError
from app.core.logger import security_logger


class SecurityValidator:
    """Input validation and sanitization utilities."""
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'on\w+\s*=',
        r'<iframe.*?>',
        r'<object.*?>',
        r'<embed.*?>',
        r'<link.*?>',
        r'<meta.*?>',
        r'<form.*?>',
        r'<input.*?>',
        r'<textarea.*?>',
        r'<select.*?>',
        r'<button.*?>',
        r'<a.*?href\s*=\s*["\']javascript:',
        r'<a.*?href\s*=\s*["\']data:',
        r'<a.*?href\s*=\s*["\']vbscript:',
    ]
    
    # File type validation
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.md', '.json', '.csv'}
    MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB (for large textbook PDFs)
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Sanitize text input to prevent XSS and injection attacks."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Limit length
        if len(text) > 10000:  # 10KB limit
            text = text[:10000]
        
        return text.strip()
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate and sanitize filename."""
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove path traversal attempts
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed."""
        ext = Path(filename).suffix.lower()
        return ext in cls.ALLOWED_EXTENSIONS
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def sanitize_domain_name(cls, domain: str) -> str:
        """Sanitize domain name for vector store paths."""
        if not domain:
            raise ValueError("Domain name cannot be empty")
        
        # Only allow alphanumeric, hyphens, and underscores
        domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
        
        # Limit length
        if len(domain) > 50:
            domain = domain[:50]
        
        return domain.lower()
    
    @classmethod
    def hash_sensitive_data(cls, data: str) -> str:
        """Create a hash of sensitive data for logging."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class PrivacyEnforcer:
    """Privacy protection utilities."""
    
    # PII patterns to detect and redact
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    @classmethod
    def redact_pii(cls, text: str) -> str:
        """Redact PII from text for logging."""
        redacted = text
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            redacted = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', redacted)
        
        return redacted
    
    @classmethod
    def safe_log(cls, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Safely log messages without exposing PII."""
        if data:
            # Redact any PII from data
            safe_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    safe_data[key] = cls.redact_pii(value)
                else:
                    safe_data[key] = value
            security_logger.info(f"{message} | Data: {safe_data}")
        else:
            security_logger.info(message)


class InputValidator:
    """Pydantic-based input validation."""
    
    class ChatRequest(BaseModel):
        query: str
        topic: Optional[str] = None
        lang: Optional[str] = None
        emotion: Optional[str] = None
        
        class Config:
            extra = "forbid"  # Reject unknown fields
    
    class UploadRequest(BaseModel):
        domain: str
        filename: str
        
        class Config:
            extra = "forbid"
    
    class EmotionRequest(BaseModel):
        domain: Optional[str] = None
        
        class Config:
            extra = "forbid"
    
    @classmethod
    def validate_chat_request(cls, data: Dict[str, Any]) -> ChatRequest:
        """Validate chat request data."""
        try:
            # Sanitize text inputs
            if 'query' in data:
                data['query'] = SecurityValidator.sanitize_text(data['query'])
            if 'topic' in data and data['topic']:
                data['topic'] = SecurityValidator.sanitize_text(data['topic'])
            
            return cls.ChatRequest(**data)
        except ValidationError as e:
            security_logger.warning(f"Invalid chat request: {e}")
            raise ValueError(f"Invalid request data: {e}")
    
    @classmethod
    def validate_upload_request(cls, data: Dict[str, Any]) -> UploadRequest:
        """Validate upload request data."""
        try:
            # Sanitize domain name
            if 'domain' in data:
                data['domain'] = SecurityValidator.sanitize_domain_name(data['domain'])
            
            return cls.UploadRequest(**data)
        except ValidationError as e:
            security_logger.warning(f"Invalid upload request: {e}")
            raise ValueError(f"Invalid request data: {e}")


# Global instances
security_validator = SecurityValidator()
privacy_enforcer = PrivacyEnforcer()
input_validator = InputValidator() 