"""
Security System for Resume Parser LLM
Handles API key management, input validation, and security best practices.
"""

import os
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime, timedelta
import json
import base64

from core.exceptions import ValidationError, ConfigurationError
from core.logging_system import get_logger

logger = get_logger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = None
    rate_limit_requests_per_minute: int = 100
    session_timeout_minutes: int = 60
    require_api_key: bool = False
    allowed_hosts: List[str] = None
    enable_cors: bool = True
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.pdf', '.txt', '.doc', '.docx']
        if self.allowed_hosts is None:
            self.allowed_hosts = ['localhost', '127.0.0.1']

class APIKeyManager:
    """Secure API key management system."""
    
    def __init__(self, key_file_path: str = ".keys/api_keys.json"):
        self.key_file_path = Path(key_file_path)
        self.key_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from secure file."""
        if self.key_file_path.exists():
            try:
                with open(self.key_file_path, 'r') as f:
                    encrypted_data = json.load(f)
                    # In production, implement proper encryption
                    self._keys = encrypted_data
                    logger.info(f"Loaded {len(self._keys)} API key configurations")
            except Exception as e:
                logger.error(f"Failed to load API keys: {str(e)}")
                self._keys = {}
        else:
            logger.info("No API key file found, starting with empty keystore")
    
    def _save_keys(self):
        """Save API keys to secure file."""
        try:
            # In production, implement proper encryption
            with open(self.key_file_path, 'w') as f:
                json.dump(self._keys, f, indent=2)
            
            # Set restrictive permissions (Unix-like systems)
            if os.name != 'nt':  # Not Windows
                os.chmod(self.key_file_path, 0o600)
            
            logger.info(f"Saved {len(self._keys)} API key configurations")
        except Exception as e:
            logger.error(f"Failed to save API keys: {str(e)}")
            raise ConfigurationError("api_keys", f"Cannot save API keys: {str(e)}")
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        if permissions is None:
            permissions = ["read", "parse"]
        
        # Generate secure random key
        key = f"rpl_{secrets.token_urlsafe(32)}"
        
        # Store key metadata
        self._keys[key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        self._save_keys()
        logger.info(f"Generated new API key for '{name}' with permissions: {permissions}")
        
        return key
    
    def validate_api_key(self, key: str, required_permission: str = None) -> bool:
        """Validate an API key and optional permission."""
        if not key or key not in self._keys:
            return False
        
        key_data = self._keys[key]
        
        # Check if key is active
        if not key_data.get("active", False):
            return False
        
        # Check permission if required
        if required_permission and required_permission not in key_data.get("permissions", []):
            return False
        
        # Update usage statistics
        key_data["last_used"] = datetime.now().isoformat()
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
        self._save_keys()
        
        return True
    
    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._keys:
            self._keys[key]["active"] = False
            self._keys[key]["revoked_at"] = datetime.now().isoformat()
            self._save_keys()
            logger.warning(f"Revoked API key: {key[:8]}...")
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        return [
            {
                "name": data["name"],
                "permissions": data["permissions"],
                "created_at": data["created_at"],
                "last_used": data.get("last_used"),
                "usage_count": data.get("usage_count", 0),
                "active": data.get("active", False)
            }
            for data in self._keys.values()
        ]

class InputValidator:
    """Comprehensive input validation system."""
    
    # Common patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$')
    URL_PATTERN = re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    
    # Dangerous patterns to detect
    SCRIPT_PATTERNS = [
        re.compile(r'<script[^>]*>', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'document\.(write|cookie)', re.IGNORECASE)
    ]
    
    SQL_INJECTION_PATTERNS = [
        re.compile(r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)', re.IGNORECASE),
        re.compile(r'(\bOR\b|\bAND\b)\s+[\'"]\d+[\'"]?\s*=\s*[\'"]\d+[\'"]?', re.IGNORECASE),
        re.compile(r'[\'"];?\s*(--|/\*)', re.IGNORECASE)
    ]
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 254:
            return False
        return cls.EMAIL_PATTERN.match(email) is not None
    
    @classmethod
    def validate_phone(cls, phone: str) -> bool:
        """Validate phone number format."""
        if not phone or len(phone) > 20:
            return False
        return cls.PHONE_PATTERN.match(phone.strip()) is not None
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL format."""
        if not url or len(url) > 2048:
            return False
        return cls.URL_PATTERN.match(url) is not None
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 10000) -> str:
        """Sanitize text input by removing dangerous patterns."""
        if not text:
            return ""
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Check for dangerous patterns
        for pattern in cls.SCRIPT_PATTERNS + cls.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Potential security threat detected in input: {pattern.pattern}")
                # In production, you might want to reject the input entirely
                text = pattern.sub('', text)
        
        return text.strip()
    
    @classmethod
    def validate_file_upload(cls, file_data: bytes, filename: str, security_config: SecurityConfig) -> bool:
        """Validate uploaded file for security."""
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > security_config.max_file_size_mb:
            raise ValidationError(
                "file_size", 
                f"{file_size_mb:.2f}MB", 
                f"<= {security_config.max_file_size_mb}MB"
            )
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in security_config.allowed_file_types:
            raise ValidationError(
                "file_type", 
                file_ext, 
                f"one of {security_config.allowed_file_types}"
            )
        
        # Check for executable signatures
        dangerous_signatures = [
            b'\x4d\x5a',  # PE/EXE files
            b'\x7f\x45\x4c\x46',  # ELF files  
            b'\xfe\xed\xfa',  # Mach-O files
            b'#!/bin/',  # Shell scripts
            b'<script',  # HTML/JS
        ]
        
        for sig in dangerous_signatures:
            if file_data.startswith(sig):
                raise ValidationError(
                    "file_content", 
                    "executable_detected", 
                    "non-executable file"
                )
        
        return True
    
    @classmethod
    def validate_json_input(cls, json_data: Any, schema: Dict[str, Any] = None) -> bool:
        """Validate JSON input against schema."""
        if not isinstance(json_data, dict):
            raise ValidationError("json_input", type(json_data).__name__, "dictionary")
        
        # Basic size check
        json_str = json.dumps(json_data)
        if len(json_str) > 100000:  # 100KB limit
            raise ValidationError("json_size", f"{len(json_str)} chars", "<= 100KB")
        
        # Check for deeply nested objects (prevent DoS)
        def check_depth(obj, current_depth=0, max_depth=10):
            if current_depth > max_depth:
                raise ValidationError("json_depth", f"{current_depth}", f"<= {max_depth}")
            
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, current_depth + 1, max_depth)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1, max_depth)
        
        check_depth(json_data)
        
        # TODO: Add schema validation if provided
        return True

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > cutoff
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        active_identifiers = 0
        total_requests = 0
        
        for identifier, requests in self.requests.items():
            recent_requests = [r for r in requests if r > cutoff]
            if recent_requests:
                active_identifiers += 1
                total_requests += len(recent_requests)
        
        return {
            "active_identifiers": active_identifiers,
            "total_recent_requests": total_requests,
            "rate_limit": self.requests_per_minute
        }

class SecurityManager:
    """Main security manager coordinating all security features."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests_per_minute)
        self.validator = InputValidator()
        
        logger.info("Security manager initialized")
    
    def check_request_security(
        self, 
        client_ip: str,
        api_key: str = None,
        required_permission: str = None
    ) -> Dict[str, Any]:
        """Comprehensive security check for incoming requests."""
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later."
            }
        
        # API key validation (if required)
        if self.config.require_api_key:
            if not api_key:
                return {
                    "allowed": False,
                    "reason": "missing_api_key",
                    "message": "API key required"
                }
            
            if not self.api_key_manager.validate_api_key(api_key, required_permission):
                logger.warning(f"Invalid API key attempt from IP: {client_ip}")
                return {
                    "allowed": False,
                    "reason": "invalid_api_key",
                    "message": "Invalid or expired API key"
                }
        
        # Host validation
        if self.config.allowed_hosts and client_ip not in self.config.allowed_hosts:
            logger.warning(f"Request from unauthorized host: {client_ip}")
            return {
                "allowed": False,
                "reason": "unauthorized_host",
                "message": "Request not allowed from this host"
            }
        
        return {"allowed": True}
    
    def secure_file_processing(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Securely process uploaded files."""
        try:
            # Validate file
            self.validator.validate_file_upload(file_data, filename, self.config)
            
            # Generate secure filename
            safe_filename = self._generate_safe_filename(filename)
            
            # Calculate file hash for integrity
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            return {
                "valid": True,
                "safe_filename": safe_filename,
                "file_hash": file_hash,
                "file_size": len(file_data)
            }
            
        except ValidationError as e:
            logger.warning(f"File validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "error_type": "validation_error"
            }
    
    def _generate_safe_filename(self, original_filename: str) -> str:
        """Generate a safe filename."""
        # Remove directory traversal attempts
        filename = os.path.basename(original_filename)
        
        # Keep only alphanumeric, dots, hyphens, underscores
        safe_chars = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_chars)
        
        return f"{name}_{timestamp}{ext}"
    
    def sanitize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output data to prevent information disclosure."""
        sensitive_keys = [
            'api_key', 'password', 'token', 'secret', 'private_key',
            'auth', 'credential', 'session_id'
        ]
        
        def clean_dict(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if any(sensitive_word in key.lower() for sensitive_word in sensitive_keys):
                        cleaned[key] = "[REDACTED]"
                    else:
                        cleaned[key] = clean_dict(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_dict(item) for item in obj]
            elif isinstance(obj, str) and len(obj) > 50:
                # Truncate very long strings that might contain sensitive data
                return obj[:47] + "..."
            else:
                return obj
        
        return clean_dict(data)
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key from environment variables or secure storage.
        
        Args:
            key_name: Name of the API key (e.g., 'PINECONE_API_KEY', 'OPENAI_API_KEY')
            
        Returns:
            API key value or None if not found
        """
        # First try environment variable
        env_key = os.getenv(key_name)
        if env_key:
            return env_key
        
        # Try alternative naming (e.g., GOOGLE_API_KEY for GEMINI_API_KEY)
        if key_name == "GEMINI_API_KEY":
            env_key = os.getenv("GOOGLE_API_KEY")
            if env_key:
                return env_key
        
        # Could also check secure storage in the future
        return None
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security status report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "api_keys": {
                "total_keys": len(self.api_key_manager.list_keys()),
                "active_keys": len([k for k in self.api_key_manager.list_keys() if k["active"]])
            },
            "configuration": {
                "require_api_key": self.config.require_api_key,
                "rate_limit_per_minute": self.config.rate_limit_requests_per_minute,
                "max_file_size_mb": self.config.max_file_size_mb,
                "allowed_file_types": self.config.allowed_file_types
            }
        }

# Global security manager instance
_security_manager: Optional[SecurityManager] = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def set_security_manager(manager: SecurityManager):
    """Set the global security manager instance."""
    global _security_manager
    _security_manager = manager

# Utility functions
def secure_string_comparison(a: str, b: str) -> bool:
    """Secure string comparison to prevent timing attacks."""
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    
    return result == 0

def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)

def mask_sensitive_data(text: str) -> str:
    """Mask sensitive data in text for logging."""
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.***', text)
    
    # Phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '***-***-****', text)
    
    # Credit card-like numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '****-****-****-****', text)
    
    return text