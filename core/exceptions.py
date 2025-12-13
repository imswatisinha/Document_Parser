"""
Comprehensive Error Handling System
Provides consistent error handling and user feedback across the application.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import logging
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization."""
    CONNECTION = "connection"
    PARSING = "parsing"
    VALIDATION = "validation"
    MODEL = "model"
    CONFIGURATION = "configuration"
    FILE_SYSTEM = "file_system"
    EXTERNAL_API = "external_api"
    USER_INPUT = "user_input"
    SYSTEM = "system"

@dataclass
class ErrorDetails:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    traceback_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error details to dictionary."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "technical_message": self.technical_message,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_info
        }

# Custom Exception Classes
class ResumeParserException(Exception):
    """Base exception for all resume parser errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.metadata = metadata or {}

class ConnectionError(ResumeParserException):
    """Raised when connection to external services fails."""
    
    def __init__(self, service: str, url: str, **kwargs):
        message = f"Failed to connect to {service} at {url}"
        super().__init__(
            message, 
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                f"Check if {service} is running",
                f"Verify the URL: {url}",
                "Check your network connection",
                "Ensure firewall allows the connection"
            ],
            metadata={"service": service, "url": url},
            **kwargs
        )

class ModelNotAvailableError(ResumeParserException):
    """Raised when requested model is not available."""
    
    def __init__(self, model_name: str, available_models: List[str] = None, **kwargs):
        message = f"Model '{model_name}' is not available"
        suggestions = ["Install the required model", "Check model name spelling"]
        
        if available_models:
            suggestions.append(f"Available models: {', '.join(available_models)}")
        
        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions,
            metadata={"requested_model": model_name, "available_models": available_models or []},
            **kwargs
        )

class ParseError(ResumeParserException):
    """Raised when document parsing fails."""
    
    def __init__(self, file_type: str, reason: str = None, **kwargs):
        message = f"Failed to parse {file_type} document"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Ensure the file is not corrupted",
                "Check if the file is password protected",
                "Try using a different file format",
                "Verify the file contains readable text"
            ],
            metadata={"file_type": file_type, "reason": reason},
            **kwargs
        )

class ValidationError(ResumeParserException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, expected: str = None, **kwargs):
        message = f"Invalid value for {field}: {value}"
        if expected:
            message += f" (expected: {expected})"
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            suggestions=[
                f"Check the format of {field}",
                f"Ensure {field} meets the required constraints"
            ],
            metadata={"field": field, "value": str(value), "expected": expected},
            **kwargs
        )

class ConfigurationError(ResumeParserException):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, issue: str, **kwargs):
        message = f"Configuration error for '{config_key}': {issue}"
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check your configuration file",
                "Verify environment variables",
                "Review configuration documentation"
            ],
            metadata={"config_key": config_key, "issue": issue},
            **kwargs
        )

class ExternalAPIError(ResumeParserException):
    """Raised when external API calls fail."""
    
    def __init__(self, api_name: str, status_code: int = None, response: str = None, **kwargs):
        message = f"{api_name} API error"
        if status_code:
            message += f" (Status: {status_code})"
        
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Check API endpoint availability",
                "Verify API credentials",
                "Review API rate limits",
                "Check network connectivity"
            ],
            metadata={"api_name": api_name, "status_code": status_code, "response": response},
            **kwargs
        )

class FileSystemError(ResumeParserException):
    """Raised when file system operations fail."""
    
    def __init__(self, operation: str, path: str, **kwargs):
        message = f"File system error during {operation}: {path}"
        
        super().__init__(
            message,
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Check file permissions",
                "Ensure directory exists",
                "Verify disk space availability",
                "Check path validity"
            ],
            metadata={"operation": operation, "path": path},
            **kwargs
        )

class DocumentProcessingError(ResumeParserException):
    """Raised when document processing operations fail."""
    
    def __init__(self, message: str = "Document processing failed", document_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check document format",
                "Verify document is not corrupted", 
                "Try with a different file",
                "Ensure document is readable"
            ],
            metadata={"document_type": document_type},
            **kwargs
        )

class ErrorHandler:
    """Centralized error handler with logging and user feedback."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self._error_count = 0
        self._error_history: List[ErrorDetails] = []
        
    def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        user_friendly: bool = True
    ) -> ErrorDetails:
        """
        Handle an error and return structured error details.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            user_friendly: Whether to return user-friendly messages
        
        Returns:
            ErrorDetails object with structured error information
        """
        self._error_count += 1
        
        # Generate unique error ID
        error_id = f"ERR-{datetime.now().strftime('%Y%m%d')}-{self._error_count:04d}"
        
        # Extract error information
        if isinstance(error, ResumeParserException):
            error_details = ErrorDetails(
                error_id=error_id,
                category=error.category,
                severity=error.severity,
                message=error.message,
                technical_message=str(error),
                suggestions=error.suggestions,
                metadata=error.metadata
            )
        else:
            # Handle generic exceptions
            error_details = ErrorDetails(
                error_id=error_id,
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                message="An unexpected error occurred" if user_friendly else str(error),
                technical_message=str(error),
                suggestions=["Please try again", "Contact support if the problem persists"]
            )
        
        # Add context if provided
        if context:
            error_details.metadata.update(context)
        
        # Add traceback for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            error_details.traceback_info = traceback.format_exc()
        
        # Log the error
        self._log_error(error_details, error)
        
        # Store in history
        self._error_history.append(error_details)
        
        # Keep only last 100 errors
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]
        
        return error_details
    
    def _log_error(self, error_details: ErrorDetails, original_error: Exception):
        """Log error with appropriate level based on severity."""
        log_message = f"[{error_details.error_id}] {error_details.message}"
        
        if error_details.metadata:
            log_message += f" | Metadata: {error_details.metadata}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=original_error)
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=original_error)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self._error_history:
            return {"total_errors": 0}
        
        category_counts = {}
        severity_counts = {}
        
        for error in self._error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self._error_history),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "recent_errors": [error.to_dict() for error in self._error_history[-5:]]
        }
    
    def clear_history(self):
        """Clear error history."""
        self._error_history.clear()
        self._error_count = 0

# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def set_error_handler(handler: ErrorHandler):
    """Set the global error handler instance."""
    global _error_handler
    _error_handler = handler

def handle_error(error: Exception, **kwargs) -> ErrorDetails:
    """Convenience function to handle errors using the global handler."""
    return get_error_handler().handle_error(error, **kwargs)

# Decorator for automatic error handling
def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    user_friendly: bool = True
):
    """Decorator to automatically handle function errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, ResumeParserException):
                    # Convert to custom exception
                    e = ResumeParserException(
                        message=str(e),
                        category=category,
                        severity=severity
                    )
                
                error_details = handle_error(e, user_friendly=user_friendly)
                
                # Return error result instead of raising
                return {
                    "error": True,
                    "error_details": error_details.to_dict()
                }
        
        return wrapper
    return decorator

# Utility functions for common error scenarios
def create_connection_error(service: str, url: str) -> ConnectionError:
    """Create a standardized connection error."""
    return ConnectionError(service, url)

def create_model_error(model: str, available: List[str] = None) -> ModelNotAvailableError:
    """Create a standardized model availability error."""
    return ModelNotAvailableError(model, available)

def create_parse_error(file_type: str, reason: str = None) -> ParseError:
    """Create a standardized parsing error."""
    return ParseError(file_type, reason)

def create_validation_error(field: str, value: Any, expected: str = None) -> ValidationError:
    """Create a standardized validation error."""
    return ValidationError(field, value, expected)