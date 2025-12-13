"""
Comprehensive Logging System
Provides structured, configurable logging across the application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json
import functools
from config.settings import get_config

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)

class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True

class LoggerManager:
    """Centralized logger management."""
    
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._configured = False
        self._context_filters: Dict[str, ContextFilter] = {}
        
    def setup_logging(self, config: Optional[Dict[str, Any]] = None):
        """Set up logging configuration."""
        if self._configured:
            return
        
        app_config = get_config()
        log_config = config or {
            "level": app_config.logging.level,
            "format": app_config.logging.format,
            "file_path": app_config.logging.file_path,
            "max_file_size_mb": app_config.logging.max_file_size_mb,
            "backup_count": app_config.logging.backup_count,
            "enable_console": app_config.logging.enable_console
        }
        
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(getattr(logging, log_config["level"]))
        
        # Create formatters
        console_formatter = logging.Formatter(log_config["format"])
        file_formatter = JsonFormatter() if app_config.environment == "production" else console_formatter
        
        # Console handler
        if log_config["enable_console"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, log_config["level"]))
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_config["file_path"]:
            log_path = Path(log_config["file_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=log_config["max_file_size_mb"] * 1024 * 1024,
                backupCount=log_config["backup_count"],
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, log_config["level"]))
            root_logger.addHandler(file_handler)
        
        # Set up third-party library logging levels
        self._configure_third_party_loggers()
        
        self._configured = True
    
    def _configure_third_party_loggers(self):
        """Configure logging for third-party libraries."""
        third_party_loggers = {
            "requests": logging.WARNING,
            "urllib3": logging.WARNING,
            "pinecone": logging.INFO,
            "streamlit": logging.WARNING,
            "sentence_transformers": logging.WARNING,
            "transformers": logging.WARNING,
            "torch": logging.WARNING,
            "langchain": logging.INFO
        }
        
        for logger_name, level in third_party_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
    
    def get_logger(self, name: str, context: Dict[str, Any] = None) -> logging.Logger:
        """Get or create a logger with optional context."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            
            # Add context filter if provided
            if context:
                context_filter = ContextFilter(context)
                logger.addFilter(context_filter)
                self._context_filters[name] = context_filter
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def add_context(self, logger_name: str, context: Dict[str, Any]):
        """Add context to an existing logger."""
        if logger_name in self._context_filters:
            self._context_filters[logger_name].context.update(context)
        elif logger_name in self._loggers:
            context_filter = ContextFilter(context)
            self._loggers[logger_name].addFilter(context_filter)
            self._context_filters[logger_name] = context_filter
    
    def remove_context(self, logger_name: str, keys: Union[str, list]):
        """Remove context keys from logger."""
        if logger_name in self._context_filters:
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                self._context_filters[logger_name].context.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "configured": self._configured,
            "loggers_count": len(self._loggers),
            "loggers": list(self._loggers.keys()),
            "context_filters": len(self._context_filters)
        }

# Global logger manager
_logger_manager: Optional[LoggerManager] = None

def get_logger_manager() -> LoggerManager:
    """Get the global logger manager."""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
        _logger_manager.setup_logging()
    return _logger_manager

def get_logger(name: str = None, context: Dict[str, Any] = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return get_logger_manager().get_logger(name, context)

def setup_logging(config: Dict[str, Any] = None):
    """Set up application logging."""
    get_logger_manager().setup_logging(config)

# Decorators for automatic logging
def log_function_calls(
    logger: logging.Logger = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False
):
    """Decorator to log function calls."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            log_data = {"function": func_name, "action": "enter"}
            
            if include_args and (args or kwargs):
                log_data["args"] = {
                    "positional": args[:3] if len(args) > 3 else args,  # Limit args for privacy
                    "keyword": {k: v for k, v in list(kwargs.items())[:5]}  # Limit kwargs
                }
                if len(args) > 3:
                    log_data["args"]["positional_truncated"] = len(args) - 3
                if len(kwargs) > 5:
                    log_data["args"]["keyword_truncated"] = len(kwargs) - 5
            
            logger.log(level, f"Entering {func_name}", extra=log_data)
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                
                # Log function success
                duration = (datetime.now() - start_time).total_seconds()
                success_data = {
                    "function": func_name,
                    "action": "exit",
                    "status": "success",
                    "duration_seconds": duration
                }
                
                if include_result and result is not None:
                    # Be careful with large results
                    result_str = str(result)
                    if len(result_str) > 200:
                        success_data["result"] = result_str[:200] + "..."
                        success_data["result_truncated"] = True
                    else:
                        success_data["result"] = result
                
                logger.log(level, f"Exiting {func_name} successfully", extra=success_data)
                return result
                
            except Exception as e:
                # Log function error
                duration = (datetime.now() - start_time).total_seconds()
                error_data = {
                    "function": func_name,
                    "action": "exit",
                    "status": "error",
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                logger.error(f"Error in {func_name}: {str(e)}", extra=error_data)
                raise
        
        return wrapper
    return decorator

def log_performance(threshold_seconds: float = 1.0, logger: logging.Logger = None):
    """Decorator to log slow function executions."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            if duration > threshold_seconds:
                func_name = f"{func.__module__}.{func.__qualname__}"
                perf_data = {
                    "function": func_name,
                    "duration_seconds": duration,
                    "threshold_seconds": threshold_seconds,
                    "slow_execution": True
                }
                
                logger.warning(
                    f"Slow execution detected: {func_name} took {duration:.2f}s",
                    extra=perf_data
                )
            
            return result
        
        return wrapper
    return decorator

# Context managers for structured logging
class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.filter = None
    
    def __enter__(self):
        self.filter = ContextFilter(self.context)
        self.logger.addFilter(self.filter)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filter:
            self.logger.removeFilter(self.filter)

def log_context(logger: logging.Logger, **context):
    """Create a logging context manager."""
    return LogContext(logger, **context)

# Utility functions
def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any] = None,
    level: int = logging.ERROR
):
    """Log error with additional context."""
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {})
    }
    
    logger.log(level, f"Error occurred: {str(error)}", extra=error_data, exc_info=error)

def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Union[int, float, str]],
    level: int = logging.INFO
):
    """Log application metrics."""
    metrics_data = {
        "metrics": metrics,
        "metric_type": "application"
    }
    
    logger.log(level, "Application metrics", extra=metrics_data)

# Application-specific loggers
def get_parser_logger() -> logging.Logger:
    """Get logger for parsing operations."""
    return get_logger("resume_parser.parsing", {"component": "parser"})

def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return get_logger("resume_parser.model", {"component": "model"})

def get_ui_logger() -> logging.Logger:
    """Get logger for UI operations."""
    return get_logger("resume_parser.ui", {"component": "ui"})

def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return get_logger("resume_parser.api", {"component": "api"})

def get_performance_logger() -> logging.Logger:
    """Get logger for performance monitoring."""
    return get_logger("resume_parser.performance", {"component": "performance"})

# Initialize logging on import
setup_logging()