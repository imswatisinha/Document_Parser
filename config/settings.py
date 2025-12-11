"""
Centralized Configuration System for Resume Parser LLM
Provides type-safe configuration management across the application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

@dataclass
class OllamaConfig:
    """Configuration for Ollama service integration."""
    base_url: str = "http://localhost:11434"
    timeout: int = 180
    max_retries: int = 3
    default_models: List[str] = field(default_factory=lambda: [
        "llama3.2:3b", 
        "phi3:mini"
    ])
    recommended_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "small_docs": ["phi3:mini"],
        "medium_docs": ["llama3.2:3b"],
        "large_docs": ["llama3.2:3b"],
        "technical_docs": ["llama3.2:3b"]
    })

@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: Optional[str] = None
    environment: str = "us-west1-gcp"
    index_name: str = "resume-parser"
    dimension: int = 1536
    metric: str = "cosine"
    timeout: int = 120
    enabled: bool = True  # Enable Pinecone by default if API key is provided

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    similarity_threshold: float = 0.7
    max_results: int = 5

@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    page_title: str = "Resume Parser LLM"
    page_icon: str = "📄"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enable_model_cache: bool = True
    enable_embedding_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 128
    cache_directory: str = ".cache"

@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/app.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    max_request_size_mb: int = 50
    rate_limit_per_minute: int = 100
    enable_api_key_auth: bool = False
    session_timeout_minutes: int = 60

@dataclass
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment detection
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_directories()
        self._load_environment_overrides()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.processing.chunk_size < self.processing.min_chunk_size:
            raise ValueError(f"chunk_size ({self.processing.chunk_size}) must be >= min_chunk_size ({self.processing.min_chunk_size})")
        
        if self.processing.chunk_size > self.processing.max_chunk_size:
            raise ValueError(f"chunk_size ({self.processing.chunk_size}) must be <= max_chunk_size ({self.processing.max_chunk_size})")
        
        if self.cache.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        
        if self.ollama.timeout <= 0:
            raise ValueError("ollama timeout must be positive")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = []
        
        if self.logging.file_path:
            log_dir = Path(self.logging.file_path).parent
            directories.append(log_dir)
        
        if self.cache.cache_directory:
            directories.append(Path(self.cache.cache_directory))
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Ollama overrides
        if os.getenv("OLLAMA_BASE_URL"):
            self.ollama.base_url = os.getenv("OLLAMA_BASE_URL")
        
        if os.getenv("OLLAMA_TIMEOUT"):
            self.ollama.timeout = int(os.getenv("OLLAMA_TIMEOUT"))
        
        # Pinecone overrides
        if os.getenv("PINECONE_API_KEY"):
            self.pinecone.api_key = os.getenv("PINECONE_API_KEY")
            self.pinecone.enabled = True  # Auto-enable if API key is provided
        
        if os.getenv("PINECONE_ENABLED"):
            self.pinecone.enabled = os.getenv("PINECONE_ENABLED").lower() == "true"
        
        if os.getenv("PINECONE_ENVIRONMENT"):
            self.pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
        
        if os.getenv("PINECONE_INDEX_NAME"):
            self.pinecone.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Processing overrides
        if os.getenv("CHUNK_SIZE"):
            self.processing.chunk_size = int(os.getenv("CHUNK_SIZE"))
        
        if os.getenv("CHUNK_OVERLAP"):
            self.processing.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        
        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL").upper()
        
        if os.getenv("LOG_FILE_PATH"):
            self.logging.file_path = os.getenv("LOG_FILE_PATH")
    
    def get_model_config(self, document_complexity: str = "medium") -> Dict[str, Any]:
        """Get model configuration based on document complexity."""
        complexity_mapping = {
            "small": "small_docs",
            "medium": "medium_docs", 
            "large": "large_docs",
            "technical": "technical_docs"
        }
        
        key = complexity_mapping.get(document_complexity, "medium_docs")
        recommended_models = self.ollama.recommended_models.get(key, self.ollama.default_models[:1])
        
        return {
            "models": recommended_models,
            "timeout": self.ollama.timeout,
            "max_retries": self.ollama.max_retries,
            "base_url": self.ollama.base_url
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enable_model_cache": self.cache.enable_model_cache,
            "enable_embedding_cache": self.cache.enable_embedding_cache,
            "ttl": self.cache.cache_ttl_seconds,
            "max_size": self.cache.max_cache_size,
            "directory": self.cache.cache_directory
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ollama": {
                "base_url": self.ollama.base_url,
                "timeout": self.ollama.timeout,
                "max_retries": self.ollama.max_retries,
                "default_models": self.ollama.default_models
            },
            "pinecone": {
                "environment": self.pinecone.environment,
                "index_name": self.pinecone.index_name,
                "dimension": self.pinecone.dimension,
                "metric": self.pinecone.metric
            },
            "processing": {
                "chunk_size": self.processing.chunk_size,
                "chunk_overlap": self.processing.chunk_overlap,
                "similarity_threshold": self.processing.similarity_threshold
            },
            "ui": {
                "page_title": self.ui.page_title,
                "layout": self.ui.layout,
                "max_file_size_mb": self.ui.max_file_size_mb
            },
            "environment": self.environment,
            "debug_mode": self.debug_mode
        }


# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

def set_config(config: AppConfig):
    """Set the global configuration instance."""
    global _config
    _config = config

def reset_config():
    """Reset configuration to default values."""
    global _config
    _config = None

# Environment-specific configurations
def get_development_config() -> AppConfig:
    """Get development environment configuration."""
    config = AppConfig()
    config.debug_mode = True
    config.logging.level = "DEBUG"
    config.logging.enable_console = True
    config.cache.cache_ttl_seconds = 300  # 5 minutes for faster testing
    return config

def get_production_config() -> AppConfig:
    """Get production environment configuration."""
    config = AppConfig()
    config.debug_mode = False
    config.logging.level = "WARNING"
    config.logging.enable_console = False
    config.cache.cache_ttl_seconds = 7200  # 2 hours
    config.security.enable_api_key_auth = True
    return config

def get_testing_config() -> AppConfig:
    """Get testing environment configuration."""
    config = AppConfig()
    config.debug_mode = True
    config.logging.level = "ERROR"
    config.logging.file_path = None  # No file logging during tests
    config.cache.enable_model_cache = False  # Disable caching for predictable tests
    config.cache.enable_embedding_cache = False
    return config