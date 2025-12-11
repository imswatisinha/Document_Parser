"""
Model Selection Service
Intelligently selects the optimal AI model based on document characteristics.
"""

import requests
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from enum import Enum
from functools import lru_cache
import hashlib

from config.settings import get_config
from core.exceptions import ModelNotAvailableError, ConnectionError as ConnError
from core.logging_system import get_logger, log_performance
from models.domain_models import DocumentAnalysis, ComplexityLevel

logger = get_logger(__name__)

class ModelCapability(str, Enum):
    """Model capabilities for different tasks."""
    FAST_PARSING = "fast_parsing"
    DETAILED_ANALYSIS = "detailed_analysis"
    TECHNICAL_CONTENT = "technical_content"
    MULTILINGUAL = "multilingual"
    LARGE_CONTEXT = "large_context"

@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    display_name: str
    memory_gb: float
    context_length: int
    speed_score: float  # 1.0 = fastest, higher = slower
    quality_score: float  # 1.0 = highest quality
    capabilities: List[ModelCapability]
    recommended_for: List[str]
    
    def __post_init__(self):
        """Validate model info."""
        if self.speed_score <= 0 or self.quality_score <= 0:
            raise ValueError("Speed and quality scores must be positive")
        if self.memory_gb <= 0:
            raise ValueError("Memory requirement must be positive")

class DocumentComplexityAnalyzer:
    """Analyzes document complexity to inform model selection."""
    
    def __init__(self):
        self.technical_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|go|rust|swift|kotlin)\b',
            # Technologies
            r'\b(react|angular|vue|django|flask|spring|docker|kubernetes)\b',
            # Databases
            r'\b(mysql|postgresql|mongodb|redis|elasticsearch)\b',
            # Cloud platforms
            r'\b(aws|azure|gcp|google cloud|amazon web services)\b',
            # Development tools
            r'\b(git|jenkins|ci/cd|devops|agile|scrum)\b'
        ]
        
        self.complexity_indicators = {
            'has_tables': r'\|.*\|.*\|',
            'has_code': r'```|def |class |function|import |#include',
            'has_math': r'\$.*\$|\\frac|\\sum|\\int',
            'has_structured_lists': r'^\s*[•·\-\*]\s+',
            'has_urls': r'https?://[^\s]+',
            'has_emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        logger.info("Document complexity analyzer initialized")
    
    @log_performance(threshold_seconds=2.0)
    def analyze_document(self, text: str) -> DocumentAnalysis:
        """Analyze document complexity and characteristics."""
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        # Content indicators
        indicators = {}
        for indicator, pattern in self.complexity_indicators.items():
            indicators[indicator] = bool(re.search(pattern, text, re.MULTILINE | re.IGNORECASE))
        
        # Technical content analysis
        technical_terms_count = 0
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_terms_count += len(matches)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            char_count, word_count, technical_terms_count, indicators
        )
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(complexity_score, char_count)
        
        # Get model recommendations
        recommended_models = self._get_model_recommendations(complexity_level, indicators)
        
        # Estimate processing time
        processing_time = self._estimate_processing_time(char_count, complexity_score)
        
        analysis = DocumentAnalysis(
            complexity_level=complexity_level,
            character_count=char_count,
            word_count=word_count,
            line_count=line_count,
            has_tables=indicators['has_tables'],
            has_code=indicators['has_code'],
            has_math=indicators['has_math'],
            has_lists=indicators['has_structured_lists'],
            technical_terms_count=technical_terms_count,
            complexity_score=complexity_score,
            recommended_models=recommended_models,
            processing_time_estimate=processing_time
        )
        
        logger.info(f"Document analysis complete: {complexity_level.value} complexity, "
                   f"{char_count} chars, {technical_terms_count} technical terms")
        
        return analysis
    
    def _calculate_complexity_score(
        self, 
        char_count: int, 
        word_count: int,
        technical_terms: int,
        indicators: Dict[str, bool]
    ) -> float:
        """Calculate overall complexity score (0-10)."""
        score = 0.0
        
        # Size-based scoring (0-3 points)
        if char_count > 15000:
            score += 3.0
        elif char_count > 8000:
            score += 2.0
        elif char_count > 3000:
            score += 1.0
        
        # Content-based scoring (0-4 points)
        content_factors = [
            indicators['has_tables'],
            indicators['has_code'],
            indicators['has_math'],
            technical_terms > 5
        ]
        score += sum(content_factors)
        
        # Technical complexity (0-3 points)
        if technical_terms > 20:
            score += 3.0
        elif technical_terms > 10:
            score += 2.0
        elif technical_terms > 5:
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _determine_complexity_level(self, complexity_score: float, char_count: int) -> ComplexityLevel:
        """Determine complexity level based on score and size."""
        if complexity_score >= 7.0:
            return ComplexityLevel.VERY_HIGH
        elif complexity_score >= 5.0:
            return ComplexityLevel.HIGH
        elif complexity_score >= 3.0 or char_count > 5000:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _get_model_recommendations(
        self, 
        complexity: ComplexityLevel, 
        indicators: Dict[str, bool]
    ) -> List[str]:
        """Get model recommendations - simplified for two models."""
        # Simple logic: use llama3.2:3b for medium+ complexity or technical content, phi3:mini for simple
        if complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH] or indicators.get('has_code') or indicators.get('has_math'):
            return ["llama3.2:3b", "phi3:mini"]
        else:
            return ["phi3:mini", "llama3.2:3b"]
    
    def _estimate_processing_time(self, char_count: int, complexity_score: float) -> float:
        """Estimate processing time in seconds."""
        base_time = char_count / 1000.0  # 1 second per 1000 characters
        complexity_multiplier = 1.0 + (complexity_score / 10.0)  # 1x to 2x based on complexity
        return base_time * complexity_multiplier

class ModelSelectionService:
    """Service for intelligent model selection and management."""
    
    def __init__(self):
        self.config = get_config()
        self.analyzer = DocumentComplexityAnalyzer()
        self._available_models: List[str] = []
        self._model_info: Dict[str, ModelInfo] = self._initialize_model_info()
        self._last_model_check: Optional[datetime] = None
        self._model_cache_ttl = timedelta(minutes=5)
        
        logger.info("Model selection service initialized")
    
    def _initialize_model_info(self) -> Dict[str, ModelInfo]:
        """Initialize model information database - simplified to only two models."""
        models = {
            "phi3:mini": ModelInfo(
                name="phi3:mini",
                display_name="Phi-3 Mini",
                memory_gb=2.0,
                context_length=4096,
                speed_score=1.0,  # Fastest
                quality_score=0.7,
                capabilities=[ModelCapability.FAST_PARSING],
                recommended_for=["small documents", "quick parsing", "basic extraction"]
            ),
            "llama3.2:3b": ModelInfo(
                name="llama3.2:3b", 
                display_name="Llama 3.2 3B",
                memory_gb=3.5,
                context_length=8192,
                speed_score=1.5,
                quality_score=0.9,
                capabilities=[
                    ModelCapability.DETAILED_ANALYSIS,
                    ModelCapability.TECHNICAL_CONTENT,
                    ModelCapability.MULTILINGUAL
                ],
                recommended_for=["technical documents", "detailed analysis", "high quality"]
            )
        }
        
        logger.info(f"Initialized {len(models)} model configurations")
        return models
    
    @log_performance(threshold_seconds=5.0)
    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of available Ollama models - filtered to supported models only with enhanced caching."""
        # Check cache first (extended TTL for better performance)
        now = datetime.now()
        if (not force_refresh and 
            self._last_model_check and 
            (now - self._last_model_check) < self._model_cache_ttl and
            self._available_models):
            logger.debug(f"Returning cached models: {self._available_models}")
            return self._available_models
        
        try:
            response = requests.get(
                f"{self.config.ollama.base_url}/api/tags",
                timeout=self.config.ollama.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            all_models = [model['name'] for model in models_data.get('models', [])]
            
            # Filter to only supported models (llama3.2:3b and phi3:mini) for performance
            supported_models = ["llama3.2:3b", "phi3:mini"]
            self._available_models = [m for m in all_models if m in supported_models]
            self._last_model_check = now
            
            logger.info(f"Retrieved {len(self._available_models)} supported models from {len(all_models)} total")
            return self._available_models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get available models: {str(e)}")
            raise ConnError("Ollama", self.config.ollama.base_url)
    
    @lru_cache(maxsize=128)
    def _cached_document_analysis(self, text_hash: str) -> DocumentAnalysis:
        """Cache document analysis results by text hash."""
        # This is a helper method - actual analysis done in analyzer
        pass
    
    def select_optimal_model(
        self, 
        document_text: str,
        preferred_model: str = None,
        requirements: Dict[str, Any] = None
    ) -> Tuple[str, DocumentAnalysis]:
        """Select the optimal model for processing a document - optimized for two models."""
        # Analyze document
        analysis = self.analyzer.analyze_document(document_text)
        
        # Get available models
        available_models = self.get_available_models()
        
        if not available_models:
            raise ModelNotAvailableError("no_models", [])
        
        # Filter to only supported models (llama3.2:3b and phi3:mini)
        supported_models = ["llama3.2:3b", "phi3:mini"]
        available_supported = [m for m in available_models if m in supported_models]
        
        if not available_supported:
            # Fallback to first available model if neither supported model is available
            fallback_model = available_models[0]
            logger.warning(f"No supported models available, using fallback: {fallback_model}")
            return fallback_model, analysis
        
        # If preferred model is specified and available, use it
        if preferred_model and preferred_model in available_supported:
            logger.info(f"Using preferred model: {preferred_model}")
            return preferred_model, analysis
        
        # Simple selection: use llama3.2:3b for medium+ complexity, phi3:mini for low complexity
        if analysis.complexity_level in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH]:
            selected = "llama3.2:3b" if "llama3.2:3b" in available_supported else available_supported[0]
        else:
            selected = "phi3:mini" if "phi3:mini" in available_supported else available_supported[0]
        
        logger.info(f"Selected model '{selected}' for {analysis.complexity_level.value} "
                   f"complexity document ({analysis.character_count} chars)")
        
        return selected, analysis
    
    def _select_by_requirements(
        self,
        available_models: List[str],
        analysis: DocumentAnalysis,
        requirements: Dict[str, Any]
    ) -> str:
        """Select model based on specific requirements."""
        # Score each available model
        model_scores = {}
        
        for model_name in available_models:
            model_info = self._model_info[model_name]
            score = 0.0
            
            # Base score from recommended models
            if model_name in analysis.recommended_models:
                score += 10.0 - analysis.recommended_models.index(model_name) * 2.0
            
            # Complexity-based scoring - simplified for two models
            complexity_bonus = {
                ComplexityLevel.LOW: {"phi3:mini": 5.0},
                ComplexityLevel.MEDIUM: {"llama3.2:3b": 5.0, "phi3:mini": 3.0},
                ComplexityLevel.HIGH: {"llama3.2:3b": 5.0},
                ComplexityLevel.VERY_HIGH: {"llama3.2:3b": 5.0}
            }
            score += complexity_bonus.get(analysis.complexity_level, {}).get(model_name, 0.0)
            
            # Speed preference
            if requirements.get("prefer_speed", False):
                score += (5.0 / model_info.speed_score)  # Higher score for faster models
            
            # Quality preference
            if requirements.get("prefer_quality", True):
                score += model_info.quality_score * 3.0
            
            # Technical content bonus
            if analysis.has_code or analysis.technical_terms_count > 5:
                if ModelCapability.TECHNICAL_CONTENT in model_info.capabilities:
                    score += 3.0
            
            # Context length requirement
            estimated_tokens = analysis.character_count / 4  # Rough token estimate
            if estimated_tokens > model_info.context_length:
                score -= 10.0  # Heavy penalty for insufficient context
            
            model_scores[model_name] = score
            
            logger.debug(f"Model {model_name} scored {score:.2f}")
        
        # Select highest scoring model
        if not model_scores:
            return available_models[0]  # Fallback
        
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        return best_model
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self._model_info.get(model_name)
    
    def get_model_recommendations(
        self, 
        complexity_level: ComplexityLevel,
        content_type: str = None
    ) -> List[Dict[str, Any]]:
        """Get model recommendations with detailed information."""
        available = self.get_available_models()
        recommendations = []
        
        for model_name in available:
            model_info = self._model_info.get(model_name)
            if not model_info:
                continue
            
            # Calculate suitability score
            suitability = self._calculate_suitability(model_info, complexity_level)
            
            recommendations.append({
                "name": model_name,
                "display_name": model_info.display_name,
                "suitability_score": suitability,
                "memory_gb": model_info.memory_gb,
                "speed_rating": "fast" if model_info.speed_score <= 1.5 else "medium" if model_info.speed_score <= 2.5 else "slow",
                "quality_rating": "high" if model_info.quality_score >= 0.9 else "medium" if model_info.quality_score >= 0.7 else "basic",
                "recommended_for": model_info.recommended_for,
                "capabilities": [cap.value for cap in model_info.capabilities]
            })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return recommendations
    
    def _calculate_suitability(self, model_info: ModelInfo, complexity: ComplexityLevel) -> float:
        """Calculate model suitability score for given complexity - simplified for two models."""
        # Base suitability mapping
        base_scores = {
            ComplexityLevel.LOW: {
                "phi3:mini": 0.9,
                "llama3.2:3b": 0.6
            },
            ComplexityLevel.MEDIUM: {
                "phi3:mini": 0.6,
                "llama3.2:3b": 0.9
            },
            ComplexityLevel.HIGH: {
                "phi3:mini": 0.3,
                "llama3.2:3b": 0.9
            },
            ComplexityLevel.VERY_HIGH: {
                "phi3:mini": 0.1,
                "llama3.2:3b": 0.9
            }
        }
        
        return base_scores.get(complexity, {}).get(model_info.name, 0.5)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "available_models": len(self._available_models),
            "configured_models": len(self._model_info),
            "last_model_check": self._last_model_check.isoformat() if self._last_model_check else None,
            "cache_ttl_minutes": self._model_cache_ttl.total_seconds() / 60
        }

# Global service instance
_model_service: Optional[ModelSelectionService] = None

def get_model_service() -> ModelSelectionService:
    """Get the global model selection service."""
    global _model_service
    if _model_service is None:
        _model_service = ModelSelectionService()
    return _model_service

# Convenience functions
def analyze_document(text: str) -> DocumentAnalysis:
    """Analyze document complexity."""
    return get_model_service().analyzer.analyze_document(text)

def select_model(text: str, preferred: str = None, **kwargs) -> Tuple[str, DocumentAnalysis]:
    """Select optimal model for document."""
    return get_model_service().select_optimal_model(text, preferred, kwargs)

def get_available_models() -> List[str]:
    """Get available models."""
    return get_model_service().get_available_models()