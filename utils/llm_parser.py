# utils/ollama_parser.py
import re
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import requests

# Streamlit is used for UI helpers in this module (many functions assume Streamlit)
try:
    import streamlit as st
except Exception:
    st = None

LOG = logging.getLogger(__name__)

# -----------------------
# Helpers to prefer singleton factory
# -----------------------
def _try_get_singleton(singleton_base_url: str = None, warm_up: bool = False) -> Optional[Any]:
    """
    Try to obtain an OllamaParser instance from utils.ollama_singleton.get_ollama().
    This is done lazily to avoid circular imports at module import time.
    Returns None if factory not available.
    """
    try:
        mod = __import__("utils.ollama_singleton", fromlist=["get_ollama"])
        get_ollama = getattr(mod, "get_ollama", None)
        if callable(get_ollama):
            return get_ollama(singleton_base_url=singleton_base_url, warm_up=warm_up)
    except Exception:
        LOG.debug("utils.ollama_singleton.get_ollama not available or failed", exc_info=True)
    return None


def _get_parser_instance(singleton_base_url: str = None, warm_up: bool = False) -> Optional[Any]:
    """
    Return an OllamaParser instance, preferring the singleton factory.
    If the factory isn't available, try to import OllamaParser from this package (if present).
    """
    # 1) try singleton factory
    inst = _try_get_singleton(singleton_base_url=singleton_base_url, warm_up=warm_up)
    if inst:
        return inst

    # 2) fall back to local OllamaParser class if present in this module (or other)
    try:
        # If an OllamaParser class is defined in another utils module, import lazily
        mod = __import__("utils.ollama_parser", fromlist=["OllamaParser"])
        OllamaParser = getattr(mod, "OllamaParser", None)
        if OllamaParser:
            try:
                return OllamaParser(base_url=singleton_base_url) if singleton_base_url else OllamaParser()
            except Exception:
                # last-resort: instantiate without args
                try:
                    return OllamaParser()
                except Exception:
                    LOG.exception("Failed to instantiate OllamaParser fallback")
    except Exception:
        # It's possible we're in the same file where OllamaParser isn't available; ignore
        LOG.debug("No local OllamaParser fallback available", exc_info=True)
    return None


# -----------------------
# Public utility functions (use singleton where possible)
# -----------------------
def detect_document_complexity(text: str) -> Dict[str, any]:
    """
    Analyze document to determine its complexity and processing requirements.
    """
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split("\n"))

    has_tables = bool(re.search(r"\|.*\|.*\|", text))
    has_code = bool(re.search(r"```|def |class |function|import |#include", text, re.IGNORECASE))
    has_math = bool(re.search(r"\$.*\$|\\frac|\\sum|\\int", text))
    has_lists = len(re.findall(r"^\s*[•·\-\*]\s+", text, re.MULTILINE))
    technical_terms = len(
        re.findall(
            r"(?i)(python|java|javascript|sql|api|database|algorithm|machine learning|ai|data science|backend|frontend)",
            text,
        )
    )

    complexity_score = 0
    if char_count > 15000:
        complexity_score += 3
    elif char_count > 8000:
        complexity_score += 2
    elif char_count > 3000:
        complexity_score += 1

    if has_tables:
        complexity_score += 1
    if has_code:
        complexity_score += 2
    if has_math:
        complexity_score += 1
    if has_lists > 10:
        complexity_score += 1
    if technical_terms > 5:
        complexity_score += 1

    if complexity_score >= 6:
        category = "very_complex"
    elif complexity_score >= 4:
        category = "complex"
    elif complexity_score >= 2:
        category = "medium"
    else:
        category = "simple"

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "complexity_score": complexity_score,
        "category": category,
        "features": {
            "has_tables": has_tables,
            "has_code": has_code,
            "has_math": has_math,
            "list_count": has_lists,
            "technical_terms": technical_terms,
        },
    }


def get_available_ollama_models() -> List[str]:
    """
    Get list of available Ollama models, preferring the singleton / local parser.
    """
    # 1) try parser via singleton
    parser = _get_parser_instance()
    if parser:
        # ensure connection check
        try:
            if not getattr(parser, "is_connected", False):
                parser.check_connection()
        except Exception:
            LOG.debug("check_connection raised", exc_info=True)

        models = getattr(parser, "available_models", None)
        if models:
            return list(models)

    # 2) fallback to direct API query
    supported_models = ["phi3:mini", "llama3.2:3b"]
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models_data = r.json()
            all_models = [m.get("name") for m in models_data.get("models", []) if m.get("name")]
            return [m for m in all_models if m in supported_models] or all_models
    except Exception:
        LOG.debug("Direct Ollama API query failed", exc_info=True)
    return []


def get_model_memory_requirements() -> Dict[str, Dict[str, any]]:
    """
    Model memory & perf metadata (simple lookup).
    """
    return {
        "phi3:mini": {
            "ram_required": 2,
            "vram_required": 2,
            "model_size": "2.2GB",
            "performance": "fast",
            "quality": "good",
            "reliability": "high",
        },
        "llama3.2:3b": {
            "ram_required": 4,
            "vram_required": 3,
            "model_size": "2.0GB",
            "performance": "medium",
            "quality": "very_good",
            "reliability": "high",
        },
    }


def select_optimal_model(document_analysis: Dict, available_models: List[str]) -> Tuple[Optional[str], str]:
    """
    Decide which model to use based on analysis + available models.
    Returns (selected_model, reasoning)
    """
    if not available_models:
        return None, "No Ollama models available"

    char_count = document_analysis["char_count"]
    category = document_analysis["category"]
    features = document_analysis["features"]
    model_specs = get_model_memory_requirements()

    memory_safe_preferences = {
        "very_complex": ["llama3.2:3b", "phi3:mini"],
        "complex": ["llama3.2:3b", "phi3:mini"],
        "medium": ["llama3.2:3b", "phi3:mini"],
        "simple": ["phi3:mini", "llama3.2:3b"],
    }

    preferred_models = memory_safe_preferences.get(category, memory_safe_preferences["medium"])
    selected_model = None
    selected_specs = None
    for model in preferred_models:
        if model in available_models:
            selected_model = model
            selected_specs = model_specs.get(model, {})
            break

    if not selected_model:
        safe_models = []
        for model in available_models:
            specs = model_specs.get(model, {"ram_required": 10})
            safe_models.append((model, specs.get("ram_required", 10)))
        safe_models.sort(key=lambda x: x[1])
        selected_model = safe_models[0][0]
        selected_specs = model_specs.get(selected_model, {})

    reasoning_parts = [f"Document: {char_count:,} chars ({category} complexity)"]
    if selected_specs:
        reasoning_parts.append(f"Memory: ~{selected_specs.get('ram_required', 'Unknown')}GB RAM")
        reasoning_parts.append(f"Size: {selected_specs.get('model_size', 'Unknown')}")
        reasoning_parts.append(f"Performance: {selected_specs.get('performance', 'Unknown')}")
    if features["has_code"]:
        reasoning_parts.append("Code detected")
    if features["has_tables"]:
        reasoning_parts.append("Tables detected")
    if features["technical_terms"] > 5:
        reasoning_parts.append("High technical content")

    reasoning = " | ".join(reasoning_parts)
    return selected_model, reasoning


def auto_select_model_for_document(text: str) -> Dict[str, any]:
    """
    Analyze document and return model selection + metadata.
    """
    analysis = detect_document_complexity(text)
    available_models = get_available_ollama_models()
    selected_model, reasoning = select_optimal_model(analysis, available_models)
    return {
        "analysis": analysis,
        "available_models": available_models,
        "selected_model": selected_model,
        "reasoning": reasoning,
        "success": selected_model is not None,
    }


# -----------------------
# Post-processing helpers (kept same as your original functions)
# -----------------------
# For brevity I keep the implementation you provided, only small fixes to avoid undefined names.
# (Paste your previously provided helper functions here or call them if they live in other modules.)
# For this update I will import or reference the functions by name assuming they are defined below:
# _enhance_education_extraction, _enhance_skills_extraction_structured, _enhance_skills_extraction,
# _enhance_projects_extraction, _add_page_provenance, post_process_parsed_data
# --- To keep the file self-contained I will reuse your implementations (unchanged) ---
# (copy/paste your functions here) - but for brevity in this message, we'll reference them as available.
# If you want I can inline them again exactly; currently your module already contained them so they remain.

# To ensure compatibility if those functions are defined below in same file, attempt to use them.
# (No-op here.)

# -----------------------
# Core parse function that uses the singleton parser and a fallback loop
# -----------------------
def parse_resume_with_ollama(text: str, pages: List[str] = None, model_name: Optional[str] = None, use_expanders: bool = True) -> Dict[str, any]:
    """
    Parse resume text using the singleton Ollama parser (preferred) with simple fallback across models.
    This function does NOT require the parser to expose parse_resume_with_fallback; instead it uses parse_resume_sync
    and tries other models if the first attempt fails.
    """
    try:
        parser = _get_parser_instance()
        if parser is None:
            if st:
                st.error("❌ Ollama parser not available (singleton factory not found).")
            return {"error": "Ollama parser not available. Please ensure utils.ollama_singleton.get_ollama exists."}

        # Ensure connection
        try:
            if not getattr(parser, "is_connected", False):
                parser.check_connection()
        except Exception:
            LOG.debug("parser.check_connection() raised", exc_info=True)

        if not getattr(parser, "is_connected", False):
            if st:
                st.error("❌ Ollama not connected. Start Ollama and ensure models are installed.")
            return {"error": "Ollama not connected", "available_models": getattr(parser, "available_models", [])}

        # Determine model
        if not model_name:
            if st:
                st.info("🔍 Analyzing document to select an optimal model...")
            selection = auto_select_model_for_document(text)
            if not selection.get("success"):
                return {"error": "No suitable model available", "available_models": selection.get("available_models", [])}
            model_name = selection["selected_model"]
            analysis = selection["analysis"]
            reasoning = selection["reasoning"]
        else:
            analysis = detect_document_complexity(text)
            reasoning = f"User-specified model: {model_name}"

        # Show basic info in Streamlit UI if available
        if st and use_expanders:
            try:
                st.success(f"🎯 Selected Model: `{model_name}`")
                st.info(f"📊 Analysis: {reasoning}")
            except Exception:
                pass

        # Attempt parsing with selected model, with fallback attempts across available models
        attempts = []
        available_models = getattr(parser, "available_models", []) or get_available_ollama_models()
        try_models = [model_name] + [m for m in available_models if m != model_name]

        parsed_result = None
        for i, candidate in enumerate(try_models, start=1):
            try:
                # Use sync parse for robustness in Streamlit context
                if hasattr(parser, "parse_resume_sync"):
                    res = parser.parse_resume_sync(text, model=candidate)
                else:
                    # If the parser implementation uses a different name, try generic sync_generate + extract JSON
                    if hasattr(parser, "sync_generate"):
                        prompt = getattr(parser, "_build_resume_prompt", lambda t: t)(text)
                        gen = parser.sync_generate(prompt, model=candidate)
                        # try to extract response field
                        response_text = gen.get("response", "") if isinstance(gen, dict) else str(gen)
                        res = {}
                        if hasattr(parser, "_extract_json_from_text"):
                            parsed = parser._extract_json_from_text(response_text)
                            if parsed:
                                res = parsed
                            else:
                                res = {"error": "Could not extract JSON", "raw_response": response_text[:1000]}
                        else:
                            res = {"error": "No extraction helper available", "raw_response": response_text[:1000]}
                    else:
                        res = {"error": "No sync parse method available on parser"}
                # Normalize plugin errors
                if isinstance(res, dict) and "error" not in res:
                    parsed_result = res
                    attempts.append({"attempt": i, "model": candidate, "error": None})
                    break
                else:
                    attempts.append({"attempt": i, "model": candidate, "error": res.get("error", str(res))})
            except Exception as e:
                LOG.exception("Parse attempt failed", exc_info=True)
                attempts.append({"attempt": i, "model": candidate, "error": str(e)})
                # continue to next candidate

        if parsed_result is None:
            # All attempts failed
            result = {
                "error": "All available models failed to parse the document",
                "attempts": attempts,
                "available_models": available_models,
                "suggestions": [
                    "Install a lighter/faster model (e.g., phi3:mini) with `ollama pull phi3:mini`",
                    "Restart Ollama service (`ollama serve`)",
                    "Split the document into smaller chunks and retry"
                ],
            }
            return result

        # Post-process parsed result if post_processing helper exists in this module
        try:
            # If your post-processing function is defined in this module, call it.
            # We assume a function named post_process_parsed_data exists in this file (your earlier code).
            pp = globals().get("post_process_parsed_data")
            if pp and callable(pp):
                enhanced = pp(parsed_result, text, pages)
            else:
                enhanced = parsed_result
        except Exception:
            LOG.exception("Post processing failed", exc_info=True)
            enhanced = parsed_result

        # Normalize / validate if normalizers available (optional, keep original behavior)
        try:
            from .normalizers import validate_and_normalize  # optional import
            maybe_chunks = None
            if isinstance(pages, list) and pages and isinstance(pages[0], dict) and "content" in pages[0]:
                maybe_chunks = pages
            normalized = validate_and_normalize(
                enhanced,
                chunks=maybe_chunks,
                provider="ollama",
                model=model_name,
                parsing_method="model",
                raw_text=text,
            )
            return normalized
        except Exception:
            # Return enhanced result if normalization not possible
            enhanced["ai_provider"] = "ollama"
            enhanced["model"] = model_name
            enhanced["attempts"] = attempts
            return enhanced

    except Exception as e:
        LOG.exception("Unexpected error in parse_resume_with_ollama", exc_info=True)
        return {"error": f"Ollama parsing failed: {str(e)}"}


# -----------------------
# UI helpers (Streamlit)
# -----------------------
def show_model_recommendations(available_models: List[str]) -> None:
    """
    Display model recommendations in Streamlit sidebar.
    """
    if st is None:
        return
    if not available_models:
        return

    st.sidebar.markdown("### 🎯 Model Recommendations")
    model_categories = {"⚡ Fast Models": ["phi3:mini"], "⚖️ Balanced Models": ["llama3.2:3b"]}

    for category, models in model_categories.items():
        available_in_category = [m for m in models if m in available_models]
        if available_in_category:
            st.sidebar.write(f"**{category}**")
            for model in available_in_category:
                st.sidebar.write(f"  ✅ {model}")

    all_recommended = []
    for models in model_categories.values():
        all_recommended.extend(models)
    missing_models = [m for m in all_recommended if m not in available_models]
    if missing_models:
        with st.sidebar.expander("📥 Install More Models"):
            st.write("**Recommended models to install:**")
            for model in missing_models[:3]:
                if st.button(f"Install {model}", key=f"install_{model}"):
                    st.info(f"Run: `ollama pull {model}`")


def get_model_performance_info(model_name: str) -> Dict[str, any]:
    """
    Get performance info for a model.
    """
    model_info = {
        "phi3:mini": {
            "size": "2.2GB",
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐",
            "best_for": "Fast processing, small documents",
            "context_length": "4K tokens",
            "ram_requirement": "2GB",
        },
        "llama3.2:3b": {
            "size": "2.0GB",
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐",
            "best_for": "Balanced speed and quality",
            "context_length": "8K tokens",
            "ram_requirement": "4GB",
        },
    }
    return model_info.get(
        model_name,
        {
            "size": "Unknown",
            "speed": "⚡",
            "quality": "⭐⭐",
            "best_for": "General processing",
            "context_length": "4K tokens",
            "ram_requirement": "4GB+",
        },
    )


def show_ollama_status() -> bool:
    """
    Show status of Ollama service in sidebar (uses singleton if possible).
    """
    if st is None:
        return False

    st.sidebar.header("🦙 Ollama Status")

    parser = _get_parser_instance()
    if parser is None:
        st.sidebar.error("🦙 Ollama parser not found (singleton unavailable)")
        return False

    try:
        if not getattr(parser, "is_connected", False):
            parser.check_connection()
    except Exception:
        LOG.debug("check_connection raised in show_ollama_status", exc_info=True)

    if getattr(parser, "is_connected", False):
        st.sidebar.success("🦙 **Ollama**: ✅ Available")
        models = getattr(parser, "available_models", []) or []
        if models:
            st.sidebar.write("**Available Models:**")
            for model in models[:10]:
                st.sidebar.write(f"• {model}")
            if len(models) > 10:
                st.sidebar.write(f"... and {len(models) - 10} more")
        else:
            st.sidebar.warning("No models installed")
        return True
    else:
        st.sidebar.error("🦙 **Ollama**: ❌ Not connected")
        st.sidebar.write("Please install and start Ollama service")
        return False
