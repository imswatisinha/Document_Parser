# utils/ollama_parser.py
import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Generator, AsyncGenerator
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter, Retry

# aiohttp is used for streaming; keep as optional import so sync-only envs still work
try:
    import aiohttp
    import asyncio
    AIOHTTP_AVAILABLE = True
except Exception:
    AIOHTTP_AVAILABLE = False

LOG = logging.getLogger(__name__)
CACHE_DIR = os.path.join(".cache", "ollama")
os.makedirs(CACHE_DIR, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simple_file_cache_get(key: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(p):
        try:
            return json.loads(open(p, "r", encoding="utf-8").read())
        except Exception:
            return None
    return None


def simple_file_cache_set(key: str, value: Dict[str, Any]) -> None:
    p = os.path.join(CACHE_DIR, key + ".json")
    try:
        open(p, "w", encoding="utf-8").write(json.dumps(value, ensure_ascii=False, indent=2))
    except Exception:
        LOG.exception("Failed to write cache")


class OllamaParser:
    """Optimized Ollama integration. No Streamlit UI calls inside this class."""

    SUPPORTED_MODELS = ["phi3:mini", "llama3.2:3b"]

    def __init__(self, base_url: str = "http://localhost:11434", session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/") if base_url else "http://localhost:11434"
        self.available_models: List[str] = []
        self.is_connected: bool = False
        self.selected_model: Optional[str] = None

        # requests session with retries
        if session is None:
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5,
                            status_forcelist=[429, 500, 502, 503, 504])
            session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session = session

        # Lazy-check connection (don't block __init__ heavily)
        try:
            self.check_connection(timeout=3)
        except Exception:
            # swallow; user can call check_connection explicitly
            LOG.debug("Initial Ollama check failed; call check_connection() later.")

    def check_connection(self, timeout: int = 5) -> bool:
        """Check if Ollama is running and read available models."""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=timeout)
            if resp.status_code == 200:
                models_data = resp.json()
                all_models = [m.get("name") for m in models_data.get("models", []) if "name" in m]
                # keep only supported models otherwise include available
                self.available_models = [m for m in all_models if m in self.SUPPORTED_MODELS] or all_models
                self.is_connected = True
                # default selection
                self.selected_model = self.available_models[0] if self.available_models else None
                return True
            else:
                self.is_connected = False
                return False
        except Exception:
            self.is_connected = False
            LOG.exception("Ollama connection check failed")
            return False

    # -------------------
    # Streaming (aiohttp) - yields raw text chunks as they arrive
    # -------------------
    async def stream_generate(self, prompt: str, model: str, timeout: int = 300) -> AsyncGenerator[str, None]:
        """Async generator streaming raw bytes/lines from Ollama. Requires aiohttp."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available. Install aiohttp to use streaming.")
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1}
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Ollama stream returned status {resp.status}: {text}")
                    async for raw in resp.content:
                        if not raw:
                            continue
                        try:
                            chunk = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            chunk = str(raw)
                        if chunk:
                            yield chunk
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Streaming generate failed")
            raise

    # -------------------
    # Sync helper (requests) - returns whole response JSON
    # -------------------
    def sync_generate(self, prompt: str, model: str, timeout: int = 120) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }
        try:
            resp = self.session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            LOG.exception("sync_generate failed")
            raise

    # -------------------
    # High-level parse API
    # -------------------
    async def parse_resume_stream(self, resume_text: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Async streaming parser. Yields chunks of raw model output (strings).
        The caller (UI) should accumulate chunks and attempt to parse JSON when complete.
        """
        if not self.is_connected:
            # try a quick connection refresh
            self.check_connection()
            if not self.is_connected:
                raise RuntimeError("Ollama not connected or no models available")

        if model is None:
            model = self.selected_model

        if model is None:
            raise ValueError("No model specified or available")

        # cache short-circuit
        key = sha256_text(model + "|" + resume_text[:100000])  # limit size in key
        cached = simple_file_cache_get(key)
        if cached:
            # Return cached parsed JSON as single chunk (stringified)
            yield json.dumps(cached)
            return

        # build prompt (keep separate helper if you want)
        prompt = self._build_resume_prompt(resume_text)

        # stream from Ollama
        async for chunk in self.stream_generate(prompt, model):
            yield chunk

        # after stream ends, caller should parse + optionally cache result

    def parse_resume_sync(self, resume_text: str, model: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
        """
        Synchronous parse (blocking) - returns parsed JSON dict or error dict.
        Uses cache if available.
        """
        if not self.is_connected:
            self.check_connection()
            if not self.is_connected:
                return {"error": "Ollama not connected", "_source": "ollama"}

        if model is None:
            model = self.selected_model

        if model is None:
            return {"error": "No model available", "_source": "ollama"}

        key = sha256_text(model + "|" + resume_text[:100000])
        cached = simple_file_cache_get(key)
        if cached:
            cached["_cached"] = True
            return cached

        prompt = self._build_resume_prompt(resume_text)
        try:
            result = self.sync_generate(prompt, model, timeout=timeout)
            response_text = result.get("response", "")
            # extract JSON safely
            parsed = self._extract_json_from_text(response_text)
            if isinstance(parsed, dict):
                parsed["_model_used"] = model
                parsed["_source"] = "ollama"
                parsed["_timestamp"] = datetime.utcnow().isoformat()
                simple_file_cache_set(key, parsed)
                return parsed
            else:
                return {"error": "Could not extract JSON from model response", "raw_response": response_text[:1000], "_source": "ollama"}
        except Exception as e:
            return {"error": f"Generate failed: {str(e)}", "_source": "ollama"}

    # -------------------
    # Utilities
    # -------------------
    def _build_resume_prompt(self, resume_text: str) -> str:
        # Keep your original detailed schema here, but keep prompt length reasonable.
        header = (
            "You are an expert resume parser. Extract the requested fields and "
            "return only valid JSON matching the schema. Dates should be MMM YYYY or 'present'.\n\n"
        )
        schema = """{
  "personal_info": {"name": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": ""},
  "summary": null,
  "experience": [{"title":"","company_name":"","location":"","start_time":"","end_time":"","summary":""}],
  "education": [{"institution":"","degree":"","year":"","location":"","gpa":""}],
  "skills": {"technical": [], "soft": [], "programming_languages": [], "tools_and_technologies": [], "domains": []},
  "projects":[{"name":"","description":"","technologies":[],"url":"","duration":""}],
  "certifications": [{"name":"","issuer":"","date":"","url":""}],
  "achievements": [],
  "languages": [],
  "keywords_extracted": [],
  "classification_tags": []
}"""
        # Keep resume text trimmed for prompt if extremely long; for large docs prefer chunking + RAG
        trimmed = resume_text if len(resume_text) < 20000 else resume_text[:20000] + "\n\n[TRUNCATED]"
        prompt = f"{header}\nSchema:\n{schema}\n\nResume text:\n{trimmed}\n\nReturn ONLY the JSON object with no extra commentary."
        return prompt

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        try:
            si = text.find("{")
            ei = text.rfind("}") + 1
            if si == -1 or ei <= si:
                return None
            json_text = text[si:ei]
            return json.loads(json_text)
        except Exception:
            LOG.exception("Failed to extract JSON")
            return None

    # -------------------
    # Ask question helper (fixes undefined attributes)
    # -------------------
    def ask_question(self, question: str, context: str = "", model: Optional[str] = None, max_tokens: int = 500) -> Dict[str, Any]:
        if model is None:
            model = self.selected_model
        if not model:
            return {"error": "No model selected", "success": False}

        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely based on the context."
        try:
            result = self.sync_generate(prompt, model, timeout=180)
            answer = result.get("response", "").strip()
            return {"answer": answer, "model_used": model, "success": True}
        except Exception as e:
            LOG.exception("ask_question failed")
            return {"error": str(e), "success": False}


# --- Compatibility / Streamlit integration helpers ---
# NOTE: these are thin UI wrappers so the Streamlit app can keep calling them.
# They intentionally use streamlit (st) because your app expects that behavior.

try:
    import streamlit as st
except Exception:
    st = None  # If streamlit isn't available, these functions will fail later.


def _try_get_singleton_from_factory(singleton_base_url: str = None, warm_up: bool = False) -> Optional[OllamaParser]:
    """
    Try to import and call utils.ollama_singleton.get_ollama lazily.
    Returns an OllamaParser instance or None.
    """
    try:
        mod = __import__("utils.ollama_singleton", fromlist=["get_ollama"])
        get_ollama = getattr(mod, "get_ollama", None)
        if callable(get_ollama):
            return get_ollama(singleton_base_url, warm_up)
    except Exception:
        LOG.debug("Could not load utils.ollama_singleton.get_ollama (will fallback).", exc_info=True)
    return None


def integrate_ollama_parser(singleton_base_url: str = None, warm_up: bool = False) -> bool:
    """
    Create and store an OllamaParser instance in streamlit session state.
    Prefer using the singleton factory (utils.ollama_singleton.get_ollama) if available.
    Returns True if parser is available/connected (has models).
    """
    if st is None:
        raise RuntimeError("Streamlit not available in this environment.")
    # If already created, return readiness
    if st.session_state.get("ollama_parser"):
        parser = st.session_state["ollama_parser"]
        return bool(getattr(parser, "is_connected", False) and getattr(parser, "available_models", []))

    # Try singleton factory first (lazy import to avoid circular import at module load)
    parser = _try_get_singleton_from_factory(singleton_base_url=singleton_base_url, warm_up=warm_up)
    if parser is None:
        # Fall back to creating a local OllamaParser instance
        try:
            parser = OllamaParser(base_url=singleton_base_url) if singleton_base_url else OllamaParser()
        except Exception:
            parser = OllamaParser()

    # ensure connection status is refreshed
    try:
        parser.check_connection()
    except Exception:
        # check_connection already logs; continue
        pass

    st.session_state["ollama_parser"] = parser
    st.session_state["ollama_model"] = parser.selected_model if getattr(parser, "selected_model", None) else None
    return bool(getattr(parser, "is_connected", False) and getattr(parser, "available_models", []))


def get_available_ollama_models() -> list:
    """Return a list of available models from the session parser (or empty list)."""
    if st is None:
        return []
    parser = st.session_state.get("ollama_parser")
    if not parser:
        # Attempt to fetch singleton (no warm-up)
        parser = _try_get_singleton_from_factory()
        if parser:
            st.session_state["ollama_parser"] = parser
    if not parser:
        return []
    return getattr(parser, "available_models", [])


def show_ollama_status():
    """Small UI helper: display current Ollama connection / model info in Streamlit."""
    if st is None:
        return
    parser = st.session_state.get("ollama_parser")
    if not parser:
        parser = _try_get_singleton_from_factory()
        if parser:
            st.session_state["ollama_parser"] = parser

    if not parser:
        st.info("Ollama parser not initialized.")
        return

    if getattr(parser, "is_connected", False):
        st.success(f"✅ Ollama connected. Models: {', '.join(parser.available_models) if parser.available_models else 'None'}")
        st.write(f"Selected model: {parser.selected_model}")
    else:
        st.error("❌ Ollama not connected or no models available.")


def show_model_recommendations(models: list):
    """Show simple recommendations / install hint in the sidebar or any Streamlit area."""
    if st is None:
        return
    if not models:
        st.warning("No recommended models found. Consider installing phi3:mini for low-memory parsing.")
        return
    st.info("Recommended models (based on availability):")
    for m in models:
        st.write(f"• {m}")
