# utils/ollama_singleton.py
from typing import Optional
import threading
from utils.ollama_parser import OllamaParser

_lock = threading.Lock()
_instance: Optional[OllamaParser] = None

def get_ollama(singleton_base_url: str = None, warm_up: bool = False) -> OllamaParser:
    """
    Return the global OllamaParser instance (one per process).
    - singleton_base_url: optional override for base_url on first creation
    - warm_up: if True, run a small warm-up generate (non-blocking best-effort)
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                if singleton_base_url:
                    _instance = OllamaParser(base_url=singleton_base_url)
                else:
                    _instance = OllamaParser()
                # Try to check connection on creation
                try:
                    _instance.check_connection()
                except Exception:
                    pass
                if warm_up:
                    # best-effort warm up (non-fatal)
                    try:
                        # a tiny prompt to warm model RAM/cache
                        _instance.parse_resume_sync("Warm up.", model=_instance.available_models[0] if _instance.available_models else None)
                    except Exception:
                        # swallow warm-up errors
                        pass
    return _instance
