import os
import json
import requests
from typing import Any, Dict, Optional

# Configuration - fallback providers are optional
ENABLE_FALLBACK_PROVIDERS = os.getenv("ENABLE_FALLBACK_PROVIDERS", "false").lower() == "true"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if ENABLE_FALLBACK_PROVIDERS else None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) if ENABLE_FALLBACK_PROVIDERS else None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _build_min_schema_prompt(resume_text: str) -> str:
    return (
        "You are an expert resume parser. Return ONLY valid JSON matching this schema: \n"
        "{\n"
        "  \"summary\": \"Short candidate summary here\",\n"
        "  \"experience\": [{\n"
        "    \"title\": \"\", \"company_name\": \"\", \"location\": \"\", \"start_time\": \"\", \"end_time\": \"\", \"summary\": \"\"\n"
        "  }],\n"
        "  \"education\": [{\n"
        "    \"institution\": \"\", \"degree\": \"\", \"year\": \"\", \"location\": \"\", \"gpa\": \"\"\n"
        "  }],\n"
        "  \"skills\": [\"\"],\n"
        "  \"tags\": [\"Technology\", \"HR\"]\n"
        "}\n"
        "Rules: Do not include any text before or after JSON. Use arrays even if empty. Prefer date format 'MMM YYYY' or 'Present'.\n\n"
        f"Resume:\n{resume_text}"
    )


def call_ollama_min_schema(text: str, model: Optional[str]) -> Dict[str, Any]:
    body = {
        "model": model or "llama3.2:3b",
        "prompt": _build_min_schema_prompt(text),
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
    if resp.status_code != 200:
        return {"error": f"Ollama HTTP {resp.status_code}", "message": resp.text}
    data = resp.json()
    content = data.get("response", "")
    i, j = content.find("{"), content.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(content[i : j + 1])
        except Exception as e:
            return {"error": f"Ollama JSON decode error: {e}", "raw": content[:500]}
    return {"error": "Ollama did not return JSON", "raw": content[:500]}


def call_openai_min_schema(text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set"}
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You output only strict JSON, no prose."},
        {"role": "user", "content": _build_min_schema_prompt(text)},
    ]
    body = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.1}
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        return {"error": f"OpenAI HTTP {resp.status_code}", "message": resp.text}
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    i, j = content.find("{"), content.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(content[i : j + 1])
        except Exception as e:
            return {"error": f"OpenAI JSON decode error: {e}", "raw": content[:500]}
    return {"error": "OpenAI did not return JSON", "raw": content[:500]}


def call_gemini_min_schema(text: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": "You output only strict JSON, no prose."},
                    {"text": _build_min_schema_prompt(text)},
                ]
            }
        ]
    }
    resp = requests.post(url, json=body, timeout=120)
    if resp.status_code != 200:
        return {"error": f"Gemini HTTP {resp.status_code}", "message": resp.text}
    data = resp.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return {"error": "Gemini malformed response", "raw": json.dumps(data)[:500]}
    i, j = content.find("{"), content.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(content[i : j + 1])
        except Exception as e:
            return {"error": f"Gemini JSON decode error: {e}", "raw": content[:500]}
    return {"error": "Gemini did not return JSON", "raw": content[:500]}


def parse_with_fallback_min_schema(text: str, preferred_ollama_model: Optional[str]) -> Dict[str, Any]:
    """
    Parse resume with fallback chain: Ollama → OpenAI → Gemini (if enabled).
    
    Set ENABLE_FALLBACK_PROVIDERS=true in .env to enable OpenAI/Gemini fallbacks.
    """
    # 1) Try Ollama first (always primary)
    try:
        res = call_ollama_min_schema(text, preferred_ollama_model)
        if "error" not in res:
            res["_provider"] = "ollama"
            res["_model"] = preferred_ollama_model or "llama3.2:3b"
            return res
    except Exception as e:
        res = {"error": f"Ollama exception: {e}"}

    # 2) Fallback to OpenAI (only if enabled)
    if ENABLE_FALLBACK_PROVIDERS and OPENAI_API_KEY:
        try:
            oa = call_openai_min_schema(text)
            if "error" not in oa:
                oa["_provider"] = "openai"
                oa["_model"] = OPENAI_MODEL
                return oa
        except Exception as e:
            pass

    # 3) Fallback to Gemini (only if enabled)
    if ENABLE_FALLBACK_PROVIDERS and GEMINI_API_KEY:
        gm = call_gemini_min_schema(text)
        if "error" not in gm:
            gm["_provider"] = "gemini"
            gm["_model"] = GEMINI_MODEL
            return gm
        return gm

    # Return Ollama error if no fallbacks available
    return res if "error" in res else {"error": "Unknown error"}


