import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def safe_extract_text(obj: Any) -> str:
    """Return a safe string from arbitrary obj without raising."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # Common LLM shapes
    if isinstance(obj, dict):
        # Try nested text fields
        for key in [
            "response",
            "content",
            "text",
            "message",
            "choices",
        ]:
            if key in obj:
                return safe_extract_text(obj[key])
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        if not obj:
            return ""
        # OpenAI choices style
        if isinstance(obj[0], dict) and obj[0].get("message", {}).get("content"):
            return obj[0]["message"]["content"]
        return "\n".join(safe_extract_text(x) for x in obj)
    return str(obj)


def extract_json_block(text: str) -> Optional[str]:
    """Extract the outermost {...} JSON block from text, if present."""
    if not isinstance(text, str):
        text = safe_extract_text(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _as_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    s = s.strip()
    return s if s else None


def _as_list_of_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        seen = set()
        for x in v:
            s = _as_str_or_none(x)
            s = _clean_skill_candidate(s)
            if s:
                k = s.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(s)
        return out
    # allow comma separated strings
    s = _as_str_or_none(v)
    if not s:
        return []
    parts = [p.strip() for p in re.split(r",|\n|;|\|", s) if p.strip()]
    dedup: List[str] = []
    seen = set()
    for p in parts:
        p = _clean_skill_candidate(p)
        if not p:
            continue
        k = p.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    return dedup


def _coerce_experience_list(v: Any) -> List[Dict[str, Any]]:
    items = []
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, list):
        for item in v:
            if not isinstance(item, dict):
                item = {}
            title = _as_str_or_none(item.get("title") or item.get("role") or item.get("position"))
            company = _as_str_or_none(item.get("company_name") or item.get("company"))
            location = _as_str_or_none(item.get("location"))
            start_time = _as_str_or_none(item.get("start_time") or item.get("start") or item.get("startDate"))
            end_time = _as_str_or_none(item.get("end_time") or item.get("end") or item.get("endDate"))
            summary = _as_str_or_none(item.get("summary") or item.get("description") or item.get("details"))
            items.append({
                "title": title or "",
                "company_name": company or "",
                "location": location,
                "start_time": start_time,
                "end_time": end_time,
                "summary": summary,
            })
    return items


def _coerce_education_list(v: Any) -> List[Dict[str, Any]]:
    items = []
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, list):
        for item in v:
            if not isinstance(item, dict):
                item = {}
            institution = _as_str_or_none(item.get("institution") or item.get("school") or item.get("university")) or ""
            degree = _as_str_or_none(item.get("degree") or item.get("qualification"))
            year = _as_str_or_none(item.get("year") or item.get("graduation") or item.get("graduation_year"))
            location = _as_str_or_none(item.get("location"))
            gpa = _as_str_or_none(item.get("gpa") or item.get("percentage") or item.get("overall_percentage"))
            # Filter out noise entries
            if _is_valid_institution(institution):
                items.append({
                    "institution": institution,
                    "degree": degree,
                    "year": year,
                    "location": location,
                    "gpa": gpa,
                })
    return items


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_and_normalize(parsed_obj: Any, *, chunks: Optional[List[Dict[str, Any]]] = None, provider: Optional[str] = None, model: Optional[str] = None, parsing_method: str = "model", raw_text: Optional[str] = None) -> Dict[str, Any]:
    """Coerce arbitrary parsed object to the standard schema, with defaults."""
    warnings: List[str] = []
    if not isinstance(parsed_obj, dict):
        # Attempt JSON parse from text
        text = safe_extract_text(parsed_obj)
        block = extract_json_block(text)
        try:
            parsed_obj = json.loads(block or text)
        except Exception:
            parsed_obj = {}
            warnings.append("Model output not valid JSON; returned minimal defaults.")

    # Personal info extraction
    personal_info = {}
    if isinstance(parsed_obj, dict):
        p = parsed_obj.get("personal_info") or {}
        if isinstance(p, dict):
            personal_info = {
                "name": _as_str_or_none(p.get("name")),
                "email": _as_str_or_none(p.get("email")),
                "phone": _as_str_or_none(p.get("phone")),
                "location": _as_str_or_none(p.get("location")),
                "linkedin": _as_str_or_none(p.get("linkedin")),
                "github": _as_str_or_none(p.get("github")),
                "portfolio": _as_str_or_none(p.get("portfolio") or p.get("website")),
            }

    # Primary fields
    summary = _as_str_or_none(parsed_obj.get("summary") or parsed_obj.get("professional_summary") or parsed_obj.get("objective"))
    experience = _coerce_experience_list(parsed_obj.get("experience"))
    education = _coerce_education_list(parsed_obj.get("education"))

    # skills and technologies - extract ALL from structured dict
    skills_flat: List[str] = []
    technologies_flat: List[str] = []
    skills_obj = parsed_obj.get("skills")
    if isinstance(skills_obj, dict):
        # Merge all skills - extract from all categories
        for key, val in skills_obj.items():
            vals = _as_list_of_str(val)
            # Map to technologies or skills
            if key in ("tools_and_technologies", "tools", "frameworks", "frameworks & libraries", "frameworks_and_libraries", "frameworks_and_libraries", "markup_and_styling", "markup & styling", "database", "databases"):
                technologies_flat.extend(vals)
            elif key in ("programming_languages", "languages", "technical", "domains"):
                skills_flat.extend(vals)
            else:
                # Include everything else in skills too
                skills_flat.extend(vals)
    else:
        skills_flat = _as_list_of_str(skills_obj)

    # technologies top-level field if provided
    technologies_flat.extend(_as_list_of_str(parsed_obj.get("technologies")))

    # tags
    tags = _as_list_of_str(parsed_obj.get("tags") or parsed_obj.get("classification_tags") or parsed_obj.get("keywords_extracted"))

    # chunks normalization (optional)
    norm_chunks: List[Dict[str, Any]] = []
    if isinstance(chunks, list):
        for idx, ch in enumerate(chunks):
            if isinstance(ch, dict):
                norm_chunks.append({
                    "chunk_id": idx,
                    "text": _as_str_or_none(ch.get("content") or ch.get("text")) or "",
                    "summary": _as_str_or_none(ch.get("summary")),
                    "page": ch.get("page") if isinstance(ch.get("page"), int) else None,
                    "start_char": ch.get("start_char") if isinstance(ch.get("start_char"), int) else None,
                    "end_char": ch.get("end_char") if isinstance(ch.get("end_char"), int) else None,
                })

    normalized = {
        "summary": summary,
        "personal_info": personal_info,
        "experience": experience,
        "education": education,
        "skills": _as_list_of_str(skills_flat)[:200],
        "technologies": _as_list_of_str(technologies_flat)[:200],
        "tags": tags[:50],
        "chunks": norm_chunks,
        "provider": provider or "offline",
        "parsing_method": parsing_method,
        "parsing_time": _now_iso_z(),
        "_model": model or None,
        "_warnings": warnings,
    }

    return normalized


# ----- Helpers for cleaning -----

_SKILL_STOPWORDS = {"ma", "me", "none", "and", "with", "using", "features", "stack", "focus", "focusing", "integration", "full", "full-stack", "development", "developer", "pvt", "ltd", "solution", "solutions"}

_KNOWN_TECH = {
    # languages
    "javascript", "typescript", "python", "c++", "java", "c", "sql",
    # web
    "html", "css", "react", "react.js", "next.js", "node", "node.js", "express", "express.js",
    # db
    "mongodb", "mysql", "postgresql", "sql", "appwrite",
    # tools
    "git", "postman", "vs code", "vscode", "docker",
    # styling
    "tailwind css", "bootstrap",
    # devops
    "aws", "azure", "ci/cd", "kubernetes",
    # misc
    "spline"
}

def _is_valid_institution(name: str) -> bool:
    if not name:
        return False
    n = name.strip()
    if n.lower() in _SKILL_STOPWORDS:
        return False
    if n.lower() in {"na", "n/a", "null"}:
        return False
    # too short or mostly digits
    if len(n) < 3:
        return False
    if sum(ch.isalpha() for ch in n) < 3:
        return False
    # avoid pure years
    if re.fullmatch(r"\d{4}", n):
        return False
    # must contain common institution keywords to reduce noise
    nlow = n.lower()
    keywords = ("university", "college", "institute", "school", "academy", "iit", "iiit")
    # Reject common noise words that slip through (single word noise)
    noise_words = {"ma", "me", "attendance", "spline", "group", "messaging", "oriented", "stack", "features", "one-on-one", "on-one"}
    if len(n.split()) == 1 and nlow in noise_words:
        return False
    # Also reject if it's just "institute" or "institute" + short noise
    if nlow == "institute" or (len(n.split()) <= 2 and "institute" in nlow and nlow not in {"institute of", "institute of technology"}):
        if not any(k in nlow for k in {"technology", "management", "engineering", "science"}):
            return False
    if any(k in nlow for k in keywords):
        return True
    # Very strict: without keywords, only allow short location-like names
    if len(n.split()) > 3:
        return False
    # Single word must be a known location
    if len(n.split()) == 1 and nlow not in {"delhi", "mumbai", "bangalore", "hyderabad", "chennai", "pune", "kolkata"}:
        return False
    return True


def _clean_skill_candidate(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # remove common prefixes
    s = re.sub(r"(?i)^(programming languages?|languages?|tools?|frameworks?\s*&?\s*libraries?:?)\s*:?\s*", "", s)
    # discard sentences (too long with punctuation)
    if len(s.split()) > 5 or "," in s and len(s) > 40 or "." in s:
        return None
    # lowercase compare, but return original casing
    sl = s.lower()
    if sl in _SKILL_STOPWORDS:
        return None
    # company noise
    if "pvt ltd" in sl or "private limited" in sl:
        return None
    # allow known tech multi-words
    if sl in _KNOWN_TECH:
        return s
    # keep 1-3 word tokens without digits heavy noise
    if any(ch.isalpha() for ch in s) and len(s.split()) <= 3:
        return s
    return None


