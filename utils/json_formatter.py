import json
from typing import Any, Dict, List


TARGET_SCHEMA_EXPERIENCE_KEYS = [
    "title",
    "company_name",
    "location",
    "start_time",
    "end_time",
    "summary",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_experience(item: Any) -> Dict[str, Any]:
    result = {k: "" for k in TARGET_SCHEMA_EXPERIENCE_KEYS}
    if not isinstance(item, dict):
        return result
    # map potential alternative keys
    mapping = {
        "company": "company_name",
        "companyName": "company_name",
        "role": "title",
        "position": "title",
        "start": "start_time",
        "startDate": "start_time",
        "end": "end_time",
        "endDate": "end_time",
        "description": "summary",
        "details": "summary",
    }
    for key, value in item.items():
        if key in TARGET_SCHEMA_EXPERIENCE_KEYS:
            result[key] = _safe_str(value)
        elif key in mapping:
            result[mapping[key]] = _safe_str(value)
    return result


def _coerce_education(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "institution": "",
            "degree": "",
            "year": "",
            "location": "",
            "gpa": "",
        }
    out = {
        "institution": _safe_str(item.get("institution") or item.get("school") or item.get("university")),
        "degree": _safe_str(item.get("degree") or item.get("qualification")),
        "year": _safe_str(item.get("year") or item.get("graduation") or item.get("graduation_year")),
        "location": _safe_str(item.get("location")),
        "gpa": _safe_str(item.get("gpa")),
    }
    return out


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for v in value:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    items.append(s)
            else:
                s = _safe_str(v).strip()
                if s:
                    items.append(s)
        # dedupe, preserve order
        seen = set()
        deduped = []
        for s in items:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                deduped.append(s)
        return deduped
    # comma-delimited fallback
    return [s.strip() for s in _safe_str(value).split(",") if s.strip()]


def coerce_to_min_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce any LLM output into the minimal target schema required by the app."""
    parsed = parsed or {}

    # summary
    summary = parsed.get("summary")
    if not isinstance(summary, str):
        # try to pull from alternate places
        summary = parsed.get("professional_summary") or parsed.get("objective") or ""
    summary = _safe_str(summary)

    # experience
    exp_raw = parsed.get("experience")
    if isinstance(exp_raw, dict):
        # some models return keys per job
        exp_list = list(exp_raw.values())
    elif isinstance(exp_raw, list):
        exp_list = exp_raw
    else:
        exp_list = []
    experience = [_coerce_experience(item) for item in exp_list][:25]

    # education
    edu_raw = parsed.get("education")
    if isinstance(edu_raw, dict):
        edu_list = list(edu_raw.values())
    elif isinstance(edu_raw, list):
        edu_list = edu_raw
    else:
        edu_list = []
    education = [_coerce_education(item) for item in edu_list][:25]

    # skills: support both list and structured dict
    skills_field = parsed.get("skills")
    skills: List[str] = []
    if isinstance(skills_field, dict):
        for v in skills_field.values():
            skills.extend(_coerce_list(v))
    else:
        skills = _coerce_list(skills_field)
    skills = skills[:100]

    # tags: prefer classification_tags, fallback to keywords_extracted
    tags = _coerce_list(parsed.get("classification_tags") or parsed.get("tags") or parsed.get("keywords_extracted"))
    tags = tags[:30]

    return {
        "summary": summary,
        "experience": experience,
        "education": education,
        "skills": skills,
        "tags": tags,
    }


def safe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


