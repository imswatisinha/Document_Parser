# parser/semantic_classifier.py
import math
import os
import tempfile
import csv
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st
import torch
from transformers import pipeline

# --- Config -----------------------------------------------------------------
# Smaller/faster NLI model (distilled)
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
# Tune this: number of labels to pass to the pipeline per forward pass
DEFAULT_LABEL_BATCH_SIZE = 12
# Number of words per text chunk (simple heuristic)
DEFAULT_CHUNK_WORDS = 120
# optional model cache dir (used only for messaging; HF uses its own cache)
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--" + MODEL_NAME.replace("/", "--"))

# CSV path for skill labels (adjust in your repo if needed)
SKILL_CSV_PATH = "./assets/skills.csv"

# Fallback default labels (used only if CSV missing/empty)
DEFAULT_CANDIDATE_LABELS = [
    "Python", "C++", "Machine Learning", "Deep Learning",
    "Cloud", "Azure", "AWS", "Communication",
    "Leadership", "Research", "Data Analysis",
    "Teamwork", "Web Development", "Frontend",
    "Backend", "DevOps"
]


# -----------------------
# Skill CSV loader
# -----------------------
@st.cache_data
def load_skill_labels_from_csv(path: str = SKILL_CSV_PATH) -> List[str]:
    """
    Load skill labels from CSV. Supports a header column named 'label' (case-insensitive)
    or plain single-column CSV. Deduplicates while preserving order.
    Returns fallback DEFAULT_CANDIDATE_LABELS when file missing or empty.
    """
    labels: List[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            # Try DictReader first (if CSV has header 'label')
            try:
                reader = csv.DictReader(f)
                if reader.fieldnames and any(h.lower() == "label" for h in reader.fieldnames):
                    f.seek(0)
                    for row in reader:
                        # accept either 'label' or the first value
                        v = row.get("label") or row.get("Label") or next(iter(row.values()), "")
                        if v:
                            labels.append(str(v).strip())
                    labels = [l for l in labels if l]
                    # dedupe and return if non-empty
                    if labels:
                        seen = set()
                        out = []
                        for l in labels:
                            key = l.lower()
                            if key not in seen:
                                seen.add(key)
                                out.append(l)
                        return out
            except Exception:
                # unknown header format — fall back
                f.seek(0)

            # Fall back to plain reader: first column per row (skip empty rows)
            f.seek(0)
            reader2 = csv.reader(f)
            for row in reader2:
                if not row:
                    continue
                val = str(row[0]).strip()
                if not val:
                    continue
                # skip single header 'label'
                if val.lower() == "label" and len(row) == 1:
                    continue
                labels.append(val)
    except FileNotFoundError:
        # file missing — fall back
        return DEFAULT_CANDIDATE_LABELS.copy()
    except Exception as e:
        try:
            st.warning(f"Failed to read skill CSV '{path}': {e}")
        except Exception:
            pass
        return DEFAULT_CANDIDATE_LABELS.copy()

    # Deduplicate preserve order
    seen = set()
    unique = []
    for l in labels:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            unique.append(l)

    if not unique:
        return DEFAULT_CANDIDATE_LABELS.copy()
    return unique


# --- Classifier loading (cached) -------------------------------------------
@st.cache_resource
def load_classifier(model_name: str = MODEL_NAME):
    """
    Load and cache a zero-shot classifier pipeline. Uses GPU if available.
    """
    device = 0 if torch.cuda.is_available() else -1
    # Inform user if model downloads are required on CPU
    if device == -1 and not os.path.exists(MODEL_CACHE_DIR):
        try:
            st.warning(f"{model_name} is loading for the first time. Please wait... (this may take a minute)")
        except Exception:
            pass
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    return classifier


# --- Utilities --------------------------------------------------------------
def chunk_list(lst: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def chunk_text_by_words(text: str, max_words: int = DEFAULT_CHUNK_WORDS) -> List[str]:
    """
    Very lightweight chunking: split text into chunks of roughly max_words words.
    Avoids heavy dependencies like nltk; works well for resumes/descriptions.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


# --- Optimized zero-shot with label batching -------------------------------
def zero_shot_skill_scores_optimized(
    text: str,
    candidate_labels: Optional[List[str]] = None,
    label_batch_size: int = DEFAULT_LABEL_BATCH_SIZE,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    model_name: str = MODEL_NAME,
    warn_if_large: bool = True,
) -> Dict[str, List[float]]:
    """
    Optimized zero-shot scoring:
      - candidate_labels optional: if None, loads labels from CSV via load_skill_labels_from_csv()
      - batches candidate labels to reduce repeated overhead
      - chunks long text and aggregates scores (max across chunks)
    Returns a dict with 'labels' (sorted desc) and 'scores'.
    """
    # Load labels from CSV if none provided
    if candidate_labels is None:
        candidate_labels = load_skill_labels_from_csv()

    if not text or not candidate_labels:
        return {
            "labels": candidate_labels or [],
            "scores": [0.0 for _ in (candidate_labels or [])],
        }

    # Warn if label set is large (optional)
    if warn_if_large and len(candidate_labels) > 200:
        try:
            st.warning(f"Large label set detected ({len(candidate_labels)} labels). This may take a while.")
        except Exception:
            pass

    classifier = load_classifier(model_name)

    # clamp batch size to reasonable bounds
    if label_batch_size is None:
        label_batch_size = DEFAULT_LABEL_BATCH_SIZE
    label_batch_size = max(4, min(label_batch_size, 64))

    # chunk the text to avoid very long inputs
    text_chunks = chunk_text_by_words(text, max_words=chunk_words)

    # maintain best score per label across chunks and batches
    best_scores = {label: 0.0 for label in candidate_labels}
    seen_counts = {label: 0 for label in candidate_labels}

    # For each chunk, run the classifier in batches of labels
    for chunk in text_chunks:
        for label_batch in chunk_list(candidate_labels, label_batch_size):
            try:
                res = classifier(chunk, label_batch, multi_label=True)
            except Exception as e:
                try:
                    st.warning(f"Classifier batch failed: {e}")
                except Exception:
                    pass
                continue

            labels_res = res.get("labels", [])
            scores_res = res.get("scores", [])
            for lab, sc in zip(labels_res, scores_res):
                if sc > best_scores.get(lab, 0.0):
                    best_scores[lab] = float(sc)
                seen_counts[lab] += 1

    # Build sorted results (desc by score), preserving all labels
    items = sorted(best_scores.items(), key=lambda x: -x[1])
    if items:
        labels_sorted, scores_sorted = zip(*items)
    else:
        labels_sorted, scores_sorted = ([], [])

    return {"labels": list(labels_sorted), "scores": [float(s) for s in scores_sorted]}


# Backwards-compatible wrapper (keeps your existing imports working)
def zero_shot_skill_scores(text: str, candidate_labels: Optional[list] = None):
    return zero_shot_skill_scores_optimized(text, candidate_labels)


# --- Radar chart -----------------------------------------------------------
def generate_radar_chart(labels: List[str], scores: List[float]) -> str:
    """
    Generate a radar/spider chart for the provided labels and scores.
    Returns the path to the generated PNG file.
    """
    if not labels or not scores or len(labels) != len(scores):
        raise ValueError("Labels and scores must be non-empty and of equal length.")

    # Normalize scores to 0-1 range for consistent plotting
    max_score = max(scores) if scores else 1.0
    normalized_scores = [score / max_score if max_score else 0.0 for score in scores]

    # Prepare angles and repeated first point for closed polygon
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values = normalized_scores + normalized_scores[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids([angle * 180 / math.pi for angle in angles[:-1]], labels)
    ax.plot(angles, values, linewidth=2, linestyle="solid")
    ax.fill(angles, values, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.grid(True)

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name, bbox_inches="tight")
        chart_path = tmp_file.name

    plt.close(fig)
    return chart_path
