import math
import os
import tempfile
from typing import Dict, List

import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--bart-large-mnli")


@st.cache_resource
def load_classifier():
    """
    Lazily load the zero-shot classifier and cache the resource for reuse.
    Shows a warning when the model needs to be downloaded for the first time.
    """
    if not os.path.exists(MODEL_CACHE_DIR):
        st.warning("BART MNLI model is loading for the first time. Please wait...")
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def zero_shot_skill_scores(text: str, candidate_labels: List[str]) -> Dict[str, List[float]]:
    """
    Run zero-shot classification on the provided text to score candidate skill labels.
    """
    if not text or not candidate_labels:
        return {"labels": candidate_labels or [], "scores": [0.0 for _ in candidate_labels] if candidate_labels else []}

    classifier = load_classifier()
    result = classifier(text, candidate_labels, multi_label=True)
    labels = result.get("labels", [])
    scores = [float(score) for score in result.get("scores", [])]
    return {"labels": labels, "scores": scores}


def generate_radar_chart(labels: List[str], scores: List[float]) -> str:
    """
    Generate a radar/spider chart for the provided labels and scores.
    Returns the path to the generated PNG file.
    """
    if not labels or not scores or len(labels) != len(scores):
        raise ValueError("Labels and scores must be non-empty and of equal length.")

    # Normalize scores to 0-1 range for consistent plotting
    max_score = max(scores) if scores else 1
    normalized_scores = [score / max_score if max_score else 0 for score in scores]

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values = normalized_scores + normalized_scores[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids([angle * 180 / math.pi for angle in angles[:-1]], labels)
    ax.plot(angles, values, linewidth=2, linestyle="solid", color="#1f77b4")
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)
    ax.set_ylim(0, 1)
    ax.grid(True)

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name, bbox_inches="tight")
        chart_path = tmp_file.name

    plt.close(fig)
    return chart_path

