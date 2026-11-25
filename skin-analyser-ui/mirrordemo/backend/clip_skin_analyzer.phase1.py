"""
Skin Analyzer interface using fine-tuned DermaCLIP.
Replaces generic OpenCLIP / CLIP with DermaCLIP (fine-tuned on dermatology facial conditions).
"""
import numpy as np
from dermaclip_analyzer import DermaCLIPAnalyzer

# Lazy-load or initialize singleton
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = DermaCLIPAnalyzer.get_instance()
    return _analyzer

def analyze_skin_clip(face):
    """
    Analyzes skin region/crop using DermaCLIP.
    Returns dictionary with semantic scores required by concept_fusion.py:
      - semantic_acne
      - semantic_pigmentation
      - semantic_texture
      - semantic_redness
      - semantic_healthy
    along with full DermaCLIP class probabilities and predicted condition.
    """
    if face is None or not hasattr(face, "size") or face.size == 0:
        return {
            "semantic_acne": 0.0,
            "semantic_pigmentation": 0.0,
            "semantic_texture": 0.0,
            "semantic_redness": 0.0,
            "semantic_healthy": 1.0,
            "Acne_DL": 0.0
        }

    analyzer = get_analyzer()
    result = analyzer.analyze(face)

    semantics = result.get("semantic_features", {})
    probs = result.get("probabilities", {})

    return {
        "semantic_acne": semantics.get("semantic_acne", 0.0),
        "semantic_pigmentation": semantics.get("semantic_pigmentation", 0.0),
        "semantic_texture": semantics.get("semantic_texture", 0.0),
        "semantic_redness": semantics.get("semantic_redness", 0.0),
        "semantic_healthy": semantics.get("semantic_healthy", 1.0),
        "semantic_wrinkles": semantics.get("semantic_wrinkles", 0.0),
        "semantic_dark_circles": semantics.get("semantic_dark_circles", 0.0),
        "Acne_DL": probs.get("Acne", 0.0),
        "probabilities": probs,
        "predicted_condition": result.get("predicted_class", "Normal"),
        "confidence": result.get("confidence")
    }
