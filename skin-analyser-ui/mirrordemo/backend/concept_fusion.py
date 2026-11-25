import numpy as np


def compute_concepts(low_level, clip_features):
    """
    Combines DermaCLIP semantic features + optional OpenCV spatial features
    into structured dermatology concepts.
    DermaCLIP model features are the genuine ground truth.
    """
    if not clip_features:
        return {}

    low_level = low_level or {}

    # ----------------------------
    # LOW-LEVEL FEATURES
    # ----------------------------
    redness = low_level.get("redness", 0)
    texture = low_level.get("texture_roughness", 0)
    edges = low_level.get("edge_density", 0)
    brightness_var = low_level.get("brightness_variation", 0)  # relative pigmentation

    # ----------------------------
    # DERMACLIP FEATURES
    # ----------------------------
    semantic_acne = clip_features.get("semantic_acne", 0)
    semantic_pigment = clip_features.get("semantic_pigmentation", 0)
    semantic_texture = clip_features.get("semantic_texture", 0)
    semantic_redness = clip_features.get("semantic_redness", 0)
    semantic_healthy = clip_features.get("semantic_healthy", 0)
    semantic_wrinkles = clip_features.get("semantic_wrinkles", 0)
    semantic_dark = clip_features.get("semantic_dark_circles", 0)

    # ----------------------------
    # DERIVED CONCEPTS
    # ----------------------------

    # 🔥 ACNE
    bump_density = np.clip(
        edges * (0.5 * texture + 0.5 * redness) * (0.5 + 0.5 * semantic_acne),
        0, 1
    )

    # Wrinkles (enhanced with DermaCLIP wrinkles)
    base_wrinkle = edges * (1 - texture)
    if semantic_wrinkles > 0:
        wrinkle_depth = np.clip(0.5 * base_wrinkle + 0.5 * semantic_wrinkles, 0, 1)
    else:
        wrinkle_depth = np.clip(base_wrinkle, 0, 1)

    # 🔥 PIGMENTATION (TONE-AWARE FIX)
    pigmentation_spread = np.clip(
        0.7 * brightness_var +
        0.3 * semantic_pigment -
        0.2 * semantic_healthy,
        0, 1
    )

    # Inflammation
    inflammation = np.clip(
        0.6 * redness +
        0.4 * semantic_redness,
        0, 1
    )

    # 🔥 DARK CIRCLES (enhanced with DermaCLIP dark circles)
    base_dark = (1 - brightness_var) * 0.6 + semantic_texture * 0.4
    if semantic_dark > 0:
        dark_circles = np.clip(0.5 * base_dark + 0.5 * semantic_dark, 0, 1)
    else:
        dark_circles = np.clip(base_dark, 0, 1)

    return {
        # raw
        "redness": redness,
        "texture": texture,
        "edges": edges,
        "brightness_var": brightness_var,

        # semantic
        "semantic_acne": semantic_acne,
        "semantic_pigment": semantic_pigment,
        "semantic_texture": semantic_texture,
        "semantic_healthy": semantic_healthy,
        "semantic_wrinkles": semantic_wrinkles,
        "semantic_dark_circles": semantic_dark,

        # derived (final concepts)
        "bump_density": bump_density,
        "wrinkle_depth": wrinkle_depth,
        "pigmentation_spread": pigmentation_spread,
        "inflammation": inflammation,
        "dark_circles": dark_circles
    }
