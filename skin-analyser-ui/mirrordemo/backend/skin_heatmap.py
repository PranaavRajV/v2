"""
skin_heatmap.py — Real-time Facial Landmark Overlay & HUD

Professional Clinical Palette:
- Acne: Crimson Red (0, 30, 255)
- Wrinkles: Cyan-Blue (255, 190, 0)
- Pigmentation: Amber-Orange (0, 140, 255)
- Dark Circles: Magenta-Violet (255, 0, 200)
- Pores & Texture: Emerald-Green (0, 255, 80)
"""

import cv2
import numpy as np

# Distinct Clinical Palette (BGR)
CONDITION_COLORS = {
    "acne": {
        "primary": (0, 30, 255),       # Crimson Red
        "glow": (80, 80, 255),
        "highlight": (200, 200, 255),
        "label": "Acne",
    },
    "wrinkles": {
        "primary": (255, 190, 0),      # Cyan-Blue
        "glow": (255, 230, 120),
        "highlight": (255, 255, 220),
        "label": "Wrinkles",
    },
    "pigmentation": {
        "primary": (0, 140, 255),      # Amber-Orange
        "glow": (80, 180, 255),
        "highlight": (220, 240, 255),
        "label": "Pigmentation",
    },
    "dark_circles": {
        "primary": (255, 0, 200),      # Magenta-Violet
        "glow": (255, 120, 230),
        "highlight": (255, 220, 255),
        "label": "Dark Circles",
    },
    "pores": {
        "primary": (0, 255, 80),       # Emerald-Green
        "glow": (100, 255, 150),
        "highlight": (220, 255, 230),
        "label": "Pores",
    },
}


def render_face_overlay(
    frame,
    transposed_points,
    scores,
    score_countdown=0,
    llm_countdown=0,
    has_real_inference=False
):
    """
    Renders live tracked condition points onto natural face frame with clean HUD.
    Each condition dot has its own distinct, highly differentiable color.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # --- 1. Draw Transposed Landmark Points (Locked to Face Mesh) ---
    if transposed_points:
        for condition, pts in transposed_points.items():
            spec = CONDITION_COLORS.get(condition, {
                "primary": (0, 255, 0),
                "glow": (100, 255, 100),
                "highlight": (255, 255, 255),
            })
            primary = spec["primary"]
            glow = spec["glow"]
            highlight = spec["highlight"]

            score_val = scores.get(condition, 25.0)
            radius = int(np.clip(5 + (score_val / 100.0) * 4, 5, 9))

            for (px, py) in pts:
                # Outer glow ring
                cv2.circle(output, (px, py), radius + 4, glow, 1, cv2.LINE_AA)
                # Primary condition ring
                cv2.circle(output, (px, py), radius + 1, primary, 1, cv2.LINE_AA)
                # Solid primary color filled dot
                cv2.circle(output, (px, py), radius, primary, -1, cv2.LINE_AA)
                # Center highlight reflection
                cv2.circle(output, (px, py), max(1, radius // 2), highlight, -1, cv2.LINE_AA)

    # --- 2. Minimalist Semi-Transparent HUD (Top Left) ---
    hud_w, hud_h = 240, 160
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
    cv2.rectangle(output, (12, 12), (12 + hud_w, 12 + hud_h), (75, 75, 85), 1)

    # Title
    cv2.putText(output, "Live Diagnostic", (22, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

    # Metrics with condition matching colors
    items = [
        (f"Acne:               {scores.get('acne', 0):.1f}%", CONDITION_COLORS["acne"]["primary"]),
        (f"Pigmentation:       {scores.get('pigmentation', 0):.1f}%", CONDITION_COLORS["pigmentation"]["primary"]),
        (f"Dark Circles:       {scores.get('dark_circles', 0):.1f}%", CONDITION_COLORS["dark_circles"]["primary"]),
        (f"Wrinkles:           {scores.get('wrinkles', 0):.1f}%", CONDITION_COLORS["wrinkles"]["primary"]),
        (f"Pores / Texture:    {scores.get('texture', 0):.1f}%", CONDITION_COLORS["pores"]["primary"]),
        (f"Skin Health:        {scores.get('skin_health', 0):.1f}%", (140, 255, 140)),
    ]
    y_start = 52
    for line, color in items:
        cv2.putText(output, line, (22, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y_start += 17

    # --- 3. Status Bar (Bottom Left) ---
    time_box_w, time_box_h = 240, 44
    by = h - time_box_h - 14
    overlay2 = output.copy()
    cv2.rectangle(overlay2, (12, by), (12 + time_box_w, by + time_box_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay2, 0.75, output, 0.25, 0, output)
    cv2.rectangle(output, (12, by), (12 + time_box_w, by + time_box_h), (60, 60, 70), 1)

    if not has_real_inference:
        score_msg = "Analysis in progress..."
        llm_msg = "Suggestions pending initial scan..."
    else:
        score_msg = f"Score Update: {max(0, int(score_countdown))}s"
        llm_msg = f"Suggestions Update: {max(0, int(llm_countdown))}s"

    cv2.putText(output, score_msg, (20, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(output, llm_msg, (20, by + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 255, 200), 1, cv2.LINE_AA)

    # --- 4. Distinct Legend (Top Right) ---
    leg_w, leg_h = 150, 115
    lx = w - leg_w - 12
    overlay3 = output.copy()
    cv2.rectangle(overlay3, (lx, 12), (w - 12, 12 + leg_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay3, 0.75, output, 0.25, 0, output)
    cv2.rectangle(output, (lx, 12), (w - 12, 12 + leg_h), (75, 75, 85), 1)

    cv2.putText(output, "Landmarks", (lx + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    ly = 44
    for cond_key, spec in CONDITION_COLORS.items():
        color = spec["primary"]
        label = spec["label"]
        cv2.circle(output, (lx + 16, ly), 5, spec["glow"], 1, cv2.LINE_AA)
        cv2.circle(output, (lx + 16, ly), 4, color, -1, cv2.LINE_AA)
        cv2.putText(output, label, (lx + 28, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
        ly += 14

    return output


# Backward compatibility shim
def draw_skin_heatmap(frame, coords, colors=None):
    return frame
