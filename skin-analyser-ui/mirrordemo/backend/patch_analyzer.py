import cv2
import numpy as np


# ----------------------------
# NORMALIZATION FUNCTION
# ----------------------------
def normalize(val, min_val, max_val):
    return np.clip((val - min_val) / (max_val - min_val + 1e-6), 0, 1)


# ----------------------------
# MAIN ANALYZER
# ----------------------------
def analyze_skin_patches(patch, frame=None, box=None):

    if patch is None or patch.size == 0:
        return {}

    # ----------------------------
    # PREPROCESSING (more stable)
    # ----------------------------
    patch = cv2.resize(patch, (128, 128))

    # Mild normalization (reduce harsh lighting impact)
    patch = cv2.convertScaleAbs(patch, alpha=1.1, beta=-5)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Slight blur to reduce camera noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = cv2.equalizeHist(gray)

    # ----------------------------
    # REDNESS (LAB space)
    # ----------------------------
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)
    redness_raw = np.mean(a_channel)
    redness = normalize(redness_raw, 120, 160)

    # ----------------------------
    # TEXTURE ROUGHNESS (fixed scaling)
    # ----------------------------
    texture_raw = np.std(gray)
    texture_roughness = normalize(texture_raw, 20, 60)

    # ----------------------------
    # EDGE DENSITY (structure)
    # ----------------------------
    edges = cv2.Canny(gray, 60, 160)
    edge_density_raw = np.sum(edges > 0) / (128 * 128)
    edge_density = normalize(edge_density_raw, 0.02, 0.25)

    # ----------------------------
    # RELATIVE + LOCALIZED PIGMENTATION (FINAL 🔥)
    # ----------------------------
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    base_tone = np.mean(v)

    dark_diff = base_tone - v
    dark_diff[dark_diff < 0] = 0

    # Slight smoothing to remove noise
    dark_diff = cv2.GaussianBlur(dark_diff, (5, 5), 0)

    # Threshold meaningful darker areas
    mask = dark_diff > 15

    # Remove smooth lighting gradients
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Connected components → keep only localized patches
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter:
        # too small = noise
        # too big = lighting/shadow
        if 30 < area < 600:
            filtered_mask[labels == i] = 1

    if np.sum(filtered_mask) > 0:
        pigmentation_raw = np.mean(dark_diff[filtered_mask == 1])
    else:
        pigmentation_raw = 0

    brightness_variation = np.clip(pigmentation_raw / 50.0, 0, 1)

    # ----------------------------
    # SMOOTHNESS CORRECTION
    # ----------------------------
    smoothness = 1 - texture_roughness

    brightness_variation = np.clip(
        brightness_variation * (0.75 + 0.25 * smoothness),
        0, 1
    )

    # ----------------------------
    # FINAL OUTPUT
    # ----------------------------
    return {
        "redness": float(redness),
        "texture_roughness": float(texture_roughness),
        "edge_density": float(edge_density),
        "brightness_variation": float(brightness_variation)
    }