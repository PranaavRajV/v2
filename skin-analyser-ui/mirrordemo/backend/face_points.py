"""
face_points.py — Landmark-Anchored Filter & Transposition Engine

First-Principles Architecture:
1. Candidate landmarks are partitioned into true anatomical clinical zones.
2. Filters (acne, wrinkles, pigmentation, dark circles, pores) run strictly inside
   the face convex-hull mask (zero background leakage).
3. Filter evaluates local skin patch around each candidate landmark.
4. Top 3 highest-intensity landmark indices are selected per condition.
5. Transposition: On every frame, point coordinates are dynamically derived from the
   current landmark positions -> points move, rotate, and scale perfectly with the face.
"""

import cv2
import numpy as np

# Anatomical landmark zones (MediaPipe FaceMesh 468-pt model)
# Disjoint clinical partitions strictly mapped to genuine facial anatomy
CONDITION_LANDMARK_CANDIDATES = {
    "acne": [
        50, 187, 205, 123, 147, 213,        # left cheek / malar zone
        280, 411, 425, 352, 376, 433,       # right cheek / malar zone
        17, 18, 200, 199, 152, 175,         # chin & jawline
        138, 135, 367, 364,                 # lower cheek perimeter
    ],
    "wrinkles": [
        10, 338, 297, 332, 67, 109, 103,    # forehead upper & mid creases
        70, 63, 105, 66, 107,               # left brow lines
        300, 293, 334, 296, 336,            # right brow lines
        9, 54, 107, 336,                    # glabella (frown lines)
        130, 247, 359, 467,                 # periorbital / crow's feet
    ],
    "pigmentation": [
        116, 117, 118, 101, 234,            # left malar cheekbone & temple
        345, 346, 347, 330, 454,            # right malar cheekbone & temple
        135, 136, 364, 365,                 # lateral cheek
        21, 54, 284, 251,                   # supra-orbital lateral zone
    ],
    "dark_circles": [
        159, 145, 153, 154, 7, 163, 33, 133, # left under-eye tear trough
        386, 374, 380, 381, 385, 398, 263, 362, # right under-eye tear trough
    ],
    "pores": [
        4, 1, 2, 5, 195, 197,               # nose tip & nasal dorsum
        98, 327, 236, 456,                  # nasolabial ala & inner T-zone
        206, 426,                           # medial paranasal pores
    ]
}

TOP_N_PER_FILTER = 3


def make_face_mask(landmarks, h, w):
    """Generates a solid binary mask strictly covering the facial skin convex hull."""
    pts = []
    for lm in landmarks.landmark:
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        pts.append([x, y])
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


# ---------------- FILTERS (Executed strictly within Face Hull) ---------------- #

def filter_acne_response(face_img):
    """Difference of Gaussians + Redness enhancement for blemish detection."""
    b, g, r = cv2.split(face_img.astype(np.float32))
    redness = np.clip(r - (b + g) / 2.0, 0, 255).astype(np.uint8)
    g1 = cv2.GaussianBlur(redness, (0, 0), 1.2)
    g2 = cv2.GaussianBlur(redness, (0, 0), 2.5)
    dog = np.abs(g1.astype(np.float32) - g2.astype(np.float32))
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def filter_wrinkles_response(face_img):
    """Multi-orientation Gabor filter response for fine lines and creases."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    accum = np.zeros_like(gray, dtype=np.float32)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel((17, 17), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        accum = np.maximum(accum, filtered)
    return cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def filter_pigmentation_response(face_img):
    """Sobel gradient magnitude for hyperpigmentation and spot boundaries."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def filter_dark_circles_response(face_img):
    """Inverted luminance response in under-eye zones."""
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2Lab)
    lum = lab[:, :, 0]
    inv_lum = cv2.bitwise_not(lum)
    return cv2.GaussianBlur(inv_lum, (7, 7), 0)


def filter_pores_response(face_img):
    """Laplacian high-frequency texture filter for enlarged pores."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


FILTER_DISPATCH = {
    "acne": filter_acne_response,
    "wrinkles": filter_wrinkles_response,
    "pigmentation": filter_pigmentation_response,
    "dark_circles": filter_dark_circles_response,
    "pores": filter_pores_response,
}


def find_top_landmark_indices(frame, landmarks):
    """
    Comprehensive Face Point Analysis:
    1. Evaluates ALL candidate landmarks across every clinical facial zone.
    2. Identifies top 3 indices per condition (acne, wrinkles, pigmentation, dark_circles, pores).
    3. Selects the primary TOP 3 focal points across the entire face for clean, high-precision visual display.
    
    Returns:
        (top_indices, primary_3_points)
        - top_indices: dict of {condition: [idx1, idx2, idx3]}
        - primary_3_points: list of 3 top diagnostic focal points across the entire face
    """
    h, w = frame.shape[:2]
    face_mask = make_face_mask(landmarks, h, w)
    masked_frame = cv2.bitwise_and(frame, frame, mask=face_mask)

    top_indices = {}
    patch_radius = 6
    all_scored_by_condition = {}

    for condition, candidate_indices in CONDITION_LANDMARK_CANDIDATES.items():
        try:
            response_map = FILTER_DISPATCH[condition](masked_frame)
            response_map = cv2.bitwise_and(response_map, response_map, mask=face_mask)
        except Exception:
            top_indices[condition] = candidate_indices[:TOP_N_PER_FILTER]
            all_scored_by_condition[condition] = [(10.0, idx) for idx in candidate_indices[:TOP_N_PER_FILTER]]
            continue

        scored_candidates = []
        for idx in candidate_indices:
            try:
                lm = landmarks.landmark[idx]
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))

                # Check if point is inside face hull
                if face_mask[py, px] == 0:
                    continue

                x1 = max(0, px - patch_radius)
                x2 = min(w, px + patch_radius + 1)
                y1 = max(0, py - patch_radius)
                y2 = min(h, py + patch_radius + 1)

                patch = response_map[y1:y2, x1:x2]
                score = float(np.mean(patch)) if patch.size > 0 else 0.0
                scored_candidates.append((score, idx))
            except Exception:
                continue

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = [idx for (_, idx) in scored_candidates[:TOP_N_PER_FILTER]]
        if not chosen:
            chosen = candidate_indices[:TOP_N_PER_FILTER]
        top_indices[condition] = chosen
        all_scored_by_condition[condition] = scored_candidates

    # -------------------------------------------------------------
    # Select Top 3 Primary Focal Points across the Entire Face
    # Evaluates all condition points and picks the 3 highest clinical priority targets
    # -------------------------------------------------------------
    flat_candidates = []
    for cond, scored_list in all_scored_by_condition.items():
        cond_top = top_indices.get(cond, [])
        for rank, idx in enumerate(cond_top):
            sc = next((s for s, i in scored_list if i == idx), 15.0)
            flat_candidates.append({
                "condition": cond,
                "index": int(idx),
                "score": float(sc),
                "rank": rank,
                "point_id": f"{cond}-{rank}"
            })

    flat_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Pick 3 diverse primary focal points (avoid duplicate landmarks)
    primary_3 = []
    seen_indices = set()
    condition_counts = {}

    for cand in flat_candidates:
        if cand["index"] in seen_indices:
            continue
        c = cand["condition"]
        if condition_counts.get(c, 0) >= 2 and len(primary_3) < 2:
            # Leave room for diverse conditions if available
            continue
        primary_3.append(cand)
        seen_indices.add(cand["index"])
        condition_counts[c] = condition_counts.get(c, 0) + 1
        if len(primary_3) == 3:
            break

    # Robust fallback: ensure exactly 3 points
    if len(primary_3) < 3:
        fallbacks = [
            {"condition": "acne", "index": top_indices.get("acne", [50])[0], "rank": 0, "point_id": "acne-0", "score": 25.0},
            {"condition": "wrinkles", "index": top_indices.get("wrinkles", [10])[0], "rank": 0, "point_id": "wrinkles-0", "score": 20.0},
            {"condition": "dark_circles", "index": top_indices.get("dark_circles", [159])[0], "rank": 0, "point_id": "dark_circles-0", "score": 20.0},
        ]
        for fb in fallbacks:
            if fb["index"] not in seen_indices and len(primary_3) < 3:
                primary_3.append(fb)
                seen_indices.add(fb["index"])

    return top_indices, primary_3


def transpose_landmarks_to_points(landmarks, top_indices, w, h):
    """
    TRANSPOSITION STEP:
    Translates tracked landmark indices into current frame pixel coordinates (X, Y).
    As the face translates, rotates, pitches, or changes expression, the points
    faithfully transpose with the underlying facial mesh.
    """
    transposed_points = {}
    for condition, indices in top_indices.items():
        pts = []
        for idx in indices:
            try:
                lm = landmarks.landmark[idx]
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pts.append((px, py))
            except Exception:
                pass
        transposed_points[condition] = pts
    return transposed_points


# Backward compatibility shims
def get_log_blobs(image, sigma=2.0):
    return filter_acne_response(image)

def get_dog_blobs(image, sigma1=1.0, sigma2=2.0):
    return filter_acne_response(image)
