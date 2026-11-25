import cv2

# Comprehensive Dermatology Region Landmark Indices (MediaPipe FaceMesh 468-pt)
# Covers all 9 anatomical clinical zones across the face
FACE_REGIONS = {
    "forehead": [10, 338, 297, 332],
    "glabella": [54, 103, 67, 109],
    "left_cheek": [50, 205, 187, 147],
    "right_cheek": [280, 425, 411, 376],
    "nose_tzone": [4, 5, 195, 197],
    "chin_jaw": [17, 18, 200, 152],
    "left_under_eye": [33, 133, 159, 145],
    "right_under_eye": [263, 362, 386, 374],
    "temples": [117, 118, 346, 347]
}


# -----------------------------
# Extract skin regions
# -----------------------------
def extract_skin_regions(frame, landmarks):
    h, w, _ = frame.shape
    regions = {}
    boxes = {}

    for name, points in FACE_REGIONS.items():
        xs = []
        ys = []

        for p in points:
            lm = landmarks.landmark[p]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x_min = max(min(xs), 0)
        x_max = min(max(xs), w)
        y_min = max(min(ys), 0)
        y_max = min(max(ys), h)

        # Skip invalid or collapsed regions
        if x_max <= x_min or y_max <= y_min:
            continue

        patch = frame[y_min:y_max, x_min:x_max]
        regions[name] = patch
        boxes[name] = (x_min, y_min, x_max - x_min, y_max - y_min)

    return regions, boxes


# -----------------------------
# Draw region boxes on frame (debug)
# -----------------------------
def draw_skin_regions(frame, landmarks):
    h, w, _ = frame.shape

    for name, points in FACE_REGIONS.items():
        xs = []
        ys = []

        for p in points:
            lm = landmarks.landmark[p]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x_min = max(min(xs), 0)
        x_max = min(max(xs), w)
        y_min = max(min(ys), 0)
        y_max = min(max(ys), h)

        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(
            frame,
            name,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return frame
