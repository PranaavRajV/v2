
import {
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

let faceLandmarker = null;

/**
 * Initializes the MediaPipe Face Landmarker (browser-side).
 * Called once; subsequent calls return the cached instance.
 * Used exclusively for rendering the live facial mesh overlay.
 * DermaCLIP inference runs on the Python backend.
 */
export async function initializeFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"   // prefer GPU for smooth overlay at 15 FPS
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false
  });

  console.log("MediaPipe Face Landmarker initialized (GPU delegate).");
  return faceLandmarker;
}

/** Returns the bounding area of a face landmark set (for largest-face selection). */
function getFaceArea(landmarks) {
  const xs = landmarks.map(p => p.x);
  const ys = landmarks.map(p => p.y);
  return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
}

/**
 * Runs face detection on a single video frame.
 * Returns { landmarks, connections } for the closest (largest) detected face,
 * or null if no face is found.
 */
export function detectFace(videoElement) {
  if (!faceLandmarker || !videoElement) return null;

  if (
    videoElement.readyState < 2 ||
    videoElement.videoWidth  === 0 ||
    videoElement.videoHeight === 0
  ) {
    return null;
  }

  const results = faceLandmarker.detectForVideo(videoElement, performance.now());
  const detected = results.faceLandmarks;

  if (!detected || detected.length === 0) return null;

  // Pick the face occupying the largest area in frame (closest to camera)
  const best = detected.reduce((largest, current) =>
    getFaceArea(current) > getFaceArea(largest) ? current : largest
  );

  return {
    landmarks: best,
    connections: FaceLandmarker.FACE_LANDMARKS_TESSELATION
  };
}

/** Releases the face landmarker and clears the cached instance. */
export function closeFaceLandmarker() {
  faceLandmarker?.close();
  faceLandmarker = null;
}
