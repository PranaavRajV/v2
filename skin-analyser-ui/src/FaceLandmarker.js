
import {
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

let faceLandmarker = null;

// Initialize MediaPipe
export async function initializeFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "VIDEO",

    // Detect multiple faces so we can choose the closest one
    numFaces: 1,

    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false
  });

  console.log("MediaPipe Face Landmarker initialized!");

  return faceLandmarker;
}

// Find the largest face using its landmark bounding box
function getFaceArea(landmarks) {
  const xs = landmarks.map(point => point.x);
  const ys = landmarks.map(point => point.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return (maxX - minX) * (maxY - minY);
}

// Detect and select the closest face
export function detectFace(videoElement) {
  if (!faceLandmarker || !videoElement) return null;

  if (
    videoElement.readyState < 2 ||
    videoElement.videoWidth === 0 ||
    videoElement.videoHeight === 0
  ) {
    return null;
  }

  const results = faceLandmarker.detectForVideo(
    videoElement,
    performance.now()
  );

  const detectedFaces = results.faceLandmarks;

  if (!detectedFaces || detectedFaces.length === 0) {
    return null;
  }

  // Select the face occupying the largest area
  const selectedFace = detectedFaces.reduce((largest, current) => {
    return getFaceArea(current) > getFaceArea(largest)
      ? current
      : largest;
  });

  return {
    landmarks: selectedFace,
    connections: FaceLandmarker.FACE_LANDMARKS_TESSELATION
  };
}

// Close MediaPipe
export function closeFaceLandmarker() {
  faceLandmarker?.close();
  faceLandmarker = null;
}
