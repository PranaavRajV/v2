import React, { useEffect, useRef, useState } from "react";

import {
  Heart,
  Leaf,
  Sparkles,
  ArrowRight,
  ScanFace,
  UserRound,
  Accessibility,
  Waves,
  Shirt,
  MoreHorizontal,
  Droplets,
  Sun,
  FlaskConical,
  Camera,
  CameraOff
} from "lucide-react";

import {
  initializeFaceLandmarker,
  detectFace,
  closeFaceLandmarker
} from "./FaceLandmarker";

// --------------------------------------------------
// LIVE DERMAClIP BACKEND CONFIGURATION
// --------------------------------------------------

// In production set VITE_API_BASE to your deployed backend URL, e.g.
//   https://dermaclip-api.onrender.com
// or the Railway public URL.  Falls back to localhost for local dev.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8081";
const SESSION_ID = "mirror-demo";

const METRIC_POSITIONS = {
  pores: { x: 39, y: 32 },
  wrinkles: { x: 38, y: 49 },
  acne: { x: 40, y: 65 },
  dark_circles: { x: 65, y: 32 },
  pigmentation: { x: 67, y: 50 }
};

function recommendationIcon(icon) {
  const icons = {
    droplets: Droplets,
    sun: Sun,
    flask: FlaskConical,
    flaskconical: FlaskConical
  };
  return icons[String(icon || "").toLowerCase()] || Sparkles;
}

function mapBackendMetrics(metrics = []) {
  return metrics.map(metric => ({
    ...metric,
    x: METRIC_POSITIONS[metric.id]?.x ?? (metric.side === "right" ? 65 : 39),
    y: METRIC_POSITIONS[metric.id]?.y ?? 50,
    // Backend metric scores represent concern severity: lower is better.
    note: `${metric.severity || "—"} severity · ${metric.note || ""}`
  }));
}

// --------------------------------------------------
// NAVIGATION
// --------------------------------------------------

const navItems = [
  { label: "Skin", Icon: ScanFace },
  { label: "Face", Icon: UserRound },
  { label: "Body", Icon: Accessibility },
  { label: "Hair", Icon: Waves },
  { label: "Style", Icon: Shirt },
  { label: "More", Icon: MoreHorizontal }
];

// --------------------------------------------------
// DATE FORMAT
// --------------------------------------------------

function formatDate(date) {
  return date.toLocaleDateString("en-US", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric"
  });
}

// --------------------------------------------------
// SCORE RING
// --------------------------------------------------

function ScoreRing({ score }) {
  return (
    <div
      className="score-ring"
      style={{ "--score": `${score * 3.6}deg` }}
    >
      <div className="score-ring-inner">
        <strong>{score}</strong>
        <span>/ 100</span>
      </div>
    </div>
  );
}


// --------------------------------------------------
// CAMERA VIEW — CLEAN LIVE VIDEO, NO DENSE LANDMARK OVERLAY
// --------------------------------------------------

function CameraView({
  videoRef,
  cameraOn,
  error,
  onStart,
  onStop,
  faceLandmarks
}) {
  return (
    <div className="camera-layer">
      {/* LIVE CAMERA — facial tracking continues in the background */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className={`camera-video ${cameraOn ? "visible" : ""}`}
      />

      {/* CAMERA PLACEHOLDER */}
      {!cameraOn && (
        <div className="camera-placeholder">
          <div className="placeholder-subtitle">
            Enable the camera to begin your skin analysis
          </div>

          <button className="camera-button" onClick={onStart}>
            <Camera size={16} />
            Enable camera
          </button>

          {error && <div className="camera-error">{error}</div>}
        </div>
      )}

      {/* CAMERA OFF BUTTON */}
      {cameraOn && (
        <button
          className="camera-toggle"
          onClick={onStop}
          title="Turn camera off"
        >
          <CameraOff size={16} />
        </button>
      )}

      <div className="camera-vignette" />
    </div>
  );
}

// --------------------------------------------------
// SKIN METRIC CALLOUTS
// --------------------------------------------------

function MetricCallout({ metric }) {
  const isLeft = metric.side === "left";

  return (
    <div
      className={`metric-callout ${
        isLeft ? "metric-left" : "metric-right"
      }`}
      style={{
        left: `${metric.x}%`,
        top: `${metric.y}%`
      }}
    >

      <span className="metric-target">
        <i />
      </span>

      <span className="metric-connector" />

      <div className="metric-copy">
        <span className="metric-name">
          {metric.name}
        </span>

        <strong>{Math.round(Number(metric.score) || 0)}<small className="metric-unit">/100</small></strong>

        <small>{metric.note}</small>
      </div>

    </div>
  );
}

// --------------------------------------------------
// MAIN APPLICATION
// --------------------------------------------------

function App() {

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const trackingRef = useRef(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const [faceLandmarks, setFaceLandmarks] = useState(null);
  const [faceTracking, setFaceTracking] = useState(false);
  const [backendData, setBackendData] = useState(null);
  const [backendError, setBackendError] = useState("");
  const [backendConnected, setBackendConnected] = useState(false);

  const [now, setNow] = useState(new Date());
  const [activeNav, setActiveNav] = useState("Skin");
  const [showRoutine, setShowRoutine] = useState(false);

  // ------------------------------------------------
  // CLOCK
  // ------------------------------------------------

  useEffect(() => {
    const timer = setInterval(() => {
      setNow(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // ------------------------------------------------
  // CAMERA CLEANUP
  // ------------------------------------------------

  useEffect(() => {
    return () => {
      trackingRef.current = false;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      streamRef.current?.getTracks().forEach(track => {
        track.stop();
      });

      streamRef.current = null;

      closeFaceLandmarker();
    };
  }, []);

  // ------------------------------------------------
  // START CAMERA
  // ------------------------------------------------

  async function startCamera() {

    setCameraError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera access is unavailable. Open this app on localhost or HTTPS."
      );
      return;
    }

    try {

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });

      streamRef.current = stream;

      if (!videoRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      setCameraOn(true);

    } catch (err) {

      streamRef.current?.getTracks().forEach(track => {
        track.stop();
      });

      streamRef.current = null;

      setCameraError(
        err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser settings."
          : "Could not start the camera. Check that it is connected and not in use by another app."
      );

    }
  }

  // ------------------------------------------------
  // STOP CAMERA
  // ------------------------------------------------

  function stopCamera() {

    trackingRef.current = false;

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    streamRef.current?.getTracks().forEach(track => {
      track.stop();
    });

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setFaceLandmarks(null);
    setFaceTracking(false);
  }

  
  // ------------------------------------------------
  // DERMAClIP BACKEND STREAM + LIVE SCORE POLLING
  // ------------------------------------------------

  useEffect(() => {
    if (!cameraOn) {
      setBackendConnected(false);
      return;
    }

    let disposed = false;
    const sessionId = SESSION_ID;
    setBackendData(null);
    setBackendError("");
    let socket = null;
    let frameTimer = null;
    let scoreTimer = null;
    let canvas = null;
    let context = null;

    async function pollScores() {
      try {
        const response = await fetch(
          `${API_BASE}/api/scores?session_id=${encodeURIComponent(sessionId)}`
        );
        if (!response.ok) throw new Error(`Scores API returned ${response.status}`);
        const data = await response.json();
        if (!disposed) {
          setBackendData(data);
          setBackendError("");
        }
      } catch (error) {
        if (!disposed) setBackendError(`Cannot read DermaCLIP results: ${error.message}`);
      }
    }

    function connectAndStream() {
      const wsUrl = `${API_BASE.replace(/^http/, "ws")}/ws/frame?session_id=${encodeURIComponent(sessionId)}`;
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        if (disposed) return;
        setBackendConnected(true);

        // Backend is designed for approximately 2 JPEG frames per second.
        canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        context = canvas.getContext("2d", { alpha: false });

        frameTimer = window.setInterval(() => {
          const video = videoRef.current;
          if (
            !video ||
            video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
            !context ||
            socket?.readyState !== WebSocket.OPEN
          ) return;

          const width = 640;
          const height = Math.max(1, Math.round(width * video.videoHeight / video.videoWidth));
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }

          context.drawImage(video, 0, 0, width, height);
          const jpegBase64 = canvas.toDataURL("image/jpeg", 0.68).split(",")[1];
          if (jpegBase64) socket.send(jpegBase64);
        }, 500);

        pollScores();
        scoreTimer = window.setInterval(pollScores, 1000);
      };

      socket.onerror = () => {
        if (!disposed) {
          setBackendConnected(false);
          setBackendError("Could not connect to the DermaCLIP WebSocket. Check that the backend is running on port 8081.");
        }
      };

      socket.onclose = () => {
        if (!disposed) setBackendConnected(false);
      };
    }

    connectAndStream();

    return () => {
      disposed = true;
      if (frameTimer) window.clearInterval(frameTimer);
      if (scoreTimer) window.clearInterval(scoreTimer);
      if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
      setBackendConnected(false);
    };
  }, [cameraOn]);

  // ------------------------------------------------
  // MEDIAPIPE INITIALIZATION AND OPTIMIZED FACE TRACKING
  // ------------------------------------------------

  useEffect(() => {
    if (!cameraOn) return;

    let cancelled = false;
    let lastDetectionTime = 0;

    const DETECTION_INTERVAL = 1000 / 15; // 15 FPS

    async function startTracking() {
      try {
        const landmarker = await initializeFaceLandmarker();

        if (cancelled) return;

        trackingRef.current = true;

        console.log("MediaPipe Face Landmarker initialized!");

        function detect(timestamp) {
          if (
            cancelled ||
            !trackingRef.current ||
            !videoRef.current
          ) {
            return;
          }

          // Run detection only 15 times per second
          if (
            timestamp - lastDetectionTime >= DETECTION_INTERVAL
          ) {
            lastDetectionTime = timestamp;

            const detection = detectFace(videoRef.current);

            setFaceLandmarks(detection);

            setFaceTracking(
              Boolean(detection?.landmarks?.length)
            );
          }

          // Keep the animation loop alive
          animationRef.current =
            requestAnimationFrame(detect);
        }

        animationRef.current =
          requestAnimationFrame(detect);

      } catch (error) {
        console.error("MediaPipe initialization error:", error);

        setFaceTracking(false);

        setCameraError(
          "Face tracking could not initialize. Check your internet connection and browser console."
        );
      }
    }

    startTracking();

    return () => {
      cancelled = true;
      trackingRef.current = false;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [cameraOn]);


  // ------------------------------------------------
  // APPLICATION UI
  // ------------------------------------------------

  return (

    <main className="mirror-screen">

      {/* CAMERA */}

      <CameraView
  videoRef={videoRef}
  cameraOn={cameraOn}
  error={cameraError}
  onStart={startCamera}
  onStop={stopCamera}
  faceLandmarks={faceLandmarks}
/>

      {/* HEADER */}

      <header className="top-header">

        <div>
          <h1>Your Skin Analysis</h1>
          <p>Real insights. Healthier you.</p>
        </div>

        <div className="clock">

          <strong>
            {now.toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit"
            })}
          </strong>

          <span>{formatDate(now)}</span>

        </div>

      </header>

      {/* GREETING */}

      <div className="radiant-greeting">
        You look radiant <Heart size={24} />
      </div>

      {/* FACIAL ANALYSIS */}

      <section
        className="face-analysis"
        aria-label="Facial skin analysis overlay"
      >

        <div className="face-particle-field" />

        {/* STATIC FACE WHEN CAMERA IS OFF */}

        {!cameraOn && (

          <div className="face-visual">

            <svg
              viewBox="0 0 300 480"
              className="face-outline"
              aria-hidden="true"
            >

              <path
                d="M150 24 C75 24 42 85 48 170
                   C52 275 91 376 150 412
                   C209 376 248 275 252 170
                   C258 85 225 24 150 24Z"
                className="face-contour"
              />

              <path
                d="M91 105 Q150 78 209 105"
                className="face-detail"
              />

              <path
                d="M78 170 Q104 148 130 166"
                className="face-detail"
              />

              <path
                d="M170 166 Q196 148 222 170"
                className="face-detail"
              />

              <path
                d="M79 190 Q105 168 131 190 Q105 210 79 190Z"
                className="face-detail"
              />

              <path
                d="M169 190 Q195 168 221 190 Q195 210 169 190Z"
                className="face-detail"
              />

              <path
                d="M150 185 Q143 240 132 267 Q150 280 168 267"
                className="face-detail"
              />

              <path
                d="M112 315 Q131 299 150 310 Q169 299 188 315 Q170 341 150 339 Q130 341 112 315Z"
                className="face-detail"
              />

              <path
                d="M105 365 Q150 393 195 365"
                className="face-detail"
              />

              {[
                [150, 70],
                [83, 190],
                [217, 190],
                [150, 260],
                [112, 315],
                [188, 315],
                [150, 390]
              ].map(([x, y], index) => (

                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="2.5"
                  className="face-scan-point"
                />

              ))}

            </svg>

          </div>

        )}

        {/* FACE TRACKING STATUS */}

        {cameraOn && (
          <div className="tracking-status">
            {faceTracking ? "Face detected" : "Searching for face..."}
            <span>
              {backendData?.has_inference
                ? " · DermaCLIP analysis live"
                : backendConnected
                  ? " · Sending frames for analysis"
                  : " · Connecting to analysis engine"}
            </span>
          </div>
        )}

        {cameraOn && backendError && (
          <div className="camera-error backend-error">{backendError}</div>
        )}

        {/* SKIN METRICS */}

        {backendData?.has_inference && mapBackendMetrics(backendData.metrics).map(metric => (
          <MetricCallout key={metric.id} metric={metric} />
        ))}

      </section>

      {/* RIGHT SIDE PANELS */}
      <aside className="right-rail">
        <section className="glass-card health-card">
          <h2>Skin Health Score</h2>

          {backendData?.has_inference ? (
            <>
              <ScoreRing score={Math.max(0, Math.min(100, Math.round(backendData.score || 0)))} />
              <div className="health-status">
                <Leaf size={21} fill="currentColor" />
                {backendData.status || "Analysis complete"}
              </div>
              <div className="card-divider" />
              <h3>Your Skin Profile</h3>
              <dl className="profile-list">
                <div><dt>Skin Type</dt><dd>{backendData.skin_type || "—"}</dd></div>
                <div><dt>Hydration</dt><dd>{backendData.hydration || "—"}</dd></div>
                <div><dt>Acne Severity</dt><dd>{backendData.acne_severity || "—"}</dd></div>
                <div><dt>Wrinkle Risk</dt><dd>{backendData.wrinkle_risk || "—"}</dd></div>
                <div><dt>Pigmentation Risk</dt><dd>{backendData.pigmentation_risk || "—"}</dd></div>
              </dl>
              <p className="backend-disclaimer">AI-generated skin screening estimate · Not a medical diagnosis</p>
            </>
          ) : (
            <div className="analysis-pending">
              <ScoreRing score={0} />
              <div className="health-status">
                <Sparkles size={19} />
                {cameraOn ? "Analysis in progress" : "Awaiting camera"}
              </div>
              <p>
                {cameraOn
                  ? "Keep your face centered and well lit while DermaCLIP analyses the camera feed."
                  : "Enable the camera to begin your live DermaCLIP skin analysis."}
              </p>
              {backendError && <small>{backendError}</small>}
            </div>
          )}
        </section>

        {/* LIVE RECOMMENDATIONS */}
        <section className="glass-card recommendations-card">
          <h2>Recommendations</h2>
          {backendData?.has_inference && backendData.recommendations?.length ? (
            <div className="recommendation-list">
              {backendData.recommendations.map((item, index) => {
                const Icon = recommendationIcon(item.icon);
                return (
                  <div className="recommendation" key={`${item.name}-${index}`}>
                    <div className="product-icon">
                      <Icon size={22} strokeWidth={1.35} />
                    </div>
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.timing || item.tag || ""}</span>
                      {item.active && <small>{item.active}</small>}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="recommendations-pending">
              Personalized recommendations will appear after the first successful skin analysis.
            </p>
          )}
          {backendData?.recommendation_text && backendData.has_inference && (
            <p className="recommendation-summary">{backendData.recommendation_text}</p>
          )}
        </section>
      </aside>

      {/* BOTTOM NAVIGATION */}

      <nav
        className="bottom-navigation"
        aria-label="Mirror modules"
      >

        {navItems.map(({ label, Icon }) => (

          <button
            key={label}
            className={`nav-item ${
              activeNav === label ? "active" : ""
            }`}
            onClick={() => setActiveNav(label)}
            aria-pressed={activeNav === label}
          >

            <span className="nav-icon">
              <Icon
                size={25}
                strokeWidth={1.25}
              />
            </span>

            <span>{label}</span>

          </button>

        ))}

      </nav>

      {/* PERSONALIZED ROUTINE BUTTON */}

      <button
        className="routine-button"
        onClick={() => setShowRoutine(value => !value)}
      >

        <Sparkles
          size={20}
          strokeWidth={1.4}
        />

        <span>
          {showRoutine
            ? "Hide Personalized Routine"
            : "See Personalized Routine"}
        </span>

        <ArrowRight size={20} />

      </button>

      {/* ROUTINE POPUP */}

      {showRoutine && (

        <div className="routine-popover glass-card">

          <button
            className="routine-close"
            onClick={() => setShowRoutine(false)}
          >
            ×
          </button>

          <h2>Your daily routine</h2>

          {backendData?.has_inference ? (
            <p>{backendData.recommendation_text || "Your personalized recommendations are shown in the panel."}</p>
          ) : (
            <p>Complete a skin analysis to generate your personalized routine.</p>
          )}

          <small>AI-generated suggestions · Not a medical prescription.</small>

        </div>

      )}

      {/* DEMO WATERMARK */}

      <div className="demo-watermark">
        DERMACLIP LIVE ANALYSIS · AI SCREENING ESTIMATE
      </div>

    </main>

  );
}

export default App;
