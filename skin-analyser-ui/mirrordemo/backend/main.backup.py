"""
DermaCLIP Mirror — Integrated Backend (FastAPI)
================================================
Replaces the Gradio frontend with a FastAPI server that:
  1. Serves the built React frontend as static files.
  2. Exposes a lightweight REST endpoint (/api/scores) that the React UI
     polls every second to get the latest real DermaCLIP scores.
  3. Receives base64-encoded JPEG frames from the browser via
     WebSocket (/ws/frame) for DermaCLIP analysis.
  4. Runs all heavy inference (DermaCLIP scoring + LLM recommendations)
     in background daemon threads, keeping the HTTP layer non-blocking.

Architecture:
  Browser (React + MediaPipe face mesh)
       |-- WebSocket /ws/frame    -> send compressed frames for analysis
       |-- GET /api/scores        -> poll for latest scores + recommendations
       |-- GET /api/health        -> liveness check
       `-- GET / (static files)   -> serves the React build

The React UI handles ALL camera capture, mirroring, face mesh rendering,
and UI animations entirely in-browser — zero Python latency for character
interactions or face landmark visualizations.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import base64
import threading
import time
import json
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# --- Load .env ---
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# --- FastAPI / Uvicorn ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# --- DermaCLIP Backend Modules ---
from patch_analyzer import analyze_skin_patches
from clip_skin_analyzer import analyze_skin_clip
from concept_fusion import compute_concepts
from scoring_engine import compute_scores
from face_regions import extract_skin_regions
from skin_condition_interpreter import interpret_skin_conditions
from llm_recommender import get_product_recommendations, rule_based_recommendation
from dermaclip_analyzer import DermaCLIPAnalyzer
from face_points import find_top_landmark_indices

# ==========================================================================
# MULTI-SESSION ISOLATION STATE (thread-safe)
# ==========================================================================
from typing import Optional, Dict

CAMERA_ACTIVE_TIMEOUT = 3.0  # seconds without a frame → considered stopped

class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.lock = threading.Lock()
        self.created_at = time.time()
        self.last_camera_frame_time = time.time()  # Start from current time to prevent instant pruning
        self.camera_ever_started = False
        self.has_real_inference_completed = False
        self.first_inference_event = threading.Event()
        self.trigger_scoring_event = threading.Event()

        self.scores = {
            "acne": 0.0,
            "pigmentation": 0.0,
            "wrinkles": 0.0,
            "texture": 0.0,
            "dark_circles": 0.0,
            "skin_health": 0.0,
        }

        # Raw full-face probabilities from the fine-tuned 10-class DermaCLIP model.
        # Values are probabilities in the 0.0-1.0 range (not the UI's 0-100 scores).
        self.dermaclip_classes = {}
        self.dermaclip_predicted_class = None
        self.dermaclip_confidence = None

        # 3 tracked landmark points per filter (MediaPipe 468-pt indices)
        self.top_indices = {
            "acne": [50, 280, 152],
            "wrinkles": [10, 67, 297],
            "pigmentation": [116, 345, 234],
            "dark_circles": [159, 386, 145],
            "pores": [4, 197, 206],
        }

        # Primary Top 3 focal points across the whole face
        self.primary_3_points = [
            {"condition": "acne", "index": 50, "rank": 0, "point_id": "acne-0", "score": 25.0},
            {"condition": "wrinkles", "index": 10, "rank": 0, "point_id": "wrinkles-0", "score": 20.0},
            {"condition": "dark_circles", "index": 159, "rank": 0, "point_id": "dark_circles-0", "score": 20.0},
        ]

        self.recommendation = (
            "*Personalized skincare suggestions will generate automatically "
            "once the initial facial diagnostic completes.*"
        )

        self.latest_frame = None
        self.latest_landmarks = None
        self.latest_frame_timestamp = 0.0

        self.next_score_time = 0.0  # Immediately score first received frame
        self.next_llm_time = time.time() + 60.0

        self.is_scoring_running = False
        self.is_llm_running = False

    def is_camera_active(self) -> bool:
        with self.lock:
            if not self.camera_ever_started:
                return False
            return (time.time() - self.last_camera_frame_time) < 8.0  # 8s buffer for frame streaming

    def update_frame(self, frame, landmarks):
        now = time.time()
        with self.lock:
            self.last_camera_frame_time = now
            self.camera_ever_started = True
            self.latest_frame = frame
            self.latest_landmarks = landmarks
            self.latest_frame_timestamp = now

    def reset_camera_state(self):
        with self.lock:
            self.has_real_inference_completed = False
            self.first_inference_event.clear()
            self.dermaclip_classes = {}
            self.dermaclip_predicted_class = None
            self.dermaclip_confidence = None
            self.latest_frame = None
            self.latest_landmarks = None
            self.recommendation = (
                "*Personalized skincare suggestions will generate automatically "
                "once the initial facial diagnostic completes.*"
            )


sessions: Dict[str, SessionState] = {}
sessions_lock = threading.Lock()

def get_session(session_id: Optional[str] = None) -> SessionState:
    sid = (session_id or "default").strip()
    if not sid:
        sid = "default"
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = SessionState(sid)
            print(f"[Session] Initialized isolated session: {sid}")
        return sessions[sid]


# ==========================================================================
# INITIALIZE DERMACLIP SINGLETON (runs at import time, once)
# ==========================================================================
print("[Init] Initializing DermaCLIP (fine-tuned dermatology model)...")
dermaclip = DermaCLIPAnalyzer.get_instance()
print("[Init] DermaCLIP ready.")

# ==========================================================================
# MEDIAPIPE FACE MESH (Python-side, used to extract skin patches from frames
# sent over WebSocket — the browser does its own face mesh for rendering)
# ==========================================================================
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_tracker = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    MEDIAPIPE_AVAILABLE = True
    print("[Init] MediaPipe FaceMesh ready.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    face_mesh_tracker = None
    print("[Init] MediaPipe not available — landmark-based region extraction disabled.")


# ==========================================================================
# WORKER 1: DermaCLIP Scoring (Multi-Session Isolated Loop)
# ==========================================================================
def scoring_worker_loop():
    while True:
        try:
            with sessions_lock:
                active_sessions = list(sessions.values())

            now = time.time()
            had_work = False

            for session in active_sessions:
                if not session.is_camera_active():
                    continue

                with session.lock:
                    if session.is_scoring_running:
                        continue
                    if session.latest_frame is None:
                        continue
                    should_score = (
                        not session.has_real_inference_completed
                        or session.trigger_scoring_event.is_set()
                        or now >= session.next_score_time
                    )
                    if not should_score:
                        continue

                    session.trigger_scoring_event.clear()
                    session.is_scoring_running = True
                    frame = session.latest_frame.copy() if session.latest_frame is not None else None
                    landmarks = session.latest_landmarks

                if frame is None:
                    with session.lock:
                        session.is_scoring_running = False
                    continue

                had_work = True
                try:
                    h, w = frame.shape[:2]
                    regions = {}

                    if landmarks is not None:
                        try:
                            regions, _ = extract_skin_regions(frame, landmarks)
                        except Exception:
                            pass
                        
                        try:
                            res = find_top_landmark_indices(frame, landmarks)
                            if isinstance(res, tuple) and len(res) == 2:
                                top_idx, primary_3 = res
                            else:
                                top_idx, primary_3 = res, None
                            if top_idx:
                                with session.lock:
                                    session.top_indices = top_idx
                                    if primary_3:
                                        session.primary_3_points = primary_3
                        except Exception:
                            pass

                    # Robust anatomical fallback if landmarks are temporarily obscured
                    if not regions or len(regions) == 0:
                        regions = {
                            "forehead": frame[int(0.14*h):int(0.35*h), int(0.25*w):int(0.75*w)],
                            "left_cheek": frame[int(0.38*h):int(0.65*h), int(0.18*w):int(0.45*w)],
                            "right_cheek": frame[int(0.38*h):int(0.65*h), int(0.55*w):int(0.82*w)],
                            "under_eye": frame[int(0.30*h):int(0.48*h), int(0.22*w):int(0.78*w)],
                        }

                    region_concept_list = []
                    # 1. Global Full-Face DermaCLIP Analysis
                    try:
                        full_face_clip = analyze_skin_clip(frame)

                        # Preserve the model's original 10-class full-face output
                        # separately from the existing fused five-metric scoring.
                        raw_probabilities = full_face_clip.get("probabilities", {})
                        if isinstance(raw_probabilities, dict) and raw_probabilities:
                            with session.lock:
                                session.dermaclip_classes = {
                                    str(name): float(value)
                                    for name, value in raw_probabilities.items()
                                }
                                session.dermaclip_predicted_class = full_face_clip.get("predicted_condition")
                                session.dermaclip_confidence = (
                                    float(full_face_clip.get("confidence"))
                                    if full_face_clip.get("confidence") is not None else None
                                )

                        full_face_low = analyze_skin_patches(frame)
                        full_face_concepts = compute_concepts(full_face_low, full_face_clip)
                        if full_face_concepts:
                            region_concept_list.append(full_face_concepts)
                    except Exception:
                        pass

                    # 2. Localized Anatomical Region DermaCLIP Analysis
                    for patch in regions.values():
                        try:
                            if patch is None or patch.size == 0:
                                continue
                            low = analyze_skin_patches(patch)
                            clip_feat = analyze_skin_clip(patch)
                            concepts = compute_concepts(low, clip_feat)
                            if isinstance(concepts, dict) and concepts:
                                region_concept_list.append(concepts)
                        except Exception:
                            continue

                    if region_concept_list:
                        avg_concepts = {}
                        for k in region_concept_list[0].keys():
                            vals = [float(rc.get(k, 0)) for rc in region_concept_list]
                            if not vals:
                                avg_concepts[k] = 0.0
                                continue
                            # Clinical peak-weighted aggregation:
                            # Preserves true focal breakout/damage severity while capturing regional distribution
                            if k in ("bump_density", "inflammation", "semantic_acne",
                                     "pigmentation_spread", "semantic_pigment",
                                     "wrinkle_depth", "edges", "dark_circles"):
                                avg_concepts[k] = float(0.70 * np.max(vals) + 0.30 * np.mean(vals))
                            else:
                                avg_concepts[k] = float(np.mean(vals))

                        with session.lock:
                            session.scores = compute_scores(
                                avg_concepts,
                                prev_scores=(session.scores if session.has_real_inference_completed else None),
                            )
                            if not session.has_real_inference_completed or "Personalized skincare suggestions will generate" in session.recommendation:
                                conds = interpret_skin_conditions(session.scores)
                                session.recommendation = rule_based_recommendation(session.scores, conds)

                            # --- PRIORITY-RANKED 3 POINTS using actual DermaCLIP scores ---
                            # Sort conditions by their real computed severity (descending)
                            # Pick the top 3 most severe conditions and map to their
                            # #1 landmark index from the pixel-filter analysis.
                            cond_score_pairs = [
                                (cond, session.scores[cond])
                                for cond in ("acne", "pigmentation", "wrinkles", "texture", "dark_circles")
                                if cond in session.scores
                            ]
                            # Map "texture" key to "pores" for top_indices lookup
                            COND_TO_IDX_KEY = {
                                "acne": "acne",
                                "pigmentation": "pigmentation",
                                "wrinkles": "wrinkles",
                                "texture": "pores",
                                "dark_circles": "dark_circles",
                            }
                            cond_score_pairs.sort(key=lambda x: x[1], reverse=True)
                            priority_3 = []
                            for cond, score in cond_score_pairs[:3]:
                                idx_key = COND_TO_IDX_KEY.get(cond, cond)
                                top_idx_list = session.top_indices.get(idx_key, [])
                                best_idx = int(top_idx_list[0]) if top_idx_list else {
                                    "acne": 50, "pigmentation": 345, "wrinkles": 10,
                                    "pores": 4, "dark_circles": 159
                                }.get(idx_key, 50)
                                # Use actual cond id that matches CONDITION_SPECS keys in frontend
                                display_cond = "pores" if cond == "texture" else cond
                                priority_3.append({
                                    "condition": display_cond,
                                    "index": best_idx,
                                    "rank": 0,
                                    "point_id": f"{display_cond}-0",
                                    "score": float(score),
                                })
                            if priority_3:
                                session.primary_3_points = priority_3

                            session.has_real_inference_completed = True
                            session.first_inference_event.set()
                            print(f"✅ [DermaCLIP Diagnostic Updated ({session.session_id})] {session.scores}")
                            print(f"   Priority Points: {[(p['condition'], round(p['score'],1)) for p in priority_3]}")

                except Exception as e:
                    print(f"⚠️ Scoring worker error for session {session.session_id}: {e}")
                finally:
                    with session.lock:
                        session.is_scoring_running = False
                        session.next_score_time = time.time() + 3.5

            # Prune stale inactive sessions (> 15 minutes without frames, keeping default)
            with sessions_lock:
                stale = []
                for sid, s in sessions.items():
                    if sid == "default":
                        continue
                    ref_time = s.last_camera_frame_time if s.camera_ever_started else s.created_at
                    if (now - ref_time) > 900:  # 15 minutes
                        stale.append(sid)
                for sid in stale:
                    del sessions[sid]
                    print(f"[Session] Pruned inactive session {sid}")

            if not had_work:
                time.sleep(0.1)
            else:
                time.sleep(0.02)

        except Exception as e:
            print(f"⚠️ Scoring worker loop outer error: {e}")
            time.sleep(0.5)


# ==========================================================================
# WORKER 2: Gemini LLM Advisor (Multi-Session Isolated Loop)
# ==========================================================================
def llm_worker_loop():
    while True:
        try:
            with sessions_lock:
                active_sessions = list(sessions.values())

            now = time.time()
            for session in active_sessions:
                if not session.is_camera_active() or not session.has_real_inference_completed:
                    continue

                with session.lock:
                    if session.is_llm_running or now < session.next_llm_time:
                        continue
                    session.is_llm_running = True
                    current_report = session.scores.copy()

                try:
                    if current_report and session.is_camera_active():
                        conditions = interpret_skin_conditions(current_report)
                        recommendations = get_product_recommendations(
                            report=current_report,
                            lifestyle={"sleep_hours": 7, "water_intake_liters": 2.5},
                            conditions=conditions,
                        )
                        with session.lock:
                            session.recommendation = recommendations
                        print(f"✅ Clinical suggestions updated for session {session.session_id}.")
                except Exception as e:
                    print(f"⚠️ LLM worker error for session {session.session_id}: {e}")
                finally:
                    with session.lock:
                        session.is_llm_running = False
                        session.next_llm_time = time.time() + 60.0

            time.sleep(1.0)
        except Exception as e:
            print(f"⚠️ LLM worker loop outer error: {e}")
            time.sleep(2.0)


# Start background workers
threading.Thread(target=scoring_worker_loop, daemon=True).start()
threading.Thread(target=llm_worker_loop, daemon=True).start()

print("[Init] Background workers started.")

# ==========================================================================
# FASTAPI APPLICATION
# ==========================================================================
app = FastAPI(title="DermaCLIP Mirror API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# /api/health — Liveness check
# --------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "dermaclip_loaded": dermaclip.is_loaded}


# --------------------------------------------------------------------------
# /api/scores — Current DermaCLIP scores + recommendation state
# --------------------------------------------------------------------------
@app.get("/api/scores")
async def get_scores(session_id: Optional[str] = None):
    session = get_session(session_id)
    now = time.time()
    cam_active = session.is_camera_active()

    with session.lock:
        scores = session.scores.copy()
        recommendation = session.recommendation
        has_inference = session.has_real_inference_completed
        s_target = session.next_score_time
        l_target = session.next_llm_time
        scoring_active = session.is_scoring_running
        llm_active = session.is_llm_running
        ever_started = session.camera_ever_started
        top_indices = session.top_indices.copy()
        primary_3_points = session.primary_3_points.copy()
        dermaclip_classes = session.dermaclip_classes.copy()
        dermaclip_predicted_class = session.dermaclip_predicted_class
        dermaclip_confidence = session.dermaclip_confidence

    # Reset state only if camera was active and has now been dead for > 15 seconds
    if ever_started and not cam_active and (now - session.last_camera_frame_time) > 15.0:
        session.reset_camera_state()

    score_cd = 0 if scoring_active else max(0, int(s_target - now))
    llm_cd = 0 if llm_active else max(0, int(l_target - now))

    skin_health_pct = scores.get("skin_health", 0.0)
    acne_val = scores.get("acne", 0.0)
    wrinkles_val = scores.get("wrinkles", 0.0)
    pigment_val = scores.get("pigmentation", 0.0)
    dark_val = scores.get("dark_circles", 0.0)
    texture_val = scores.get("texture", 0.0)

    from skin_condition_interpreter import interpret_skin_conditions
    conditions = interpret_skin_conditions(scores)

    def _get_sev(val):
        if val < 25.0:
            return "Low"
        elif val < 50.0:
            return "Moderate"
        else:
            return "Elevated"

    # Distinct Clinical Filter specifications (3 points per filter + distinct color)
    filters = [
        {
            "id": "pores",
            "name": "Pores & Texture",
            "clinical_name": "Follicular Dilatation & Roughness",
            "score": round(texture_val, 1),
            "severity": _get_sev(texture_val),
            "note": _texture_note(texture_val),
            "color": "#00E676",
            "glow": "rgba(0, 230, 118, 0.45)",
            "side": "left",
            "anchor_index": 4,
            "indices": top_indices.get("pores", [4, 197, 206])
        },
        {
            "id": "wrinkles",
            "name": "Rhytides & Lines",
            "clinical_name": "Micro-Creasing & Elasticity",
            "score": round(wrinkles_val, 1),
            "severity": _get_sev(wrinkles_val),
            "note": _wrinkles_note(wrinkles_val),
            "color": "#00BEFF",
            "glow": "rgba(0, 190, 255, 0.45)",
            "side": "left",
            "anchor_index": 10,
            "indices": top_indices.get("wrinkles", [10, 67, 297])
        },
        {
            "id": "acne",
            "name": "Active Blemishes",
            "clinical_name": "Papules & Comedonal Load",
            "score": round(acne_val, 1),
            "severity": _get_sev(acne_val),
            "note": _acne_note(acne_val),
            "color": "#FF2A3A",
            "glow": "rgba(255, 42, 58, 0.45)",
            "side": "left",
            "anchor_index": 50,
            "indices": top_indices.get("acne", [50, 280, 152])
        },
        {
            "id": "dark_circles",
            "name": "Periorbital Circles",
            "clinical_name": "Infraorbital Venous Stasis",
            "score": round(dark_val, 1),
            "severity": _get_sev(dark_val),
            "note": _dark_circles_note(dark_val),
            "color": "#D946EF",
            "glow": "rgba(217, 70, 239, 0.45)",
            "side": "right",
            "anchor_index": 159,
            "indices": top_indices.get("dark_circles", [159, 386, 145])
        },
        {
            "id": "pigmentation",
            "name": "Hyperpigmentation",
            "clinical_name": "Epidermal Melanin Clusters",
            "score": round(pigment_val, 1),
            "severity": _get_sev(pigment_val),
            "note": _pigmentation_note(pigment_val),
            "color": "#FF8C00",
            "glow": "rgba(255, 140, 0, 0.45)",
            "side": "right",
            "anchor_index": 345,
            "indices": top_indices.get("pigmentation", [116, 345, 234])
        }
    ]

    # Dynamic personalized suggestions based on live DermaCLIP diagnostic
    rec_products = [
        {
            "name": "Barrier Recovery Cleanser",
            "timing": "AM + PM",
            "icon": "droplets",
            "tag": "Cleanser",
            "active": "Gentle Amino Acid Complex"
        }
    ]

    concerns = [
        ("acne", acne_val),
        ("wrinkles", wrinkles_val),
        ("pigmentation", pigment_val),
        ("texture", texture_val),
        ("dark_circles", dark_val),
    ]
    concerns.sort(key=lambda x: x[1], reverse=True)
    top_concern = concerns[0]

    if top_concern[0] == "acne" and top_concern[1] > 15:
        rec_products.append({
            "name": "2% BHA Salicylic Clarifier",
            "timing": "PM (Alt Days)",
            "icon": "flask",
            "tag": "Clarifier",
            "active": "Salicylic Acid + Zinc PCA"
        })
    elif top_concern[0] == "wrinkles" and top_concern[1] > 15:
        rec_products.append({
            "name": "0.05% Encapsulated Retinal",
            "timing": "PM",
            "icon": "moon",
            "tag": "Repair Active",
            "active": "Retinaldehyde + Peptides"
        })
    elif top_concern[0] == "pigmentation" and top_concern[1] > 15:
        rec_products.append({
            "name": "10% Tranexamic + Niacinamide",
            "timing": "AM + PM",
            "icon": "flask",
            "tag": "Brightening",
            "active": "Tranexamic Acid + Arbutin"
        })
    elif top_concern[0] == "dark_circles" and top_concern[1] > 15:
        rec_products.append({
            "name": "5% Caffeine + EGCG Serum",
            "timing": "AM + PM",
            "icon": "flask",
            "tag": "Eye Protocol",
            "active": "Caffeine + Epigallocatechin"
        })
    else:
        rec_products.append({
            "name": "15% L-Ascorbic Acid (Vit C)",
            "timing": "AM",
            "icon": "flask",
            "tag": "Antioxidant",
            "active": "Pure L-Ascorbic + Ferulic"
        })

    if texture_val > 35:
        rec_products.append({
            "name": "5% Niacinamide Pore Refiner",
            "timing": "AM + PM",
            "icon": "flask",
            "tag": "Pore Support",
            "active": "Niacinamide + NAG"
        })
    else:
        rec_products.append({
            "name": "Multi-Molecular Hyaluronic Gel",
            "timing": "AM + PM",
            "icon": "droplets",
            "tag": "Hydrator",
            "active": "5D Multi-Weight HA"
        })

    rec_products.append({
        "name": "Mineral UV Shield SPF 50+",
        "timing": "AM (Daily)",
        "icon": "sun",
        "tag": "Photoprotection",
        "active": "Zinc Oxide 18.5% + Ectoin"
    })

    return JSONResponse({
        "camera_active": cam_active,
        "camera_ever_started": ever_started,
        "has_inference": has_inference,
        "score": round(skin_health_pct, 1),
        "status": _health_label(skin_health_pct),
        "skin_type": conditions.get("Skin Type", "Combination"),
        "hydration": conditions.get("Hydration Level", "Moderate"),
        "acne_severity": conditions.get("Acne Severity", "Low"),
        "wrinkle_risk": conditions.get("Wrinkle Risk", "Low"),
        "pigmentation_risk": conditions.get("Pigmentation Risk", "Low"),
        "metrics": filters,
        "top_indices": top_indices,
        "primary_3_points": primary_3_points,
        "recommendations": rec_products,
        "recommendation_text": recommendation,
        "scoring_active": scoring_active,
        "llm_active": llm_active,
        "score_countdown": score_cd,
        "llm_countdown": llm_cd,
        "raw_scores": scores,
        "dermaclip_classes": dermaclip_classes,
        "dermaclip_predicted_class": dermaclip_predicted_class,
        "dermaclip_confidence": dermaclip_confidence,
    })


def _texture_note(val):
    if val < 20: return "Refined micro-texture"
    elif val < 35: return "Slight follicular visibility"
    elif val < 50: return "Moderate T-zone pores"
    return "Prominent pore dilation"


def _wrinkles_note(val):
    if val < 15: return "Minimal dynamic lines"
    elif val < 30: return "Fine superficial creasing"
    elif val < 50: return "Moderate dermal lines"
    return "Prominent rhytides"


def _acne_note(val):
    if val < 15: return "Clear epidermal stratum"
    elif val < 30: return "Mild microcomedones"
    elif val < 50: return "Moderate active papules"
    return "Elevated blemish activity"


def _dark_circles_note(val):
    if val < 20: return "Minimal periorbital congestion"
    elif val < 35: return "Mild tear-trough shadow"
    elif val < 50: return "Moderate infraorbital pooling"
    return "Pronounced dark circles"


def _pigmentation_note(val):
    if val < 20: return "Uniform melanin distribution"
    elif val < 35: return "Localized mild dyschromia"
    elif val < 50: return "Moderate photodamage"
    return "Elevated melanin clusters"


def _health_label(score):
    if score >= 80:
        return "Optimal Barrier Integrity"
    elif score >= 65:
        return "Stable Barrier Function"
    elif score >= 50:
        return "Fair Epidermal Health"
    else:
        return "Compromised Barrier"


# --------------------------------------------------------------------------
# /ws/frame — WebSocket: receive base64 JPEG frames from browser
# --------------------------------------------------------------------------
@app.websocket("/ws/frame")
async def websocket_frame(websocket: WebSocket, session_id: Optional[str] = None):
    """
    Receives compressed JPEG frames from the browser (base64 encoded).
    Runs MediaPipe face detection on them and stores the latest face data
    in the isolated session so the scoring worker can consume it.

    Frame sending rate from the browser is throttled to ~2 FPS to minimise
    bandwidth and CPU overhead without impacting UI responsiveness (the
    browser does its own full-speed face mesh for rendering).
    """
    session = get_session(session_id)
    await websocket.accept()
    print(f"[WS] Client connected for session [{session.session_id}] — frame receiver active.")

    try:
        while True:
            message = await websocket.receive_text()

            try:
                # Decode base64 → JPEG → numpy BGR
                img_data = base64.b64decode(message)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None or frame.size == 0:
                    continue

                # Downscale for efficiency (analysis doesn't need full resolution)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))

                # Light exposure normalization
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                y_ch, cr, cb = cv2.split(ycrcb)
                y_ch = cv2.equalizeHist(y_ch)
                frame = cv2.cvtColor(cv2.merge((y_ch, cr, cb)), cv2.COLOR_YCrCb2BGR)

                # Run MediaPipe face detection if available
                landmarks = None
                if MEDIAPIPE_AVAILABLE and face_mesh_tracker is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh_tracker.process(rgb)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]

                session.update_frame(frame, landmarks)

                # Trigger immediate scoring on first face detection
                if not session.has_real_inference_completed:
                    session.next_score_time = 0.0
                    session.trigger_scoring_event.set()

            except Exception as e:
                print(f"[WS] Frame decode error ({session.session_id}): {e}")
                continue

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected [{session.session_id}].")
    except Exception as e:
        print(f"[WS] Unexpected error [{session.session_id}]: {e}")


# --------------------------------------------------------------------------
# Static file serving — serves the integrated frontend HTML
# --------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_V2 = FRONTEND_DIST / "mirror_v2.html"


@app.get("/v2", include_in_schema=False)
async def serve_v2():
    if FRONTEND_V2.exists():
        return FileResponse(str(FRONTEND_V2), media_type="text/html")
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX), media_type="text/html")
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/", include_in_schema=False)
async def serve_root():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX), media_type="text/html")
    return JSONResponse({
        "message": "DermaCLIP Mirror API running. No frontend found.",
        "hint": "Place the frontend/dist/index.html at the expected path.",
    }, status_code=404)


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    # Serve static assets if they exist, otherwise fall back to index.html
    asset_file = FRONTEND_DIST / full_path
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(str(asset_file))
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX), media_type="text/html")
    return JSONResponse({"error": "Not found"}, status_code=404)


# ==========================================================================
# ENTRYPOINT
# ==========================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
