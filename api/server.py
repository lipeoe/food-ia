import base64
import uuid
from typing import Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import TRAINED_MODEL_PATH, CONFIDENCE, FRONTEND_ORIGIN
from core.model_loader import load_model
from core.tracker import ObjectTracker
from inference.predict import predict_frame
from utils.counter import ConveyorCounter

app = FastAPI(title="AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(TRAINED_MODEL_PATH)
sessions: Dict[str, Dict[str, Any]] = {}


class StartSessionRequest(BaseModel):
    direction: str = "lr"


class FrameRequest(BaseModel):
    image_base64: str
    confidence: float | None = None


def _decode_base64_image(data_url: str):
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Imagem inválida")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao decodificar imagem: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start")
def start_session(payload: StartSessionRequest):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "tracker": ObjectTracker(max_distance=50, max_frames_missing=30),
        "counter": ConveyorCounter(
            min_seen_frames=3,
            line_ratio=0.5,
            direction=payload.direction,
            cross_margin=12,
        ),
    }
    return {"session_id": session_id}


@app.post("/session/{session_id}/frame")
def process_frame(session_id: str, payload: FrameRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    frame = _decode_base64_image(payload.image_base64)
    conf = payload.confidence if payload.confidence is not None else CONFIDENCE

    session = sessions[session_id]
    tracker: ObjectTracker = session["tracker"]
    counter: ConveyorCounter = session["counter"]

    _, detections = predict_frame(model, frame, conf=conf)
    active_tracks, finished_tracks = tracker.update(detections, frame.shape)

    newly_counted = counter.update_from_active_tracks(active_tracks, frame.shape)
    counter.cleanup_finished_tracks(finished_tracks)

    # padroniza retorno para backend
    new_events: List[Dict[str, Any]] = []
    for ev in newly_counted:
        new_events.append({
            "track_id": str(ev["id"]),
            "class": ev["class"],
            "confidence": 1.0,
            "timestamp": ev["timestamp"],
        })

    return {
        "new_events": new_events,
        "totals": dict(counter.totals),
        "total_items": int(sum(counter.totals.values())),
    }


@app.post("/session/{session_id}/finalize")
def finalize_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    session = sessions[session_id]
    tracker: ObjectTracker = session["tracker"]
    counter: ConveyorCounter = session["counter"]

    remaining = tracker.flush()
    counter.cleanup_finished_tracks(remaining)
    report = counter.get_json_report()

    del sessions[session_id]
    return report