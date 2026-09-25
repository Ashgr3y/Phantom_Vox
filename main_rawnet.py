"""
FastAPI backend for SIH 2026 - Voice Cloning Detection
RawNetLite version.

IMPORTANT:
- Existing main.py is NOT modified.
- Uses the pretrained augmented triple cross-domain RawNetLite model.
- Keeps the existing API response structure so the current frontend can work.
"""

import json
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# RawNetLite location
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAWNET_ROOT = BASE_DIR / "models"

RAWNET_MODEL_PATH = (
    RAWNET_ROOT
    / "augmented_triple_cross_domain_focal_rawnet_lite.pt"
)

# Allow Python to import RawNetLite.py from its own project.
if str(RAWNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RAWNET_ROOT))

from RawNetLite import RawNetLite


# ============================================================
# Existing loopback capture system
# ============================================================

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

from record_loopback import (
    CHUNK_SECONDS,
    CHUNKS_DIR,
    SAVE_CHUNKS,
    get_default_loopback_device,
    record_chunk,
)


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 16000

# RawNetLite expects 3 seconds @ 16 kHz.
RAWNET_SECONDS = 3
RAWNET_SAMPLES = TARGET_SR * RAWNET_SECONDS

# 50% overlap between RawNetLite windows.
RAWNET_OVERLAP = 0.50
RAWNET_HOP = int(RAWNET_SAMPLES * (1 - RAWNET_OVERLAP))


# ------------------------------------------------------------
# Prototype decision thresholds
# ------------------------------------------------------------

# These are initially kept simple.
# We are NOT tuning these against the 20-file test set yet.

RAWNET_FAKE_THRESHOLD = 0.50

BLOCK_THRESHOLD = 0.60
FLAG_SPOOF_SCORE = 0.30
FLAG_WINDOW_RATIO = 0.25


AUDIT_LOG_PATH = Path("detection_log.jsonl")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model state
# ============================================================

ml_state = {}


# ============================================================
# Load RawNetLite once at server startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    if not RAWNET_MODEL_PATH.exists():
        raise RuntimeError(
            f"RawNetLite checkpoint not found:\n"
            f"{RAWNET_MODEL_PATH}"
        )

    print("=" * 60)
    print("Loading RawNetLite")
    print("=" * 60)
    print(f"Model:  {RAWNET_MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    model = RawNetLite().to(DEVICE)

    checkpoint = torch.load(
        RAWNET_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.eval()

    ml_state["model"] = model

    print("RawNetLite loaded successfully.")
    print("=" * 60)

    yield

    ml_state.clear()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Voice Cloning Detection API - RawNetLite",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Response schemas
# ============================================================

class WindowResult(BaseModel):
    start_time: float
    end_time: float
    spoof_probability: float
    bonafide_probability: float
    prediction: str


class DetectionResult(BaseModel):
    filename: str
    duration_seconds: float
    windows_analyzed: int

    mean_spoof_score: float
    mean_bonafide_score: float

    spoof_window_ratio: float

    final_prediction: str

    action: str
    message: str

    windows: List[WindowResult]


# ============================================================
# Prevention layer
# ============================================================

def evaluate_policy(
    mean_spoof_score: float,
    spoof_window_ratio: float
) -> tuple[str, str]:

    if mean_spoof_score >= BLOCK_THRESHOLD:
        return (
            "BLOCKED",
            "This audio shows strong signs of being an "
            "AI-generated or cloned voice and has been blocked."
        )

    if (
        mean_spoof_score >= FLAG_SPOOF_SCORE
        or spoof_window_ratio >= FLAG_WINDOW_RATIO
    ):
        return (
            "FLAGGED_FOR_REVIEW",
            "Parts of this audio look suspicious. "
            "It has been flagged for manual review rather "
            "than blocked outright."
        )

    return (
        "ALLOWED",
        "No significant signs of voice cloning were detected."
    )


# ============================================================
# Logging
# ============================================================

def log_detection(result: DetectionResult) -> None:

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": result.filename,
        "mean_spoof_score": result.mean_spoof_score,
        "spoof_window_ratio": result.spoof_window_ratio,
        "final_prediction": result.final_prediction,
        "action": result.action,
    }

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================
# Audio preprocessing
# ============================================================

def load_and_prepare_audio(audio_path: str):

    audio, sample_rate = sf.read(audio_path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise HTTPException(
            status_code=400,
            detail="The audio file is empty."
        )

    # Resample to 16 kHz
    if sample_rate != TARGET_SR:

        audio_tensor = torch.from_numpy(audio)

        audio_tensor = torchaudio.functional.resample(
            audio_tensor,
            sample_rate,
            TARGET_SR
        )

        audio = audio_tensor.numpy()

    return audio


# ============================================================
# Create RawNetLite windows
# ============================================================

def create_rawnet_windows(audio):

    duration_seconds = len(audio) / TARGET_SR

    # Audio shorter than 3 seconds:
    # repeat it until we have 3 seconds.
    if len(audio) < RAWNET_SAMPLES:

        repeat_count = int(
            np.ceil(RAWNET_SAMPLES / len(audio))
        )

        padded_audio = np.tile(
            audio,
            repeat_count
        )[:RAWNET_SAMPLES]

        return [
            (
                padded_audio,
                0.0,
                duration_seconds
            )
        ], duration_seconds

    # Audio >= 3 seconds
    starts = list(
        range(
            0,
            len(audio) - RAWNET_SAMPLES + 1,
            RAWNET_HOP
        )
    )

    # Always include the final possible window.
    last_start = len(audio) - RAWNET_SAMPLES

    if not starts:
        starts = [0]

    if starts[-1] != last_start:
        starts.append(last_start)

    windows = []

    for start in starts:

        end = start + RAWNET_SAMPLES

        windows.append(
            (
                audio[start:end],
                start / TARGET_SR,
                end / TARGET_SR
            )
        )

    return windows, duration_seconds


# ============================================================
# RawNetLite inference
# ============================================================

def run_inference(
    audio_path: str,
    display_name: str
) -> DetectionResult:

    model = ml_state["model"]

    audio = load_and_prepare_audio(audio_path)

    windows, duration_seconds = create_rawnet_windows(audio)

    spoof_scores = []
    bonafide_scores = []

    window_results = []

    with torch.no_grad():

        for (
            window,
            start_time,
            end_time
        ) in windows:

            waveform = (
                torch.from_numpy(
                    window.copy()
                )
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            # RawNetLite output:
            # [batch, 1] = probability of FAKE
            fake_probability = (
                model(waveform)
                .squeeze()
                .item()
            )

            # Numerical safety
            fake_probability = float(
                np.clip(
                    fake_probability,
                    0.0,
                    1.0
                )
            )

            bonafide_probability = (
                1.0 - fake_probability
            )

            prediction = (
                "SPOOF"
                if fake_probability >= RAWNET_FAKE_THRESHOLD
                else "BONAFIDE"
            )

            spoof_scores.append(
                fake_probability
            )

            bonafide_scores.append(
                bonafide_probability
            )

            window_results.append(
                WindowResult(
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    spoof_probability=round(
                        fake_probability,
                        4
                    ),
                    bonafide_probability=round(
                        bonafide_probability,
                        4
                    ),
                    prediction=prediction,
                )
            )

    # ========================================================
    # Aggregate
    # ========================================================

    spoof_scores_arr = np.array(
        spoof_scores
    )

    bonafide_scores_arr = np.array(
        bonafide_scores
    )

    mean_spoof = float(
        np.mean(spoof_scores_arr)
    )

    mean_bonafide = float(
        np.mean(bonafide_scores_arr)
    )

    spoof_window_ratio = float(
        np.mean(
            spoof_scores_arr >= RAWNET_FAKE_THRESHOLD
        )
    )

    final_prediction = (
        "SPOOF"
        if mean_spoof >= RAWNET_FAKE_THRESHOLD
        else "BONAFIDE"
    )

    action, message = evaluate_policy(
        mean_spoof,
        spoof_window_ratio
    )

    result = DetectionResult(
        filename=display_name,
        duration_seconds=round(
            duration_seconds,
            2
        ),
        windows_analyzed=len(windows),
        mean_spoof_score=round(
            mean_spoof,
            4
        ),
        mean_bonafide_score=round(
            mean_bonafide,
            4
        ),
        spoof_window_ratio=round(
            spoof_window_ratio,
            4
        ),
        final_prediction=final_prediction,
        action=action,
        message=message,
        windows=window_results,
    )

    log_detection(result)

    return result


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "RawNetLite",
        "device": str(DEVICE),
        "checkpoint": str(
            RAWNET_MODEL_PATH
        ),
    }


# ============================================================
# /detect
# ============================================================

@app.post(
    "/detect",
    response_model=DetectionResult
)
async def detect(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(
        (".wav", ".flac", ".ogg")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a .wav, .flac, or .ogg file."
            )
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix
    ) as tmp:

        shutil.copyfileobj(
            file.file,
            tmp
        )

        tmp_path = tmp.name

    try:

        result = run_inference(
            tmp_path,
            file.filename
        )

    finally:

        Path(tmp_path).unlink(
            missing_ok=True
        )

    return result


# ============================================================
# Live monitoring
# ============================================================

def result_to_dict(
    result: DetectionResult
) -> dict:

    return (
        result.model_dump()
        if hasattr(result, "model_dump")
        else result.dict()
    )


monitor_lock = threading.Lock()

monitor_state = {
    "running": False,
    "latest": None,
    "history": [],
    "error": None,
    "next_seq": 0,
}

MAX_HISTORY = 20


def monitoring_loop():

    pa = pyaudio.PyAudio()

    try:

        device = get_default_loopback_device(pa)

        with monitor_lock:
            monitor_state["error"] = None

        while True:

            with monitor_lock:

                if not monitor_state["running"]:
                    break

            # Existing 5-second capture
            wav_buffer = record_chunk(
                pa,
                device,
                CHUNK_SECONDS
            )

            chunk_name = "live_chunk.wav"

            if SAVE_CHUNKS:

                CHUNKS_DIR.mkdir(
                    exist_ok=True
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                chunk_path = (
                    CHUNKS_DIR
                    / f"chunk_{timestamp}.wav"
                )

                chunk_path.write_bytes(
                    wav_buffer.getvalue()
                )

                wav_buffer.seek(0)

                chunk_name = chunk_path.name

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(
                    wav_buffer.getvalue()
                )

                tmp_path = tmp.name

            try:

                result = run_inference(
                    tmp_path,
                    chunk_name
                )

                entry = result_to_dict(
                    result
                )

                with monitor_lock:

                    entry["seq"] = monitor_state["next_seq"]
                    monitor_state["next_seq"] += 1

                    monitor_state["latest"] = entry

                    monitor_state["history"].append(
                        entry
                    )

                    monitor_state["history"] = (
                        monitor_state["history"]
                        [-MAX_HISTORY:]
                    )

            except Exception as exc:

                with monitor_lock:
                    monitor_state["error"] = str(exc)

            finally:

                Path(tmp_path).unlink(
                    missing_ok=True
                )

    except Exception as exc:

        with monitor_lock:
            monitor_state["error"] = str(exc)

    finally:

        pa.terminate()

        with monitor_lock:
            monitor_state["running"] = False


# ============================================================
# Start monitoring
# ============================================================

@app.post("/monitor/start")
def start_monitoring():

    if pyaudio is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "PyAudioWPatch is not installed "
                "on the server."
            )
        )

    with monitor_lock:

        if monitor_state["running"]:
            return {
                "status": "already_running"
            }

        monitor_state["running"] = True
        monitor_state["error"] = None
        monitor_state["latest"] = None
        monitor_state["history"] = []
        monitor_state["next_seq"] = 0

    threading.Thread(
        target=monitoring_loop,
        daemon=True
    ).start()

    return {
        "status": "started"
    }


# ============================================================
# Stop monitoring
# ============================================================

@app.post("/monitor/stop")
def stop_monitoring():

    with monitor_lock:
        monitor_state["running"] = False

    return {
        "status": "stopping"
    }


# ============================================================
# Monitor status
# ============================================================

@app.get("/monitor/status")
def monitor_status():

    with monitor_lock:

        return {
            "running": monitor_state["running"],
            "latest": monitor_state["latest"],
            "history": monitor_state["history"],
            "error": monitor_state["error"],
        }