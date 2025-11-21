#!/usr/bin/env python3
"""
FastAPI сервер для Emotion→Meme.
"""

from __future__ import annotations

import base64
import logging
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from emotion_recognition import EMOTIONS, memes, recognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("emotion_api")

app = FastAPI(title="Emotion→Meme API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def _classify_payload(payload: bytes, detect_face: bool) -> dict:
    analysis = recognizer.analyze(payload, detect_face=detect_face)
    emotion = analysis["dominant"]
    return {
        "mode": "face" if detect_face else "meme",
        "dominant_emotion": emotion,
        "confidence": analysis["confidence"],
        "emotions": analysis["emotions"],
        "meme_available": memes.has_meme(emotion),
    }


def _encode_file(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.get("/emotions")
async def emotions() -> dict:
    return {"emotions": EMOTIONS}


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    detect_face: bool = True,
) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")
    return _classify_payload(payload, detect_face)


@app.get("/meme/{emotion}/base64")
async def meme(emotion: str) -> dict:
    emotion = emotion.lower()
    if emotion not in EMOTIONS:
        raise HTTPException(status_code=404, detail="Unknown emotion")
    candidate = memes.pick_random(emotion)
    if not candidate:
        raise HTTPException(status_code=404, detail="Meme not found")
    return {"emotion": emotion, "image": _encode_file(candidate)}


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def _free_port(port: int) -> None:
    if platform.system() == "Windows":
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, check=False
        )
        pids = {
            line.split()[-1]
            for line in result.stdout.splitlines()
            if f":{port}" in line and "LISTENING" in line
        }
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
    else:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False
        )
        for pid in result.stdout.splitlines():
            if pid.strip():
                subprocess.run(["kill", "-9", pid], check=False)


def main() -> None:
    port = int(os.getenv("EMOTION_API_PORT", "8000"))
    if _is_port_in_use(port):
        logger.warning("Port %s busy. Trying to free...", port)
        _free_port(port)
        if _is_port_in_use(port):
            logger.error("Port %s still busy. Abort.", port)
            sys.exit(1)

    uvicorn.run("run_api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as exc:
        logger.exception("Server crashed: %s", exc)
        sys.exit(1)
