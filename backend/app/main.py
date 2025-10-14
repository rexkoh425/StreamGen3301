from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .camera import CameraStream
from .encoding import encode_frame_to_jpeg
from .hooks import annotate_frame, sharpen_frame

app = FastAPI(title="USB Camera Stream API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGGER = logging.getLogger(__name__)

RECORD_OUTPUT_DIR = Path(os.getenv("RECORD_OUTPUT_DIR", "recordings"))
RECORD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_FILE_PREFIX = os.getenv("RECORD_FILE_PREFIX", "capture")


class RecordStartRequest(BaseModel):
    path: Optional[str] = None
    fps: Optional[float] = None
    codec: Optional[str] = "mp4v"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_camera() -> CameraStream:
    """
    Build the PiCamera2-backed stream. Camera configuration is controlled via
    environment variables documented in ``backend/app/camera.py``.
    """
    return CameraStream()

camera = _build_camera()

# Sharpen edges slightly to improve perceived crispness without changing framing.
if _env_flag("STREAM_ENABLE_SHARPEN", False):
    camera.register_hook(sharpen_frame)
camera.register_hook(annotate_frame)

@app.on_event("startup")
async def startup_event() -> None:
    try:
        camera.start()
    except Exception:
        LOGGER.exception("Camera failed to start during FastAPI startup.")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    camera.stop()

@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": camera.is_running()}

@app.post("/camera/start")
async def camera_start() -> dict[str, bool]:
    try:
        camera.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True}

@app.post("/camera/stop")
async def camera_stop() -> dict[str, bool]:
    camera.stop()
    return {"ok": True}

@app.get("/camera/status")
async def camera_status() -> dict[str, object]:
    return {
        "running": camera.is_running(),
        "backend": camera.backend_name(),
        "recording": camera.is_recording(),
        "recording_path": camera.recording_path(),
    }

@app.post("/record/start")
async def record_start(
    payload: RecordStartRequest | None = Body(default=None),
) -> dict[str, object]:
    data = payload or RecordStartRequest()
    path = data.path
    fps = data.fps
    codec = data.codec or "mp4v"

    if not path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"{RECORD_FILE_PREFIX}_{timestamp}.mp4"
        path = str((RECORD_OUTPUT_DIR / filename).resolve())

    try:
        actual_path = camera.start_recording(output_path=path, fps=fps, codec=codec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"ok": True, "path": actual_path, "recording": True}

@app.post("/record/stop")
async def record_stop() -> dict[str, object]:
    path = camera.stop_recording()
    if path is None:
        raise HTTPException(status_code=409, detail="Recording not active.")
    return {"ok": True, "path": path, "recording": False}

@app.get("/frame")
async def single_frame() -> Response:
    frame = camera.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame not available yet.")
    jpeg_bytes = encode_frame_to_jpeg(frame, quality=95)
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="Unable to encode frame as JPEG.")
    return Response(content=jpeg_bytes, media_type="image/jpeg")

async def mjpeg_generator(boundary: str) -> AsyncIterator[bytes]:
    boundary_bytes = f"--{boundary}".encode()
    last_ts = 0.0
    stale_repeats = 0
    while True:
        frame, ts = camera.read_latest()
        if frame is None:
            stale_repeats = 0
            await asyncio.sleep(0.01)
            continue

        if ts <= last_ts:
            stale_repeats += 1
            if stale_repeats < 8:
                await asyncio.sleep(0.005)
                continue
        else:
            stale_repeats = 0

        last_ts = ts

        # Encode the freshest frame only; drop anything stale
        jpeg_bytes = encode_frame_to_jpeg(frame, quality=95)
        if jpeg_bytes is None:
            await asyncio.sleep(0.005)
            continue

        yield (
            b"%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
            % (boundary_bytes, len(jpeg_bytes))
        )
        yield jpeg_bytes + b"\r\n"

        # Yield to event loop without enforcing artificial frame delay
        await asyncio.sleep(0)


@app.get("/stream.mjpg")
async def stream() -> StreamingResponse:
    boundary = "frame"
    generator = mjpeg_generator(boundary=boundary)
    return StreamingResponse(
        generator,
        media_type=f"multipart/x-mixed-replace; boundary=%s" % boundary,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )
