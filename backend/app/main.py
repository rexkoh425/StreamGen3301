from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .camera import CameraStream
from .encoding import encode_frame_to_jpeg
from .hooks import annotate_frame

app = FastAPI(title="Pi Camera Stream API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _build_camera() -> CameraStream:
    import os

    config = {"width": 1536, "height": 864, "target_fps": 20.0}
    preferred_backend = os.getenv("PI_CAMERA_BACKEND", "picamera").lower()
    if preferred_backend == "opencv":
        return CameraStream(use_picamera=False, **config)

    try:
        return CameraStream(use_picamera=True, **config)
    except RuntimeError:
        # Fall back to OpenCV automatically if PiCamera2 is unavailable.
        return CameraStream(use_picamera=False, **config)


camera = _build_camera()
camera.register_hook(annotate_frame)


@app.on_event("startup")
async def startup_event() -> None:
    camera.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    camera.stop()


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": camera.is_running()}


@app.post("/camera/start")
async def camera_start() -> dict[str, bool]:
    camera.start()
    return {"ok": True}


@app.post("/camera/stop")
async def camera_stop() -> dict[str, bool]:
    camera.stop()
    return {"ok": True}


@app.get("/camera/status")
async def camera_status() -> dict[str, object]:
    return {"running": camera.is_running(), "backend": camera.backend_name()}


@app.get("/frame")
async def single_frame() -> Response:
    frame = camera.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame not available yet.")
    jpeg_bytes = encode_frame_to_jpeg(frame)
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="Unable to encode frame as JPEG.")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


async def mjpeg_generator(boundary: str) -> AsyncIterator[bytes]:
    boundary_bytes = f"--{boundary}".encode()
    while True:
        frame = camera.read()
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        jpeg_bytes = encode_frame_to_jpeg(frame)
        if jpeg_bytes is None:
            await asyncio.sleep(0.05)
            continue
        yield (
            b"%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
            % (boundary_bytes, len(jpeg_bytes))
        )
        yield jpeg_bytes + b"\r\n"
        await asyncio.sleep(0.03)


@app.get("/stream.mjpg")
async def stream() -> StreamingResponse:
    boundary = "frame"
    generator = mjpeg_generator(boundary=boundary)
    return StreamingResponse(
        generator, media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )
