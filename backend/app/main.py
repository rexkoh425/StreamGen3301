from __future__ import annotations

import os
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .camera import CameraStream
from .encoding import encode_frame_to_jpeg
from .hooks import annotate_frame, rotate_frame, sharpen_frame

app = FastAPI(title="USB Camera Stream API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _build_camera() -> CameraStream:
    """
    USB-only OpenCV (V4L2) camera builder.
    You can tweak device index, resolution, fps, and controls via environment vars:
      CAM_DEVICE_INDEX (int) default 0
      CAM_WIDTH (int)        default 1536
      CAM_HEIGHT (int)       default 864
      CAM_FPS (float)        default 20.0
      CAM_PREFER_MJPEG (0/1) default 1
      CAM_AUTOFOCUS (0/1/None) default None (leave driver default)
      CAM_EXPOSURE (float/None) default None (leave driver default)
    """
    device_index = int(os.getenv("CAM_DEVICE_INDEX", "0"))
    width = int(os.getenv("CAM_WIDTH", "1280"))
    height = int(os.getenv("CAM_HEIGHT", "1024"))
    fps = float(os.getenv("CAM_FPS", "15.0"))
    prefer_mjpeg = os.getenv("CAM_PREFER_MJPEG", "1") not in ("0", "false", "False")
    af_env = os.getenv("CAM_AUTOFOCUS", "").strip()
    exposure_env = os.getenv("CAM_EXPOSURE", "").strip()

    autofocus = None
    if af_env != "":
        autofocus = 1 if af_env in ("1", "true", "True") else 0

    exposure = None
    if exposure_env != "":
        try:
            exposure = float(exposure_env)
        except ValueError:
            exposure = None

    cam = CameraStream(
        device_index=device_index,
        width=width,
        height=height,
        target_fps=fps,
        prefer_mjpeg=prefer_mjpeg,
        reopen_on_fail=True,
        autofocus=autofocus,
        exposure=exposure,
    )
    return cam

camera = _build_camera()
# Rotate incoming frames so the stream is upright before other hooks run
camera.register_hook(rotate_frame(90))
# Sharpen edges slightly to improve perceived crispness without changing framing
camera.register_hook(sharpen_frame)
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
