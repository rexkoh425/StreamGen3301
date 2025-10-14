# Pi Camera Streaming Pipeline

This repo contains a lightweight pipeline for streaming Raspberry Pi camera frames to a browser. It is split into a FastAPI backend and a static frontend:

```
backend/   # FastAPI application that exposes MJPEG and single-frame endpoints
frontend/  # Static site that talks to the backend, displays the stream, and controls the camera
```

## Docker Compose

You can run the backend and frontend with Docker (compose v2). The stack mounts a persistent recordings volume and exposes the services on the host.

```bash
# Optional: enable the camera when running on a Raspberry Pi host.
export CAMERA_ENABLED=1

docker compose up --build
```

Access the frontend at <http://localhost:3000>. The backend is forwarded to <http://localhost:8000>.

> **Note**  
> The backend image is based on `dtcooper/raspberrypi-os:bookworm` so it expects an ARMv7/ARM64 host (e.g., Raspberry Pi). Buildx or emulation is required to run it on other architectures.

### Pi camera hardware inside the container

The backend image disables the camera by default (`CAMERA_ENABLED=0`) so it can run on development machines without hardware. When deploying on a Raspberry Pi:

1. Set `CAMERA_ENABLED=1` in the environment (as shown above).
2. Uncomment the `devices`, `security_opt`, and `privileged` lines under the `backend` service in `docker-compose.yml`, then adjust for your camera nodes (e.g. `/dev/video0`, `/dev/video1`).
3. Ensure the host has the PiCamera2 stack installed so the container can access the bindings.

Recordings are stored in a named Docker volume (`backend-recordings`). You can point `RECORD_OUTPUT_DIR` to a bind mount path instead if you need direct host access.

## Backend

### Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Raspberry Pi devices with PiCamera2 installed, the service automatically uses it. Streams default to **1536×864 @ 20 FPS** to match the HQ camera profile. If the module or hardware is unavailable, start the API with `CAMERA_ENABLED=0` so that non-camera features remain accessible.

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Key endpoints:

| Endpoint              | Method | Description                                |
|-----------------------|--------|--------------------------------------------|
| `/health`             | GET    | Basic readiness check                      |
| `/camera/status`      | GET    | Indicates backend name (`picamera2` etc.)  |
| `/camera/start`       | POST   | Ensures capture thread is running          |
| `/camera/stop`        | POST   | Stops capture                              |
| `/frame`              | GET    | Returns a single JPEG frame                |
| `/stream.mjpg`        | GET    | MJPEG stream for browsers/clients          |

Hook in custom frame logic through `backend/app/hooks.py`. The default implementation is a stub; return a modified `numpy` array to replace the stream frame, or `None` to leave it intact.

## Frontend

The frontend is a static page that interacts with the backend via fetch and displays the MJPEG stream.

### Quick start

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` (adjust the port if you chose a different one). Set the backend URL (defaults to `http://localhost:8000`), press **Connect**, then **Start Camera**. Use **Capture Frame** to request `/frame`.

Feel free to serve the frontend with any static file server (`npx serve`, nginx, etc.).

## Record the Stream

You can persist the live stream using either a built-in script or `ffmpeg`.

### Python recorder

```bash
cd backend
pip install -r requirements.txt  # ensures requests/opencv are available
cd ..
python scripts/record_stream.py --url http://localhost:8000/stream.mjpg --output recording.mp4
```

Press `Ctrl+C` to stop recording; the script finalises the MP4 automatically. Use `--limit-frames` to capture a fixed number of frames.

### ffmpeg

```bash
ffmpeg -i http://localhost:8000/stream.mjpg -c:v libx264 recording.mp4
```

This copies the MJPEG stream into an H.264 MP4. Adjust codecs or destinations as needed.

## Next Steps

- Implement annotation/model inference inside `backend/app/hooks.py`.
- Protect the endpoints (authentication, HTTPS) for production scenarios.
- Add tests (e.g., backend unit tests with mocked camera sources).
