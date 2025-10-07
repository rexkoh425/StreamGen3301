# Pi Camera Streaming Pipeline (No Docker)

This repo contains a lightweight pipeline for streaming Raspberry Pi camera frames to a browser. It is split into a FastAPI backend and a static frontend:

```
backend/   # FastAPI application that exposes MJPEG and single-frame endpoints
frontend/  # Static site that talks to the backend, displays the stream, and controls the camera
```

## Backend

### Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Raspberry Pi devices with PiCamera2 installed, the service automatically uses it. Streams default to **1536×864 @ 20 FPS** to match the HQ camera profile. If PiCamera2 is missing (e.g. on a development laptop), it falls back to OpenCV. To force the OpenCV backend, set `PI_CAMERA_BACKEND=opencv`.

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Key endpoints:

| Endpoint              | Method | Description                                |
|-----------------------|--------|--------------------------------------------|
| `/health`             | GET    | Basic readiness check                      |
| `/camera/status`      | GET    | Indicates backend (`picamera2`/`opencv`)   |
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
