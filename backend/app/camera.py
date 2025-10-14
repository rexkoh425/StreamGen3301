"""
PiCamera2-based capture pipeline with optional frame hooks and disk recording.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
FrameHook = Callable[[np.ndarray, float], Optional[np.ndarray]]

# PiCamera2 prefers the V4L2 compatibility layer for OpenCV interoperability.
os.environ.setdefault("PICAMERA2_USE_V4L2", "1")

try:  # Lazy import so the backend can start on non-Pi hosts while mocked.
    from picamera2 import Picamera2  # type: ignore
    from libcamera import controls  # type: ignore
except ImportError:  # pragma: no cover - availability depends on runtime image
    Picamera2 = None  # type: ignore
    controls = None  # type: ignore

# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #

def _parse_bool(env: str, default: bool) -> bool:
    value = os.getenv(env)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_size(env: str, default: Tuple[int, int]) -> Tuple[int, int]:
    raw = os.getenv(env)
    if not raw:
        return default
    raw = raw.strip().lower().replace("x", ",")
    parts = [p for p in re.split(r"[, ]+", raw) if p]
    if len(parts) != 2:
        return default
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return default
    if w <= 0 or h <= 0:
        return default
    return w, h


def _select_enum(enum_cls, env: str, default):
    if enum_cls is None:
        return None
    value = os.getenv(env)
    if value is None:
        return default
    lookup = {name.lower(): getattr(enum_cls, name) for name in dir(enum_cls) if not name.startswith("_")}
    return lookup.get(value.strip().lower(), default)


CAMERA_ENABLED = _parse_bool("CAMERA_ENABLED", True)
PICAM_VIDEO_SIZE = _parse_size("PICAM_VIDEO_SIZE", (640, 480))
PICAM_STILL_SIZE = _parse_size("PICAM_STILL_SIZE", (640, 480))
PICAM_FRAME_RATE = float(os.getenv("PICAM_FRAME_RATE", "30"))
PICAM_AF_MODE = _select_enum(getattr(controls, "AfModeEnum", None), "PICAM_AF_MODE", getattr(controls, "AfModeEnum", None).Continuous if controls else None)
PICAM_AF_RANGE = _select_enum(getattr(controls, "AfRangeEnum", None), "PICAM_AF_RANGE", getattr(controls, "AfRangeEnum", None).Full if controls else None)
PICAM_AF_SPEED = _select_enum(getattr(controls, "AfSpeedEnum", None), "PICAM_AF_SPEED", getattr(controls, "AfSpeedEnum", None).Normal if controls else None)

PICAM_COLOR_SPACE = os.getenv("PICAM_COLOR_SPACE", "RGB").strip().upper()
ROTATE_STREAM_90 = _parse_bool("ROTATE_STREAM_90", True)
ROTATE_STREAM_180 = _parse_bool("ROTATE_STREAM_180", False)

TARGET_STREAM_FPS = max(1.0, float(os.getenv("TARGET_STREAM_FPS", "8")))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "85"))

# --------------------------------------------------------------------------- #
# Recording state                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class RecordingState:
    writer: cv2.VideoWriter
    path: str
    fps: float
    started_at: float


# --------------------------------------------------------------------------- #
# Camera stream                                                               #
# --------------------------------------------------------------------------- #


class CameraStream:
    def __init__(self) -> None:
        self._enabled = CAMERA_ENABLED
        self._video_size = PICAM_VIDEO_SIZE
        self._still_size = PICAM_STILL_SIZE
        self._frame_rate = PICAM_FRAME_RATE
        self._frame_period = 1.0 / TARGET_STREAM_FPS
        self._stream_quality = STREAM_JPEG_QUALITY

        self._hooks: List[FrameHook] = []
        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0

        self._run_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._picam: Optional[Picamera2] = None
        self._video_config = None
        self._still_config = None
        self._camera_started = False
        self._capture_lock = threading.Lock()
        self._rotation_warning_logged = False

        self._record_lock = threading.Lock()
        self._record_state: Optional[RecordingState] = None

    # ------------------------ Public API ---------------------------------- #
    def register_hook(self, hook: FrameHook) -> None:
        self._hooks.append(hook)

    def start(self) -> None:
        if not self._enabled:
            LOGGER.warning("Camera disabled via CAMERA_ENABLED=0; start() is a no-op.")
            return
        if Picamera2 is None:
            raise RuntimeError("picamera2 module not available; ensure it is installed on this device.")
        if self._run_event.is_set():
            return
        self._run_event.set()
        self._thread = threading.Thread(target=self._loop, name="PiCamera2CaptureThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._run_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._stop_recording()
        self._shutdown_camera()

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._frame_lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp

    def read(self) -> Optional[np.ndarray]:
        frame, _ = self.read_latest()
        return frame

    def is_running(self) -> bool:
        return self._run_event.is_set()

    def backend_name(self) -> str:
        return "picamera2" if Picamera2 is not None else "picamera2-unavailable"

    def start_recording(self, output_path: str, fps: Optional[float] = None, codec: str = "mp4v") -> str:
        if not output_path:
            raise ValueError("output_path must be provided for recording.")
        fps = fps or TARGET_STREAM_FPS
        with self._record_lock:
            if self._record_state is not None:
                raise RuntimeError("Recording already in progress.")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            with self._frame_lock:
                frame_shape = None if self._frame is None else self._frame.shape

            if frame_shape is not None and len(frame_shape) >= 2:
                size = (int(frame_shape[1]), int(frame_shape[0]))
            elif ROTATE_STREAM_90 and not ROTATE_STREAM_180:
                size = (self._video_size[1], self._video_size[0])
            else:
                size = self._video_size
            writer = cv2.VideoWriter(output_path, fourcc, fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
            self._record_state = RecordingState(writer=writer, path=output_path, fps=fps, started_at=time.time())
            LOGGER.info("Recording started: path=%s fps=%.2f size=%s", output_path, fps, size)
            return output_path

    def stop_recording(self) -> Optional[str]:
        return self._stop_recording()

    def is_recording(self) -> bool:
        with self._record_lock:
            return self._record_state is not None

    def recording_path(self) -> Optional[str]:
        with self._record_lock:
            return self._record_state.path if self._record_state else None

    # ------------------------ Internals ----------------------------------- #
    def _ensure_camera_started(self) -> Optional[Picamera2]:
        if not self._enabled:
            return None
        if Picamera2 is None:
            raise RuntimeError("picamera2 module not available on this system.")

        if self._picam is None or self._video_config is None:
            self._picam = Picamera2()
            controls_payload = {}
            if PICAM_AF_MODE is not None:
                controls_payload["AfMode"] = PICAM_AF_MODE
            if PICAM_AF_RANGE is not None:
                controls_payload["AfRange"] = PICAM_AF_RANGE
            if PICAM_AF_SPEED is not None:
                controls_payload["AfSpeed"] = PICAM_AF_SPEED
            self._video_config = self._picam.create_video_configuration(
                main={"size": self._video_size},
                controls={
                    "FrameRate": self._frame_rate,
                    **controls_payload,
                },
            )
            self._still_config = self._picam.create_still_configuration(
                main={"size": self._still_size},
                controls=controls_payload or None,
            )

        if not self._camera_started:
            try:
                self._picam.configure(self._video_config)
                self._picam.start()
                self._camera_started = True
                LOGGER.info(
                    "PiCamera2 started (%dx%d@%s)",
                    self._video_size[0],
                    self._video_size[1],
                    self._frame_rate,
                )
            except Exception as exc:
                LOGGER.error("Failed to start PiCamera2: %s", exc)
                self._shutdown_camera()
                raise

        return self._picam

    def _shutdown_camera(self) -> None:
        if self._picam is None:
            return
        try:
            if self._camera_started:
                self._picam.stop()
        except Exception:
            LOGGER.debug("Ignoring error while stopping camera", exc_info=True)
        try:
            self._picam.close()
        except Exception:
            LOGGER.debug("Ignoring error while closing camera", exc_info=True)
        finally:
            self._picam = None
            self._video_config = None
            self._still_config = None
            self._camera_started = False

    def _stop_recording(self) -> Optional[str]:
        state: Optional[RecordingState]
        with self._record_lock:
            state = self._record_state
            self._record_state = None
        if state is None:
            return None
        try:
            state.writer.release()
        except Exception:
            LOGGER.debug("Error releasing VideoWriter", exc_info=True)
        LOGGER.info("Recording stopped: path=%s duration=%.2fs", state.path, time.time() - state.started_at)
        return state.path

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        if ROTATE_STREAM_90 and ROTATE_STREAM_180 and not self._rotation_warning_logged:
            LOGGER.warning("Both ROTATE_STREAM_90 and ROTATE_STREAM_180 enabled; defaulting to 90-degree counter-clockwise rotation.")
            self._rotation_warning_logged = True

        if ROTATE_STREAM_90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ROTATE_STREAM_180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _run_hooks(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        output = frame
        for hook in self._hooks:
            try:
                result = hook(output, timestamp)
                if isinstance(result, np.ndarray):
                    output = result
            except Exception:
                LOGGER.exception("Frame hook error; continuing.")
        return output

    def _record_frame(self, frame: np.ndarray) -> None:
        with self._record_lock:
            state = self._record_state
        if state is None:
            return
        try:
            state.writer.write(frame)
        except Exception:
            LOGGER.exception("Failed to write frame to recorder; stopping recording.")
            self._stop_recording()

    def _loop(self) -> None:
        while self._run_event.is_set():
            try:
                cam = self._ensure_camera_started()
            except Exception:
                time.sleep(0.5)
                continue

            frame: Optional[np.ndarray] = None
            try:
                with self._capture_lock:
                    if cam is None:
                        raise RuntimeError("Camera not available")
                    frame = cam.capture_array("main")
            except Exception as exc:
                LOGGER.error("Capture failed: %s", exc)
                self._shutdown_camera()
                time.sleep(0.1)
                continue

            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            frame = self._normalize_color_order(frame)
            frame = self._apply_rotation(frame)

            timestamp = time.time()
            processed = self._run_hooks(frame, timestamp)

            with self._frame_lock:
                self._frame = processed
                self._timestamp = timestamp

            self._record_frame(processed)

            elapsed = time.time() - timestamp
            delay = self._frame_period - elapsed
            if delay > 0:
                time.sleep(delay)

        self._shutdown_camera()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.stop()
        except Exception:
            pass

    def _normalize_color_order(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert libcamera frames (typically RGB/RGBA) into the BGR layout OpenCV expects.
        """
        if frame.ndim != 3:
            return frame

        channels = frame.shape[2]
        try:
            if channels == 4:
                if PICAM_COLOR_SPACE == "RGB":
                    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                if PICAM_COLOR_SPACE == "BGR":
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame[:, :, :3]

            if channels >= 3:
                if PICAM_COLOR_SPACE == "RGB":
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if PICAM_COLOR_SPACE == "BGR":
                    return frame
        except Exception:
            LOGGER.debug("Failed to normalize frame color order", exc_info=True)

        return frame
