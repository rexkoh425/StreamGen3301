"""
Camera stream management for Raspberry Pi devices.

This module provides a small abstraction that hides whether frames
come from a PiCamera2 pipeline or a generic OpenCV ``VideoCapture``.
It captures frames on a background thread so the FastAPI layer can
serve them without blocking.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

try:
    from picamera2 import Picamera2  # type: ignore

    _PICAMERA_AVAILABLE = True
except ImportError:  # pragma: no cover - handled dynamically at runtime
    Picamera2 = None  # type: ignore
    _PICAMERA_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - handled dynamically at runtime
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False


FrameHook = Callable[[np.ndarray, float], None]


class CameraStream:
    """Continuously grabs frames from the configured camera source."""

    def __init__(
        self,
        use_picamera: bool = True,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[float] = 20.0,
    ) -> None:
        if use_picamera and not _PICAMERA_AVAILABLE:
            raise RuntimeError(
                "PiCamera2 was requested but the library is not installed. "
                "Install it or pass use_picamera=False to fallback to OpenCV."
            )
        if not use_picamera and not _CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV fallback requested but cv2 is not available. "
                "Install opencv-python or enable PiCamera2."
            )

        self._use_picamera = use_picamera and _PICAMERA_AVAILABLE
        self._device_index = device_index
        self._width = width or 1536
        self._height = height or 864
        self._target_fps = target_fps or 20.0

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._hooks: List[FrameHook] = []

        self._run_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._picam: Optional[Picamera2] = None
        self._capture = None
        self._backend_name = "picamera2" if self._use_picamera else "opencv"

    # Public API -----------------------------------------------------------------
    def register_hook(self, hook: FrameHook) -> None:
        """Register a callable that will receive frames and timestamps."""
        self._hooks.append(hook)

    def start(self) -> None:
        """Start the capture thread if not already running."""
        if self._run_event.is_set():
            return

        self._run_event.set()
        if self._use_picamera:
            self._start_picamera()
        else:
            self._start_opencv()

        self._thread = threading.Thread(
            target=self._capture_loop, name="CameraCaptureThread", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing frames."""
        run_event = getattr(self, "_run_event", None)
        if run_event is None or not run_event.is_set():
            return

        run_event.clear()
        thread = getattr(self, "_thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

        use_picamera = getattr(self, "_use_picamera", False)
        if use_picamera:
            picam = getattr(self, "_picam", None)
            if picam:
                picam.stop()
                picam.close()
            self._picam = None
        else:
            capture = getattr(self, "_capture", None)
            if capture:
                capture.release()
            self._capture = None

    def read(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame, or None if unavailable."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_running(self) -> bool:
        run_event = getattr(self, "_run_event", None)
        return bool(run_event and run_event.is_set())

    def backend_name(self) -> str:
        return getattr(self, "_backend_name", "unknown")

    # Internals ------------------------------------------------------------------
    def _start_picamera(self) -> None:
        assert Picamera2 is not None  # for type checkers
        self._picam = Picamera2()

        video_config = self._picam.create_video_configuration(
            main={
                "size": (
                    self._width,
                    self._height,
                )
            },
            buffer_count=4,
        )
        self._picam.configure(video_config)
        if self._target_fps:
            try:
                self._picam.set_controls({"FrameRate": float(self._target_fps)})
            except Exception:
                # Some sensor modes may not support the requested FPS; ignore.
                pass
        self._picam.start()

    def _start_opencv(self) -> None:
        assert cv2 is not None  # for type checkers
        self._capture = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.open(self._device_index)

        if self._width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        if self._target_fps:
            self._capture.set(cv2.CAP_PROP_FPS, float(self._target_fps))

    def _capture_loop(self) -> None:
        interval = 1.0 / self._target_fps if self._target_fps else 0.0
        while self._run_event.is_set():
            frame, timestamp = self._grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_to_store = frame
            for hook in self._hooks:
                try:
                    result = hook(frame_to_store.copy(), timestamp)
                    if isinstance(result, np.ndarray):
                        frame_to_store = result
                except Exception:  # pragma: no cover - defensive
                    # Hook exceptions should not kill the capture loop.
                    continue

            with self._frame_lock:
                self._latest_frame = frame_to_store
                self._latest_timestamp = timestamp

            if interval:
                elapsed = time.time() - timestamp
                delay = interval - elapsed
                if delay > 0:
                    time.sleep(delay)

    def _grab_frame(self) -> tuple[Optional[np.ndarray], float]:
        timestamp = time.time()
        if self._use_picamera and self._picam:
            frame = self._picam.capture_array()
            # PiCamera2 returns RGB frames by default; normalize to BGR for OpenCV.
            if frame.ndim == 3 and frame.shape[2] == 3:
                if _CV2_AVAILABLE and cv2 is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = frame[:, :, ::-1]
            return frame, timestamp
        if self._capture is None:
            return None, timestamp
        ok, frame = self._capture.read()
        if not ok:
            return None, timestamp
        return frame, timestamp

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            # Suppress exceptions in GC context; resources are already gone.
            pass
