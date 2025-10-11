"""
USB camera stream using OpenCV (V4L2) with MJPEG preferred.
Threaded grabber with hook support and auto-reopen on failure.
"""

from __future__ import annotations
import logging, threading, time
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np
import cv2

LOGGER = logging.getLogger(__name__)
FrameHook = Callable[[np.ndarray, float], Optional[np.ndarray]]

class CameraStream:
    def __init__(
        self,
        device_index: int = 1,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        prefer_mjpeg: bool = True,
        reopen_on_fail: bool = True,
        autofocus: Optional[int] = None,   # 0=off, 1=on, None=leave default
        exposure: Optional[float] = None,  # driver-specific scale; None=leave default
    ) -> None:
        self._idx = int(device_index)
        self._w, self._h = int(width), int(height)
        self._fps = float(target_fps)
        self._prefer_mjpeg = bool(prefer_mjpeg)
        self._reopen = bool(reopen_on_fail)
        self._autofocus = autofocus
        self._exposure = exposure

        self._cap: Optional[cv2.VideoCapture] = None
        self._run = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._hooks: List[FrameHook] = []

        self._lk = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0

        self._dbg_counts: Dict[str, int] = {}
        self._dbg_limit = 3

    # -------- Public API --------
    def register_hook(self, hook: FrameHook) -> None:
        self._hooks.append(hook)

    def start(self) -> None:
        if self._run.is_set():
            return
        self._run.set()
        self._open()
        self._thr = threading.Thread(target=self._loop, name="CameraCaptureThread", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        if not self._run.is_set():
            return
        self._run.clear()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=2.0)
        self._thr = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lk:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    def read(self) -> Optional[np.ndarray]:
        frame, _ = self.read_latest()
        return frame

    def is_running(self) -> bool:
        return self._run.is_set()

    def backend_name(self) -> str:
        return "opencv-v4l2"

    # -------- Internals --------
    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self._idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.open(self._idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{self._idx}")

        # Prefer MJPEG for USB bandwidth; fallback to YUYV if MJPG unsupported
        if self._prefer_mjpeg:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Resolution / FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._w))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._h))
        self._cap.set(cv2.CAP_PROP_FPS, float(self._fps))

        # Try to disable driver-side buffering so we always grab the freshest frame
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                LOGGER.debug(
                    "Unable to set capture buffer size; continuing.", exc_info=True
                )

        # Optional controls (best-effort; drivers differ)
        if self._autofocus is not None:
            try:
                self._cap.set(cv2.CAP_PROP_AUTOFOCUS, int(self._autofocus))
            except Exception:
                LOGGER.debug("Unable to set autofocus, continuing.", exc_info=True)
        if self._exposure is not None:
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(self._exposure))
            except Exception:
                LOGGER.debug("Unable to set exposure, continuing.", exc_info=True)

    def _reopen_if_needed(self) -> None:
        if not self._reopen:
            return
        try:
            if self._cap is None or not self._cap.isOpened():
                self.stop()
                self._open()
        except Exception as e:
            LOGGER.exception("Reopen failed: %s", e)

    def _loop(self) -> None:
        interval = 1.0 / self._fps if self._fps > 0 else 0.0
        misses = 0
        while self._run.is_set():
            ts = time.time()
            ok, frame = (False, None) if self._cap is None else self._cap.read()
            if not ok or frame is None:
                misses += 1
                if self._reopen and misses >= 30:
                    self._reopen_if_needed()
                    misses = 0
                time.sleep(0.05)
                continue
            misses = 0

            # Hook pipeline
            out = frame
            for hook in self._hooks:
                try:
                    res = hook(out.copy(), ts)
                    if isinstance(res, np.ndarray):
                        out = res
                except Exception:
                    LOGGER.exception("Hook error; continuing.")
                    continue

            # 💡 Only keep the newest frame (overwrite immediately)
            with self._lk:
                self._frame = out
                self._ts = ts

            # Sleep only if needed; otherwise, we always read next frame
            if interval:
                dt = time.time() - ts
                delay = max(0.0, interval - dt)
                if delay > 0:
                    time.sleep(delay)


    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
