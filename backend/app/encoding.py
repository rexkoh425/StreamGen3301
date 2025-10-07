"""Utility helpers to encode numpy frames into JPEG bytes."""

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    _PIL_AVAILABLE = False


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> Optional[bytes]:
    """Encode a numpy frame into JPEG bytes."""
    if frame is None:
        return None

    frame, mode = _normalize_frame_channels(frame)

    if _CV2_AVAILABLE:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if success:
            return buffer.tobytes()

    if _PIL_AVAILABLE:
        rgb_frame = _to_rgb_for_pil(frame, mode)
        pil_mode = "L" if mode == "L" else "RGB"
        image = Image.fromarray(rgb_frame, mode=pil_mode)
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=quality)
            return output.getvalue()

    return None


def _normalize_frame_channels(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Normalize to OpenCV-friendly layout. We treat 3-ch as BGR (OpenCV default).
    Returns (frame_in_BGR_or_gray, mode_for_PIL_bridge)
    """
    if frame.ndim == 2:
        return frame, "L"
    if frame.ndim != 3:
        raise ValueError("Unsupported frame shape for JPEG encoding.")
    channels = frame.shape[2]
    if channels == 3:
        return frame, "BGR"
    if channels == 4:
        # Drop alpha/filler channel, keep first three components as BGR for OpenCV.
        return frame[:, :, :3], "BGR"
    raise ValueError(f"Unsupported channel count for JPEG encoding: {channels}")


def _to_rgb_for_pil(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "L":
        return frame
    if mode == "BGR":
        return frame[:, :, ::-1]
    raise ValueError(f"Unsupported PIL mode: {mode}")
