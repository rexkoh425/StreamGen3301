"""Utility helpers to encode numpy frames into JPEG bytes."""

from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)
_FRAME_LOG_LIMIT = 3
_frame_log_counts: Dict[str, int] = {}


def _log_frame_debug(
    stage: str, frame: Optional[np.ndarray], note: str, mode_hint: str | None = None
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return

    count = _frame_log_counts.get(stage, 0)
    if count >= _FRAME_LOG_LIMIT:
        return
    _frame_log_counts[stage] = count + 1

    if frame is None:
        LOGGER.debug(
            "Encoding[%s #%d]: frame=None note=%s mode_hint=%s",
            stage,
            count + 1,
            note,
            mode_hint,
        )
        return

    if frame.size == 0:
        LOGGER.debug(
            "Encoding[%s #%d]: empty frame shape=%s dtype=%s note=%s mode_hint=%s",
            stage,
            count + 1,
            frame.shape,
            frame.dtype,
            note,
            mode_hint,
        )
        return

    shape = frame.shape
    dtype = frame.dtype
    channels = 1
    sample_value: object = None
    try:
        if frame.ndim == 2:
            channels = 1
            sample_value = int(frame[0, 0])
        elif frame.ndim >= 3:
            channels = shape[2]
            sample = frame[0, 0]
            if isinstance(sample, np.ndarray):
                subset = sample.flatten()[: min(3, channels)]
                sample_value = [int(x) for x in subset]
            else:
                sample_value = int(sample)
        elif frame.ndim == 1:
            channels = 1
            sample_value = int(frame[0])
        else:
            channels = shape[-1] if shape else 0
    except Exception:
        sample_value = "unavailable"

    LOGGER.debug(
        "Encoding[%s #%d]: shape=%s dtype=%s channels=%d sample=%s note=%s mode_hint=%s",
        stage,
        count + 1,
        shape,
        dtype,
        channels,
        sample_value,
        note,
        mode_hint,
    )

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

    _log_frame_debug(
        "encode_input", frame, "Input to encode_frame_to_jpeg", "BGR expected"
    )
    frame, mode = _normalize_frame_channels(frame)
    _log_frame_debug(
        "encode_normalized",
        frame,
        "Normalized for encoder consumption",
        f"{mode} (post-normalization)",
    )

    if _CV2_AVAILABLE:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        LOGGER.debug(
            "Attempting cv2.imencode; mode=%s quality=%d shape=%s",
            mode,
            quality,
            frame.shape,
        )
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if success:
            LOGGER.debug(
                "cv2.imencode succeeded; output_size=%d bytes assumed_mode=%s",
                buffer.size,
                mode,
            )
            return buffer.tobytes()
        LOGGER.debug("cv2.imencode failed; falling back to PIL if available")

    if _PIL_AVAILABLE:
        LOGGER.debug("Falling back to PIL.Image for JPEG encoding; mode=%s", mode)
        rgb_frame = _to_rgb_for_pil(frame, mode)
        pil_mode = "L" if mode == "L" else "RGB"
        _log_frame_debug(
            "pil_bridge_input",
            rgb_frame,
            "Frame after conversion for PIL encoder",
            f"{pil_mode} (PIL mode)",
        )
        image = Image.fromarray(rgb_frame, mode=pil_mode)
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=quality)
            LOGGER.debug(
                "PIL.Image save completed; output_size=%d bytes mode=%s",
                output.tell(),
                pil_mode,
            )
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
        trimmed = frame[:, :, :3]
        _log_frame_debug(
            "normalize_drop_alpha",
            trimmed,
            "Dropped extra channel during normalization",
            "BGR",
        )
        return trimmed, "BGR"
    raise ValueError(f"Unsupported channel count for JPEG encoding: {channels}")


def _to_rgb_for_pil(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "L":
        return frame
    if mode == "BGR":
        return frame[:, :, ::-1]
    raise ValueError(f"Unsupported PIL mode: {mode}")