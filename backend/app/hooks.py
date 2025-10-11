"""
Placeholders for integrating custom frame processing / annotation logic.

Register your callable with ``camera.register_hook`` to receive every frame.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

# This module is intentionally lightweight. In your actual project you can
# import your ML model(s) here, warm them up during FastAPI startup, and invoke
# them inside ``annotate_frame``. The hook is invoked on the camera worker
# thread, so heavy processing should be offloaded to a queue if it risks
# blocking streaming.

def annotate_frame(frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
    """
    Stub for custom annotation logic.

    Return a modified frame to replace the original, or ``None`` to leave the
    stream untouched.
    """

    # Example (commented out):
    # detections = my_model.predict(frame)
    # annotated = draw_boxes(frame, detections)
    # return annotated

    return None

def sharpen_frame(frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
    """
    Lightweight unsharp mask to crisp up edges without changing framing.
    """
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5, sigmaY=1.5)
        return cv2.addWeighted(frame, 1.4, blurred, -0.4, 0)
    except Exception:
        return None

def rotate_frame(angle: int = 90):
    """
    Returns a hook that rotates frames by the specified angle (90, 180, or 270).
    """
    def _hook(frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return None
    return _hook
