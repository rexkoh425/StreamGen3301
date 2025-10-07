"""
Placeholders for integrating custom frame processing / annotation logic.

Register your callable with ``camera.register_hook`` to receive every frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

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
