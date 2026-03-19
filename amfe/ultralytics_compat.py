"""Ultralytics compatibility helpers.

This project reuses the Ultralytics Detect head and detection loss, but some
headless environments ship an OpenCV wheel that fails to load because GUI
libraries such as ``libGL`` are unavailable. To keep Phase D testable without
redesigning Ultralytics internals, this module installs a tiny ``cv2`` shim only
when importing the real package fails for that specific reason.

The shim is intentionally minimal: it only provides the attributes Ultralytics
needs during model construction, loss computation, and lightweight smoke tests.
If a later workflow needs richer OpenCV functionality, users should install a
working OpenCV build (for example ``opencv-python-headless``).
"""

from __future__ import annotations

import sys
import types
from typing import Any


_LIBGL_ERROR_FRAGMENT = "libGL.so.1"


def _install_cv2_stub() -> None:
    """Install a conservative ``cv2`` stub for headless smoke-test environments."""

    if "cv2" in sys.modules:
        return

    cv2 = types.ModuleType("cv2")
    cv2.setNumThreads = lambda _threads=0: None
    cv2.resize = lambda image, dsize, interpolation=None: image
    cv2.copyMakeBorder = lambda image, top, bottom, left, right, borderType, value=0: image
    cv2.getTextSize = lambda text, fontFace, fontScale, thickness: ((max(len(text), 1) * 10, 20), 0)
    cv2.putText = lambda *args, **kwargs: None
    cv2.rectangle = lambda *args, **kwargs: None
    cv2.cvtColor = lambda image, code: image
    cv2.imread = lambda *args, **kwargs: None
    cv2.imwrite = lambda *args, **kwargs: True
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: 0
    cv2.destroyAllWindows = lambda *args, **kwargs: None

    constants: dict[str, Any] = {
        "INTER_LINEAR": 1,
        "INTER_NEAREST": 0,
        "BORDER_CONSTANT": 0,
        "BORDER_REFLECT_101": 4,
        "BORDER_TRANSPARENT": 5,
        "FONT_HERSHEY_SIMPLEX": 0,
        "LINE_AA": 16,
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "COLOR_BGR2RGB": 4,
        "COLOR_RGB2BGR": 4,
    }
    for name, value in constants.items():
        setattr(cv2, name, value)

    sys.modules["cv2"] = cv2


try:
    from ultralytics.nn.modules import Detect
    from ultralytics.utils.loss import v8DetectionLoss
except ImportError as exc:  # pragma: no cover - exercised in dependency-limited environments.
    if _LIBGL_ERROR_FRAGMENT not in str(exc):
        raise
    _install_cv2_stub()
    from ultralytics.nn.modules import Detect
    from ultralytics.utils.loss import v8DetectionLoss


__all__ = ["Detect", "v8DetectionLoss"]
