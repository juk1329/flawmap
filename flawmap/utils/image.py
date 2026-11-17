"""``cv2.imread`` / ``cv2.imwrite`` that also work on non-ASCII paths on Windows.

OpenCV's own file functions take narrow (ANSI code page) strings on Windows, so they fail silently
when the repository sits under a folder such as ``바탕 화면``. These read and write the bytes with
numpy and decode / encode them in memory, which behaves the same on every platform.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Like ``cv2.imread``: the decoded image, or None when the file is missing or unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """Like ``cv2.imwrite`` (format from the file extension); raises if the folder is missing."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True
