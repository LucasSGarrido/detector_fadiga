from __future__ import annotations

from pathlib import Path


def is_webcam_source(source: str) -> bool:
    return source.lower() == "webcam"


def resolve_source(source: str, camera_index: int = 0) -> int | str:
    if is_webcam_source(source):
        return camera_index

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    return str(path)


def open_capture(source: str, camera_index: int = 0):
    import cv2

    resolved = resolve_source(source, camera_index)
    capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return capture
