import numpy as np

from src.pipeline import prepare_video_frame, resize_frame, resolve_timestamp_mode
from src.utils import load_config


def test_resize_frame_preserves_aspect_ratio_for_vertical_video():
    config = load_config()
    frame = np.zeros((850, 478, 3), dtype=np.uint8)

    resized = resize_frame(frame, config)

    assert resized.shape[0] == 540
    assert resized.shape[1] == 303


def test_resize_frame_can_force_target_dimensions():
    config = load_config()
    config["video"]["preserve_aspect_ratio"] = False
    frame = np.zeros((850, 478, 3), dtype=np.uint8)

    resized = resize_frame(frame, config)

    assert resized.shape[:2] == (540, 960)


def test_resolve_timestamp_mode_auto_uses_real_for_webcam_and_video_for_files():
    assert resolve_timestamp_mode("webcam", "auto") == "real"
    assert resolve_timestamp_mode("video.mp4", "auto") == "video"
    assert resolve_timestamp_mode("webcam", "video") == "video"


def test_prepare_video_frame_uses_even_dimensions():
    frame = np.zeros((541, 303, 3), dtype=np.uint8)

    prepared = prepare_video_frame(frame)

    assert prepared.shape[:2] == (540, 302)
