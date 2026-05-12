import numpy as np

from src.fatigue_rules import FatigueResult
from src.features import FaceFeatures
from src.visualization import draw_overlay


def test_overlay_keeps_face_center_clear_when_debug_is_enabled():
    frame = np.zeros((540, 320, 3), dtype=np.uint8)
    features = FaceFeatures(
        face_detected=True,
        ear_mean=0.18,
        mar=0.12,
        head_roll_deg=1.5,
    )
    result = FatigueResult(
        state="Atencao",
        score=63.7,
        alert_triggered=False,
        perclos=0.45,
        valid_frames_ratio=1.0,
        blink_count=4,
        long_blink_count=2,
        yawn_count=0,
        current_eye_closed_seconds=1.1,
    )

    draw_overlay(
        frame,
        points=None,
        features=features,
        result=result,
        fps=120.0,
        latency_ms=5.0,
        show_landmarks=False,
        show_debug_panel=True,
    )

    center_strip = frame[120:400, 40:280]
    assert not center_strip.any()
