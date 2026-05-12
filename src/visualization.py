from __future__ import annotations

from .fatigue_rules import FatigueResult
from .features import LANDMARKS_TO_DRAW, FaceFeatures, Point


STATE_COLORS = {
    "Atento": (40, 180, 80),
    "Atencao": (0, 190, 255),
    "Fadiga": (40, 40, 220),
    "Rosto ausente": (150, 150, 150),
}


def draw_overlay(
    frame,
    *,
    points: list[Point] | None,
    features: FaceFeatures,
    result: FatigueResult,
    fps: float,
    latency_ms: float,
    show_landmarks: bool = True,
    show_debug_panel: bool = True,
):
    import cv2

    color = STATE_COLORS.get(result.state, (255, 255, 255))
    height, width = frame.shape[:2]
    font_scale = max(0.42, min(0.68, width / 900))
    small_scale = max(0.36, min(0.50, width / 1050))

    header_height = 52
    _fill_rect_alpha(frame, (0, 0), (width, header_height), (16, 18, 18), 0.68)
    cv2.putText(
        frame,
        f"Estado: {result.state}  |  Score: {result.score:04.1f}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"PERCLOS {result.perclos:.2f}  |  Face {result.valid_frames_ratio:.0%}",
        (14, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    if result.alert_triggered:
        _fill_rect_alpha(frame, (0, height - 44), (width, height), (30, 30, 200), 0.78)
        cv2.putText(
            frame,
            "ALERTA DE FADIGA",
            (14, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.58, font_scale),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if show_landmarks and points:
        for index in LANDMARKS_TO_DRAW:
            x, y = points[index]
            cv2.circle(frame, (int(x), int(y)), 2, (255, 220, 120), thickness=-1)

    if show_debug_panel:
        _draw_debug_panel(frame, features, result, fps, latency_ms)

    return frame


def _draw_debug_panel(frame, features: FaceFeatures, result: FatigueResult, fps: float, latency_ms: float):
    import cv2

    lines = [
        f"EAR {_fmt(features.ear_mean)}   MAR {_fmt(features.mar)}   Roll {_fmt(features.head_roll_deg)}",
        f"Piscadas {result.blink_count}   Longas {result.long_blink_count}   Bocejos {result.yawn_count}",
        f"Olhos fechados {result.current_eye_closed_seconds:.1f}s   FPS {fps:.1f}   Lat {latency_ms:.1f}ms",
    ]

    height, width = frame.shape[:2]
    line_height = max(18, int(height * 0.035))
    panel_height = line_height * len(lines) + 16
    y0 = max(68, height - panel_height + 2)
    font_scale = max(0.34, min(0.48, width / 1100))
    _fill_rect_alpha(frame, (0, height - panel_height), (width, height), (24, 28, 26), 0.58)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, y0 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def _fill_rect_alpha(frame, top_left: tuple[int, int], bottom_right: tuple[int, int], color, alpha: float) -> None:
    import cv2

    x0, y0 = top_left
    x1, y1 = bottom_right
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"
