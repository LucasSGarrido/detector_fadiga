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

    cv2.rectangle(frame, (0, 0), (width, 92), (20, 20, 20), thickness=-1)
    cv2.putText(
        frame,
        f"Estado: {result.state}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Score: {result.score:05.1f} | PERCLOS: {result.perclos:.2f}",
        (18, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    if result.alert_triggered:
        cv2.rectangle(frame, (0, height - 64), (width, height), (30, 30, 200), thickness=-1)
        cv2.putText(
            frame,
            "ALERTA DE FADIGA",
            (18, height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
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
        f"EAR: {_fmt(features.ear_mean)}",
        f"MAR: {_fmt(features.mar)}",
        f"Roll: {_fmt(features.head_roll_deg)} deg",
        f"Piscadas: {result.blink_count} | Longas: {result.long_blink_count}",
        f"Bocejos: {result.yawn_count}",
        f"Olhos fechados: {result.current_eye_closed_seconds:.1f}s",
        f"Face valida: {result.valid_frames_ratio:.0%}",
        f"FPS: {fps:.1f} | Lat: {latency_ms:.1f}ms",
    ]

    x0, y0, width, line_height = 14, 108, 330, 24
    cv2.rectangle(
        frame,
        (x0 - 8, y0 - 22),
        (x0 + width, y0 + line_height * len(lines) + 8),
        (35, 35, 35),
        thickness=-1,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x0, y0 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"
