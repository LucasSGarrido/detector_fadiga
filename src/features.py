from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Sequence


Point = tuple[float, float]

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (61, 13, 291, 14)

LANDMARKS_TO_DRAW = sorted(set(LEFT_EYE + RIGHT_EYE + MOUTH))


@dataclass(frozen=True)
class FaceFeatures:
    face_detected: bool
    ear_left: float | None = None
    ear_right: float | None = None
    ear_mean: float | None = None
    mar: float | None = None
    head_roll_deg: float | None = None

    @classmethod
    def missing(cls) -> "FaceFeatures":
        return cls(face_detected=False)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _get(points: Sequence[Point], index: int) -> Point:
    try:
        return points[index]
    except IndexError as exc:
        raise ValueError(f"Landmark index {index} is missing") from exc


def eye_aspect_ratio(points: Sequence[Point], indices: Sequence[int]) -> float:
    """Compute EAR using six eye landmarks."""
    if len(indices) != 6:
        raise ValueError("EAR requires six landmark indices")

    p1, p2, p3, p4, p5, p6 = (_get(points, idx) for idx in indices)
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(points: Sequence[Point], indices: Sequence[int] = MOUTH) -> float:
    """Compute a compact mouth opening ratio from four mouth landmarks."""
    left, top, right, bottom = (_get(points, idx) for idx in indices)
    horizontal = distance(left, right)

    if horizontal == 0:
        return 0.0

    return distance(top, bottom) / horizontal


def head_roll(points: Sequence[Point]) -> float:
    """Estimate face roll angle from the line between both eye centers."""
    left_center = _center(points, LEFT_EYE)
    right_center = _center(points, RIGHT_EYE)
    return degrees(atan2(right_center[1] - left_center[1], right_center[0] - left_center[0]))


def _center(points: Sequence[Point], indices: Sequence[int]) -> Point:
    selected = [_get(points, idx) for idx in indices]
    return (
        sum(point[0] for point in selected) / len(selected),
        sum(point[1] for point in selected) / len(selected),
    )


def extract_face_features(points: Sequence[Point] | None) -> FaceFeatures:
    if not points:
        return FaceFeatures.missing()

    ear_left = eye_aspect_ratio(points, LEFT_EYE)
    ear_right = eye_aspect_ratio(points, RIGHT_EYE)
    ear_mean = (ear_left + ear_right) / 2.0

    return FaceFeatures(
        face_detected=True,
        ear_left=ear_left,
        ear_right=ear_right,
        ear_mean=ear_mean,
        mar=mouth_aspect_ratio(points),
        head_roll_deg=head_roll(points),
    )
