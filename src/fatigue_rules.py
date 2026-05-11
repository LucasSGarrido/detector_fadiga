from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from .features import FaceFeatures


@dataclass(frozen=True)
class FatigueConfig:
    window_seconds: float = 5.0
    min_valid_frames_ratio: float = 0.60
    ear_closed: float = 0.21
    mar_yawn: float = 0.60
    perclos_attention: float = 0.25
    perclos_fatigue: float = 0.40
    long_blink_seconds: float = 0.70
    yawn_min_seconds: float = 1.00
    head_roll_warning_deg: float = 18.0
    fatigue_score_attention: float = 35.0
    fatigue_score_alarm: float = 65.0
    alert_min_duration_seconds: float = 2.0
    alert_cooldown_seconds: float = 5.0


@dataclass(frozen=True)
class FrameObservation:
    timestamp: float
    face_detected: bool
    ear_mean: float | None
    mar: float | None
    head_roll_deg: float | None

    @property
    def valid(self) -> bool:
        return self.face_detected and self.ear_mean is not None and self.mar is not None


@dataclass(frozen=True)
class FatigueResult:
    state: str
    score: float
    alert_triggered: bool
    perclos: float
    valid_frames_ratio: float
    blink_count: int
    long_blink_count: int
    yawn_count: int
    current_eye_closed_seconds: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class FatigueEstimator:
    def __init__(self, config: FatigueConfig | None = None) -> None:
        self.config = config or FatigueConfig()
        self._window: Deque[FrameObservation] = deque()
        self._high_score_since: float | None = None
        self._last_alert_at: float | None = None

    def update(self, features: FaceFeatures, timestamp: float) -> FatigueResult:
        observation = FrameObservation(
            timestamp=timestamp,
            face_detected=features.face_detected,
            ear_mean=features.ear_mean,
            mar=features.mar,
            head_roll_deg=features.head_roll_deg,
        )
        self._window.append(observation)
        self._trim(timestamp)

        return self._evaluate(timestamp)

    def _trim(self, timestamp: float) -> None:
        min_timestamp = timestamp - self.config.window_seconds
        while self._window and self._window[0].timestamp < min_timestamp:
            self._window.popleft()

    def _evaluate(self, timestamp: float) -> FatigueResult:
        observations = list(self._window)
        total = len(observations)

        if total == 0:
            return self._empty_result("Rosto ausente")

        valid = [item for item in observations if item.valid]
        valid_ratio = len(valid) / total

        if valid_ratio < self.config.min_valid_frames_ratio:
            self._high_score_since = None
            return FatigueResult(
                state="Rosto ausente",
                score=0.0,
                alert_triggered=False,
                perclos=0.0,
                valid_frames_ratio=valid_ratio,
                blink_count=0,
                long_blink_count=0,
                yawn_count=0,
                current_eye_closed_seconds=0.0,
                reasons=("Rosto detectado em poucos frames da janela",),
            )

        closed_flags = [self._is_eye_closed(item) for item in valid]
        yawn_flags = [self._is_yawn(item) for item in valid]
        perclos = sum(closed_flags) / len(closed_flags)
        blink_runs = self._count_runs(valid, closed_flags)
        yawn_runs = self._count_runs(valid, yawn_flags)
        blink_count = len(blink_runs)
        long_blink_count = sum(
            1 for start, end in blink_runs if end - start >= self.config.long_blink_seconds
        )
        yawn_count = sum(1 for start, end in yawn_runs if end - start >= self.config.yawn_min_seconds)
        current_eye_closed_seconds = self._current_closed_duration(timestamp, valid)
        roll_warning = self._roll_warning(valid)

        score, reasons = self._score(
            perclos=perclos,
            yawn_count=yawn_count,
            long_blink_count=long_blink_count,
            current_eye_closed_seconds=current_eye_closed_seconds,
            roll_warning=roll_warning,
        )
        state = self._state_from_score(score)
        alert_triggered = self._update_alert_state(timestamp, score)

        return FatigueResult(
            state=state,
            score=score,
            alert_triggered=alert_triggered,
            perclos=perclos,
            valid_frames_ratio=valid_ratio,
            blink_count=blink_count,
            long_blink_count=long_blink_count,
            yawn_count=yawn_count,
            current_eye_closed_seconds=current_eye_closed_seconds,
            reasons=tuple(reasons),
        )

    def _empty_result(self, state: str) -> FatigueResult:
        return FatigueResult(
            state=state,
            score=0.0,
            alert_triggered=False,
            perclos=0.0,
            valid_frames_ratio=0.0,
            blink_count=0,
            long_blink_count=0,
            yawn_count=0,
            current_eye_closed_seconds=0.0,
        )

    def _is_eye_closed(self, item: FrameObservation) -> bool:
        return item.ear_mean is not None and item.ear_mean < self.config.ear_closed

    def _is_yawn(self, item: FrameObservation) -> bool:
        return item.mar is not None and item.mar > self.config.mar_yawn

    @staticmethod
    def _count_runs(
        observations: list[FrameObservation],
        flags: list[bool],
    ) -> list[tuple[float, float]]:
        runs: list[tuple[float, float]] = []
        start: float | None = None
        last_true: float | None = None

        for observation, flag in zip(observations, flags):
            if flag and start is None:
                start = observation.timestamp
            if flag:
                last_true = observation.timestamp
            if not flag and start is not None and last_true is not None:
                runs.append((start, last_true))
                start = None
                last_true = None

        if start is not None and last_true is not None:
            runs.append((start, last_true))

        return runs

    def _current_closed_duration(self, timestamp: float, valid: list[FrameObservation]) -> float:
        if not valid or not self._is_eye_closed(valid[-1]):
            return 0.0

        start = valid[-1].timestamp
        for item in reversed(valid):
            if not self._is_eye_closed(item):
                break
            start = item.timestamp

        return max(0.0, timestamp - start)

    def _roll_warning(self, valid: list[FrameObservation]) -> bool:
        values = [
            abs(item.head_roll_deg)
            for item in valid
            if item.head_roll_deg is not None
        ]
        if not values:
            return False

        avg_roll = sum(values) / len(values)
        return avg_roll >= self.config.head_roll_warning_deg

    def _score(
        self,
        *,
        perclos: float,
        yawn_count: int,
        long_blink_count: int,
        current_eye_closed_seconds: float,
        roll_warning: bool,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        perclos_component = min(perclos / self.config.perclos_fatigue, 1.0) * 35.0
        score += perclos_component
        if perclos >= self.config.perclos_attention:
            reasons.append(f"PERCLOS alto ({perclos:.2f})")

        eye_duration_component = min(
            current_eye_closed_seconds / self.config.alert_min_duration_seconds,
            1.0,
        ) * 25.0
        score += eye_duration_component
        if current_eye_closed_seconds > 0:
            reasons.append(f"Olhos fechados por {current_eye_closed_seconds:.1f}s")

        if yawn_count:
            score += min(yawn_count, 2) * 10.0
            reasons.append(f"Bocejos detectados: {yawn_count}")

        if long_blink_count:
            score += min(long_blink_count, 2) * 7.5
            reasons.append(f"Piscadas longas: {long_blink_count}")

        if roll_warning:
            score += 10.0
            reasons.append("Inclinacao da cabeca elevada")

        score = max(0.0, min(100.0, score))
        if not reasons:
            reasons.append("Sinais dentro do padrao da janela")

        return score, reasons

    def _state_from_score(self, score: float) -> str:
        if score >= self.config.fatigue_score_alarm:
            return "Fadiga"
        if score >= self.config.fatigue_score_attention:
            return "Atencao"
        return "Atento"

    def _update_alert_state(self, timestamp: float, score: float) -> bool:
        if score < self.config.fatigue_score_alarm:
            self._high_score_since = None
            return False

        if self._high_score_since is None:
            self._high_score_since = timestamp
            return False

        high_score_duration = timestamp - self._high_score_since
        if high_score_duration < self.config.alert_min_duration_seconds:
            return False

        if self._last_alert_at is None:
            self._last_alert_at = timestamp
            return True

        if timestamp - self._last_alert_at >= self.config.alert_cooldown_seconds:
            self._last_alert_at = timestamp
            return True

        return False
