from src.fatigue_rules import FatigueConfig, FatigueEstimator
from src.features import FaceFeatures


def test_estimator_stays_attentive_with_open_eyes():
    estimator = FatigueEstimator(FatigueConfig(window_seconds=5.0))

    result = None
    for index in range(20):
        result = estimator.update(
            FaceFeatures(face_detected=True, ear_mean=0.30, mar=0.20, head_roll_deg=0.0),
            timestamp=index * 0.1,
        )

    assert result is not None
    assert result.state == "Atento"
    assert result.score < 35.0


def test_estimator_reaches_fatigue_after_persistent_closed_eyes():
    estimator = FatigueEstimator(
        FatigueConfig(
            window_seconds=5.0,
            ear_closed=0.21,
            fatigue_score_alarm=65.0,
            alert_min_duration_seconds=1.0,
            alert_cooldown_seconds=5.0,
        )
    )

    result = None
    for index in range(40):
        result = estimator.update(
            FaceFeatures(face_detected=True, ear_mean=0.10, mar=0.20, head_roll_deg=0.0),
            timestamp=index * 0.1,
        )

    assert result is not None
    assert result.state == "Fadiga"
    assert result.score >= 65.0


def test_estimator_marks_missing_face_when_detection_is_unstable():
    estimator = FatigueEstimator(FatigueConfig(window_seconds=5.0, min_valid_frames_ratio=0.8))

    result = None
    for index in range(10):
        result = estimator.update(FaceFeatures.missing(), timestamp=index * 0.1)

    assert result is not None
    assert result.state == "Rosto ausente"
