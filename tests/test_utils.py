from src.utils import build_fatigue_config, load_config


def test_load_default_config():
    config = load_config()

    assert config["video"]["source"] == "webcam"
    assert config["thresholds"]["ear_closed"] == 0.21


def test_build_fatigue_config_from_default():
    fatigue_config = build_fatigue_config(load_config())

    assert fatigue_config.window_seconds == 5.0
    assert fatigue_config.ear_closed == 0.21
