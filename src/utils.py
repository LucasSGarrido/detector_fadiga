from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .fatigue_rules import FatigueConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "video": {
        "source": "webcam",
        "camera_index": 0,
        "target_width": 960,
        "target_height": 540,
        "preserve_aspect_ratio": True,
    },
    "face_mesh": {
        "max_num_faces": 1,
        "refine_landmarks": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "window": {
        "seconds": 5.0,
        "min_valid_frames_ratio": 0.60,
    },
    "thresholds": {
        "ear_closed": 0.21,
        "mar_yawn": 0.60,
        "perclos_attention": 0.25,
        "perclos_fatigue": 0.40,
        "long_blink_seconds": 0.70,
        "yawn_min_seconds": 1.00,
        "head_roll_warning_deg": 18.0,
        "fatigue_score_attention": 35.0,
        "fatigue_score_alarm": 65.0,
    },
    "alert": {
        "min_duration_seconds": 2.0,
        "cooldown_seconds": 5.0,
        "sound_enabled": True,
    },
    "ui": {
        "show_landmarks": True,
        "show_fps": True,
        "show_debug_panel": False,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files") from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    return deep_merge(DEFAULT_CONFIG, loaded)


def build_fatigue_config(config: dict[str, Any]) -> FatigueConfig:
    window = config["window"]
    thresholds = config["thresholds"]
    alert = config["alert"]

    return FatigueConfig(
        window_seconds=float(window["seconds"]),
        min_valid_frames_ratio=float(window["min_valid_frames_ratio"]),
        ear_closed=float(thresholds["ear_closed"]),
        mar_yawn=float(thresholds["mar_yawn"]),
        perclos_attention=float(thresholds["perclos_attention"]),
        perclos_fatigue=float(thresholds["perclos_fatigue"]),
        long_blink_seconds=float(thresholds["long_blink_seconds"]),
        yawn_min_seconds=float(thresholds["yawn_min_seconds"]),
        head_roll_warning_deg=float(thresholds["head_roll_warning_deg"]),
        fatigue_score_attention=float(thresholds["fatigue_score_attention"]),
        fatigue_score_alarm=float(thresholds["fatigue_score_alarm"]),
        alert_min_duration_seconds=float(alert["min_duration_seconds"]),
        alert_cooldown_seconds=float(alert["cooldown_seconds"]),
    )
