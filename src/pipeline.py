from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .alert import AlertPlayer
from .capture import is_webcam_source, open_capture
from .fatigue_rules import FatigueEstimator, FatigueResult
from .features import FaceFeatures, extract_face_features
from .landmarks import FaceDetection, MediaPipeFaceMeshDetector
from .plots import write_html_charts
from .reporting import (
    SessionSummary,
    build_report_payload,
    load_log,
    summarize_log,
    write_json_report,
    write_markdown_report,
)
from .utils import build_fatigue_config
from .visualization import draw_overlay


ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class ProcessedFrame:
    frame: object
    detection: FaceDetection
    features: FaceFeatures
    result: FatigueResult
    fps: float
    latency_ms: float
    timestamp: float


@dataclass(frozen=True)
class SessionArtifacts:
    session_id: str
    source: str
    log_path: Path
    video_path: Path | None
    report_json_path: Path | None
    report_md_path: Path | None
    chart_paths: list[Path]
    summary: SessionSummary
    frames_processed: int


def process_video(
    *,
    source: str,
    config: dict,
    root: Path,
    camera_index: int | None = None,
    headless: bool = True,
    save_video: bool = False,
    sound_enabled: bool = False,
    write_report: bool = True,
    max_frames: int | None = None,
    timestamp_mode: str = "auto",
    session_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SessionArtifacts:
    root = Path(root)
    session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = _session_paths(root, session_id)

    camera_index = int(config["video"]["camera_index"]) if camera_index is None else camera_index
    capture = open_capture(source, camera_index=camera_index)
    total_frames = _total_frames(capture)
    detector = MediaPipeFaceMeshDetector(**config["face_mesh"])
    estimator = FatigueEstimator(build_fatigue_config(config))
    alert = AlertPlayer(enabled=sound_enabled)
    writer = None

    frame_id = 0
    last_time = time.perf_counter()
    session_start = last_time
    resolved_timestamp_mode = resolve_timestamp_mode(source, timestamp_mode)

    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    paths["video"].parent.mkdir(parents=True, exist_ok=True)

    with paths["log"].open("w", newline="", encoding="utf-8") as log_file:
        logger = csv.DictWriter(log_file, fieldnames=log_fieldnames())
        logger.writeheader()

        try:
            while True:
                ok, raw_frame = capture.read()
                if not ok:
                    break

                frame_id += 1
                if max_frames and frame_id > max_frames:
                    break

                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                timestamp = frame_timestamp(
                    capture,
                    resolved_timestamp_mode,
                    now,
                    session_start,
                )
                processed = process_frame(
                    raw_frame,
                    detector=detector,
                    estimator=estimator,
                    config=config,
                    timestamp=timestamp,
                    fps=fps,
                )

                if processed.result.alert_triggered:
                    alert.play()

                logger.writerow(
                    log_row(
                        frame_id=frame_id,
                        timestamp=processed.timestamp,
                        features=processed.features,
                        result=processed.result,
                        fps=processed.fps,
                        latency_ms=processed.latency_ms,
                    )
                )

                if save_video and writer is None:
                    writer = build_writer(processed.frame, capture, paths["video"])
                if writer is not None:
                    writer.write(processed.frame)

                if not headless:
                    import cv2

                    cv2.imshow("Detector de Fadiga", processed.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if progress_callback is not None:
                    progress_callback(frame_id, total_frames)
        finally:
            capture.release()
            detector.close()
            if writer is not None:
                writer.release()
            if not headless:
                import cv2

                cv2.destroyAllWindows()

    log_df = load_log(paths["log"])
    chart_paths: list[Path] = []
    report_json_path: Path | None = None
    report_md_path: Path | None = None
    video_path = paths["video"] if save_video else None

    if write_report:
        paths["report_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["charts"].mkdir(parents=True, exist_ok=True)
        chart_paths = write_html_charts(log_df, paths["charts"], prefix=f"session_{session_id}")
        payload = build_report_payload(
            log_df,
            source=source,
            log_path=paths["log"],
            video_path=video_path,
            chart_paths=chart_paths,
        )
        report_json_path = write_json_report(payload, paths["report_json"])
        report_md_path = write_markdown_report(payload, paths["report_md"])

    return SessionArtifacts(
        session_id=session_id,
        source=source,
        log_path=paths["log"],
        video_path=video_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
        chart_paths=chart_paths,
        summary=summarize_log(log_df),
        frames_processed=frame_id,
    )


def process_frame(
    raw_frame,
    *,
    detector: MediaPipeFaceMeshDetector,
    estimator: FatigueEstimator,
    config: dict,
    timestamp: float,
    fps: float,
) -> ProcessedFrame:
    frame = resize_frame(raw_frame, config)
    start = time.perf_counter()
    detection = detector.detect(frame)
    features = extract_face_features(detection.points)
    result = estimator.update(features, timestamp=timestamp)
    latency_ms = (time.perf_counter() - start) * 1000.0

    draw_overlay(
        frame,
        points=detection.points,
        features=features,
        result=result,
        fps=fps,
        latency_ms=latency_ms,
        show_landmarks=bool(config["ui"]["show_landmarks"]),
        show_debug_panel=bool(config["ui"]["show_debug_panel"]),
    )

    return ProcessedFrame(
        frame=frame,
        detection=detection,
        features=features,
        result=result,
        fps=fps,
        latency_ms=latency_ms,
        timestamp=timestamp,
    )


def resize_frame(frame, config: dict):
    import cv2

    target_width = int(config["video"]["target_width"])
    target_height = int(config["video"]["target_height"])
    if target_width <= 0 or target_height <= 0:
        return frame

    if bool(config["video"].get("preserve_aspect_ratio", True)):
        height, width = frame.shape[:2]
        scale = min(target_width / width, target_height / height)
        if scale <= 0:
            return frame

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(frame, (new_width, new_height))

    return cv2.resize(frame, (target_width, target_height))


def build_writer(frame, capture, path: Path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def resolve_timestamp_mode(source: str, mode: str) -> str:
    if mode != "auto":
        return mode

    return "real" if is_webcam_source(source) else "video"


def frame_timestamp(capture, mode: str, now: float, session_start: float) -> float:
    if mode == "video":
        import cv2

        timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms and timestamp_ms > 0:
            return timestamp_ms / 1000.0

    return now - session_start


def log_fieldnames() -> list[str]:
    return [
        "timestamp",
        "frame_id",
        "face_detected",
        "ear_left",
        "ear_right",
        "ear_mean",
        "mar",
        "head_roll_deg",
        "perclos",
        "valid_frames_ratio",
        "blink_count",
        "long_blink_count",
        "yawn_count",
        "current_eye_closed_seconds",
        "fatigue_score",
        "state",
        "alert_triggered",
        "fps",
        "latency_ms",
        "reasons",
    ]


def log_row(*, frame_id, timestamp, features, result, fps, latency_ms) -> dict[str, object]:
    return {
        "timestamp": f"{timestamp:.6f}",
        "frame_id": frame_id,
        "face_detected": features.face_detected,
        "ear_left": _fmt(features.ear_left),
        "ear_right": _fmt(features.ear_right),
        "ear_mean": _fmt(features.ear_mean),
        "mar": _fmt(features.mar),
        "head_roll_deg": _fmt(features.head_roll_deg),
        "perclos": f"{result.perclos:.6f}",
        "valid_frames_ratio": f"{result.valid_frames_ratio:.6f}",
        "blink_count": result.blink_count,
        "long_blink_count": result.long_blink_count,
        "yawn_count": result.yawn_count,
        "current_eye_closed_seconds": f"{result.current_eye_closed_seconds:.6f}",
        "fatigue_score": f"{result.score:.6f}",
        "state": result.state,
        "alert_triggered": result.alert_triggered,
        "fps": f"{fps:.3f}",
        "latency_ms": f"{latency_ms:.3f}",
        "reasons": " | ".join(result.reasons),
    }


def _session_paths(root: Path, session_id: str) -> dict[str, Path]:
    return {
        "log": root / "outputs" / "logs" / f"session_{session_id}.csv",
        "video": root / "outputs" / "videos" / f"session_{session_id}.mp4",
        "report_json": root / "outputs" / "reports" / f"session_{session_id}.json",
        "report_md": root / "outputs" / "reports" / f"session_{session_id}.md",
        "charts": root / "outputs" / "charts",
    }


def _total_frames(capture) -> int | None:
    import cv2

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return total if total > 0 else None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
