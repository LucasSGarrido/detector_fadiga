from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


NUMERIC_COLUMNS = (
    "timestamp",
    "frame_id",
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
    "fps",
    "latency_ms",
)


@dataclass(frozen=True)
class SessionSummary:
    frames: int
    duration_seconds: float
    avg_fps: float
    avg_latency_ms: float
    face_valid_ratio: float
    max_score: float
    mean_score: float
    alert_count: int
    fatigue_frame_ratio: float
    attention_frame_ratio: float
    state_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
            "avg_fps": self.avg_fps,
            "avg_latency_ms": self.avg_latency_ms,
            "face_valid_ratio": self.face_valid_ratio,
            "max_score": self.max_score,
            "mean_score": self.mean_score,
            "alert_count": self.alert_count,
            "fatigue_frame_ratio": self.fatigue_frame_ratio,
            "attention_frame_ratio": self.attention_frame_ratio,
            "state_counts": self.state_counts,
        }


def load_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_log(df)


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "alert_triggered" in normalized.columns:
        normalized["alert_triggered"] = normalized["alert_triggered"].map(_to_bool)

    if "state" in normalized.columns:
        normalized["state"] = normalized["state"].fillna("Desconhecido")

    return normalized


def summarize_log(df: pd.DataFrame) -> SessionSummary:
    if df.empty:
        return SessionSummary(
            frames=0,
            duration_seconds=0.0,
            avg_fps=0.0,
            avg_latency_ms=0.0,
            face_valid_ratio=0.0,
            max_score=0.0,
            mean_score=0.0,
            alert_count=0,
            fatigue_frame_ratio=0.0,
            attention_frame_ratio=0.0,
            state_counts={},
        )

    normalized = normalize_log(df)
    frames = len(normalized)
    duration_seconds = _duration_seconds(normalized)
    state_counts = _state_counts(normalized)

    return SessionSummary(
        frames=frames,
        duration_seconds=duration_seconds,
        avg_fps=_mean(normalized, "fps"),
        avg_latency_ms=_mean(normalized, "latency_ms"),
        face_valid_ratio=_mean(normalized, "valid_frames_ratio"),
        max_score=_max(normalized, "fatigue_score"),
        mean_score=_mean(normalized, "fatigue_score"),
        alert_count=_alert_count(normalized),
        fatigue_frame_ratio=_state_ratio(normalized, "Fadiga"),
        attention_frame_ratio=_state_ratio(normalized, "Atencao"),
        state_counts=state_counts,
    )


def latest_log(log_dir: str | Path) -> Path | None:
    path = Path(log_dir)
    if not path.exists():
        return None

    logs = sorted(path.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def alert_rows(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_log(df)
    if "alert_triggered" not in normalized.columns:
        return normalized.iloc[0:0]

    return normalized[normalized["alert_triggered"]].copy()


def build_report_payload(
    df: pd.DataFrame,
    *,
    source: str,
    log_path: str | Path,
    video_path: str | Path | None = None,
    chart_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    summary = summarize_log(df)
    alerts = alert_rows(df)

    return {
        "source": source,
        "log_path": str(log_path),
        "video_path": str(video_path) if video_path else None,
        "chart_paths": [str(path) for path in chart_paths or []],
        "summary": summary.as_dict(),
        "alerts": _alerts_payload(alerts),
    }


def write_json_report(payload: dict[str, Any], path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return report_path


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_markdown_report(payload), encoding="utf-8")
    return report_path


def format_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    state_counts = summary.get("state_counts", {})
    alerts = payload.get("alerts", [])

    lines = [
        "# Relatório da Sessão",
        "",
        f"- Fonte: `{payload['source']}`",
        f"- Log: `{payload['log_path']}`",
    ]

    if payload.get("video_path"):
        lines.append(f"- Vídeo processado: `{payload['video_path']}`")

    if payload.get("chart_paths"):
        lines.append("")
        lines.append("## Gráficos")
        lines.append("")
        for chart_path in payload["chart_paths"]:
            lines.append(f"- `{chart_path}`")

    lines.extend(
        [
            "",
            "## Resumo",
            "",
            f"- Frames analisados: {summary['frames']}",
            f"- Duração estimada: {summary['duration_seconds']:.2f}s",
            f"- FPS médio de processamento: {summary['avg_fps']:.2f}",
            f"- Latência média: {summary['avg_latency_ms']:.2f}ms",
            f"- Face válida média: {summary['face_valid_ratio']:.1%}",
            f"- Score máximo: {summary['max_score']:.2f}",
            f"- Score médio: {summary['mean_score']:.2f}",
            f"- Alertas: {summary['alert_count']}",
            f"- Frames em atenção: {summary['attention_frame_ratio']:.1%}",
            f"- Frames em fadiga: {summary['fatigue_frame_ratio']:.1%}",
            "",
            "## Estados",
            "",
        ]
    )

    if state_counts:
        for state, count in state_counts.items():
            lines.append(f"- {state}: {count}")
    else:
        lines.append("- Nenhum estado registrado.")

    lines.extend(["", "## Alertas", ""])

    if alerts:
        for alert in alerts:
            lines.append(
                "- Frame {frame_id}: score {fatigue_score:.2f}, PERCLOS {perclos:.2f}, motivos: {reasons}".format(
                    frame_id=alert.get("frame_id", ""),
                    fatigue_score=alert.get("fatigue_score", 0.0),
                    perclos=alert.get("perclos", 0.0),
                    reasons=alert.get("reasons", ""),
                )
            )
    else:
        lines.append("- Nenhum alerta registrado.")

    lines.append("")
    return "\n".join(lines)


def _duration_seconds(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        return 0.0

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    return max(0.0, float(end - start))


def _mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0

    value = df[column].mean(skipna=True)
    if pd.isna(value):
        return 0.0

    return float(value)


def _max(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0

    value = df[column].max(skipna=True)
    if pd.isna(value):
        return 0.0

    return float(value)


def _alert_count(df: pd.DataFrame) -> int:
    if "alert_triggered" not in df.columns:
        return 0

    return int(df["alert_triggered"].fillna(False).sum())


def _state_ratio(df: pd.DataFrame, state: str) -> float:
    if "state" not in df.columns or df.empty:
        return 0.0

    return float((df["state"] == state).mean())


def _state_counts(df: pd.DataFrame) -> dict[str, int]:
    if "state" not in df.columns:
        return {}

    counts = df["state"].value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def _alerts_payload(alerts: pd.DataFrame) -> list[dict[str, Any]]:
    if alerts.empty:
        return []

    columns = [
        column
        for column in ["frame_id", "timestamp", "fatigue_score", "perclos", "state", "reasons"]
        if column in alerts.columns
    ]
    return alerts[columns].to_dict(orient="records")
