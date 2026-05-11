from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .reporting import normalize_log


LABEL_COLUMNS = ("start_seconds", "end_seconds", "label")
DEFAULT_CLASSES = ("Atento", "Atencao", "Fadiga", "Rosto ausente")
UNLABELED = "Sem label"


@dataclass(frozen=True)
class EvaluationResult:
    frames_evaluated: int
    accuracy: float
    false_alarm_events: int
    false_alarms_per_minute: float
    missed_fatigue_frame_ratio: float
    confusion_matrix: dict[str, dict[str, int]]
    class_metrics: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_evaluated": self.frames_evaluated,
            "accuracy": self.accuracy,
            "false_alarm_events": self.false_alarm_events,
            "false_alarms_per_minute": self.false_alarms_per_minute,
            "missed_fatigue_frame_ratio": self.missed_fatigue_frame_ratio,
            "confusion_matrix": self.confusion_matrix,
            "class_metrics": self.class_metrics,
        }


def load_labels(path: str | Path) -> pd.DataFrame:
    return normalize_labels(pd.read_csv(path))


def normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"Label CSV missing required columns: {', '.join(missing)}")

    normalized = labels.loc[:, LABEL_COLUMNS].copy()
    normalized["start_seconds"] = pd.to_numeric(
        normalized["start_seconds"],
        errors="coerce",
    )
    normalized["end_seconds"] = pd.to_numeric(
        normalized["end_seconds"],
        errors="coerce",
    )
    normalized["label"] = normalized["label"].astype(str).str.strip()
    normalized = normalized.dropna(subset=["start_seconds", "end_seconds"])
    normalized = normalized[normalized["end_seconds"] > normalized["start_seconds"]]
    normalized = normalized.sort_values(["start_seconds", "end_seconds"]).reset_index(drop=True)

    return normalized


def attach_labels(log_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    log = normalize_log(log_df)
    labels = normalize_labels(labels_df)
    labeled = log.copy()
    labeled["true_label"] = UNLABELED

    if "timestamp" not in labeled.columns:
        raise ValueError("Log CSV needs a timestamp column to attach labels")

    for row in labels.itertuples(index=False):
        mask = (labeled["timestamp"] >= row.start_seconds) & (
            labeled["timestamp"] < row.end_seconds
        )
        labeled.loc[mask, "true_label"] = row.label

    return labeled


def evaluate_predictions(
    log_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    drop_unlabeled: bool = True,
) -> EvaluationResult:
    labeled = attach_labels(log_df, labels_df)
    if drop_unlabeled:
        labeled = labeled[labeled["true_label"] != UNLABELED].copy()

    if labeled.empty:
        return EvaluationResult(
            frames_evaluated=0,
            accuracy=0.0,
            false_alarm_events=0,
            false_alarms_per_minute=0.0,
            missed_fatigue_frame_ratio=0.0,
            confusion_matrix={},
            class_metrics={},
        )

    class_names = tuple(dict.fromkeys(classes + tuple(labeled["true_label"]) + tuple(labeled["state"])))
    confusion = _confusion_matrix(labeled, class_names)
    metrics = _class_metrics(confusion, class_names)

    correct = (labeled["true_label"] == labeled["state"]).sum()
    accuracy = float(correct / len(labeled))
    false_alarm_events = _false_alarm_events(labeled)
    false_alarm_rate = false_alarm_events / max(_duration_minutes(labeled), 1e-9)
    missed_fatigue_ratio = _missed_fatigue_ratio(labeled)

    return EvaluationResult(
        frames_evaluated=len(labeled),
        accuracy=accuracy,
        false_alarm_events=false_alarm_events,
        false_alarms_per_minute=false_alarm_rate,
        missed_fatigue_frame_ratio=missed_fatigue_ratio,
        confusion_matrix=confusion,
        class_metrics=metrics,
    )


def evaluate_files(log_path: str | Path, labels_path: str | Path) -> EvaluationResult:
    log = pd.read_csv(log_path)
    labels = load_labels(labels_path)
    return evaluate_predictions(log, labels)


def write_evaluation_json(result: EvaluationResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(result.as_dict(), file, ensure_ascii=False, indent=2)

    return target


def write_evaluation_markdown(result: EvaluationResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_evaluation_markdown(result), encoding="utf-8")
    return target


def format_evaluation_markdown(result: EvaluationResult) -> str:
    lines = [
        "# Avaliacao do Detector",
        "",
        "## Resumo",
        "",
        f"- Frames avaliados: {result.frames_evaluated}",
        f"- Acuracia: {result.accuracy:.2%}",
        f"- Falsos alertas: {result.false_alarm_events}",
        f"- Falsos alertas por minuto: {result.false_alarms_per_minute:.2f}",
        f"- Razao de fadiga perdida: {result.missed_fatigue_frame_ratio:.2%}",
        "",
        "## Metricas por classe",
        "",
    ]

    if result.class_metrics:
        lines.append("| Classe | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for class_name, metrics in result.class_metrics.items():
            lines.append(
                "| {class_name} | {precision:.2%} | {recall:.2%} | {f1:.2%} | {support:.0f} |".format(
                    class_name=class_name,
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    support=metrics["support"],
                )
            )
    else:
        lines.append("Nenhuma metrica calculada.")

    lines.extend(["", "## Matriz de confusao", ""])

    if result.confusion_matrix:
        predicted_classes = list(next(iter(result.confusion_matrix.values())).keys())
        lines.append("| Real \\ Previsto | " + " | ".join(predicted_classes) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(predicted_classes)) + "|")
        for actual, row in result.confusion_matrix.items():
            values = [str(row[predicted]) for predicted in predicted_classes]
            lines.append("| " + actual + " | " + " | ".join(values) + " |")
    else:
        lines.append("Nenhuma matriz calculada.")

    lines.append("")
    return "\n".join(lines)


def _confusion_matrix(
    labeled: pd.DataFrame,
    class_names: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        actual: {predicted: 0 for predicted in class_names}
        for actual in class_names
    }

    for row in labeled.itertuples(index=False):
        matrix[str(row.true_label)][str(row.state)] += 1

    return matrix


def _class_metrics(
    confusion: dict[str, dict[str, int]],
    class_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[actual][class_name] for actual in class_names if actual != class_name)
        fn = sum(confusion[class_name][predicted] for predicted in class_names if predicted != class_name)
        support = sum(confusion[class_name].values())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return metrics


def _false_alarm_events(labeled: pd.DataFrame) -> int:
    if "alert_triggered" in labeled.columns:
        alerts = labeled["alert_triggered"].fillna(False).astype(bool)
        return int((alerts & (labeled["true_label"] != "Fadiga")).sum())

    predicted_fatigue = labeled["state"] == "Fadiga"
    starts = predicted_fatigue & ~predicted_fatigue.shift(fill_value=False)
    false_starts = starts & (labeled["true_label"] != "Fadiga")
    return int(false_starts.sum())


def _missed_fatigue_ratio(labeled: pd.DataFrame) -> float:
    true_fatigue = labeled["true_label"] == "Fadiga"
    total = int(true_fatigue.sum())
    if total == 0:
        return 0.0

    missed = true_fatigue & (labeled["state"] != "Fadiga")
    return float(missed.sum() / total)


def _duration_minutes(labeled: pd.DataFrame) -> float:
    if "timestamp" not in labeled.columns or labeled["timestamp"].dropna().empty:
        return 0.0

    duration_seconds = float(labeled["timestamp"].max() - labeled["timestamp"].min())
    return max(duration_seconds / 60.0, 0.0)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
