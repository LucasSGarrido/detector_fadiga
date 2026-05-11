import pandas as pd

from src.evaluation import (
    attach_labels,
    evaluate_predictions,
    normalize_labels,
    write_evaluation_json,
    write_evaluation_markdown,
)


def _sample_log():
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "frame_id": [1, 2, 3, 4, 5, 6, 7],
            "state": ["Atento", "Atento", "Atento", "Atento", "Atencao", "Fadiga", "Atencao"],
            "alert_triggered": [False, False, False, False, False, True, False],
        }
    )


def _sample_labels():
    return pd.DataFrame(
        {
            "start_seconds": [0.0, 2.0],
            "end_seconds": [2.0, 3.5],
            "label": ["Atento", "Fadiga"],
        }
    )


def test_normalize_labels_requires_expected_columns():
    labels = normalize_labels(_sample_labels())

    assert list(labels.columns) == ["start_seconds", "end_seconds", "label"]
    assert labels.iloc[0]["label"] == "Atento"


def test_attach_labels_adds_true_label_by_timestamp_interval():
    labeled = attach_labels(_sample_log(), _sample_labels())

    assert labeled.iloc[0]["true_label"] == "Atento"
    assert labeled.iloc[4]["true_label"] == "Fadiga"


def test_evaluate_predictions_returns_confusion_and_metrics():
    result = evaluate_predictions(_sample_log(), _sample_labels())

    assert result.frames_evaluated == 7
    assert result.confusion_matrix["Atento"]["Atento"] == 4
    assert result.confusion_matrix["Fadiga"]["Fadiga"] == 1
    assert result.class_metrics["Fadiga"]["recall"] == 1 / 3
    assert result.false_alarm_events == 0


def test_evaluation_writers_create_artifacts(tmp_path):
    result = evaluate_predictions(_sample_log(), _sample_labels())

    json_path = write_evaluation_json(result, tmp_path / "evaluation.json")
    markdown_path = write_evaluation_markdown(result, tmp_path / "evaluation.md")

    assert json_path.exists()
    assert markdown_path.exists()
    assert "Avaliacao do Detector" in markdown_path.read_text(encoding="utf-8")
