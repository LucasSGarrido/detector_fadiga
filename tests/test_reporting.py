import pandas as pd

from src.reporting import (
    alert_rows,
    build_report_payload,
    normalize_log,
    summarize_log,
    write_json_report,
    write_markdown_report,
)


def _sample_log():
    return pd.DataFrame(
        {
            "timestamp": ["10.0", "10.5", "11.0", "11.5"],
            "frame_id": ["1", "2", "3", "4"],
            "fatigue_score": ["10", "40", "72", "80"],
            "valid_frames_ratio": ["1.0", "1.0", "0.8", "0.8"],
            "fps": ["30", "28", "29", "30"],
            "latency_ms": ["12", "14", "13", "15"],
            "state": ["Atento", "Atencao", "Fadiga", "Fadiga"],
            "alert_triggered": ["False", "False", "True", "False"],
        }
    )


def test_normalize_log_coerces_numeric_and_bool_columns():
    normalized = normalize_log(_sample_log())

    assert normalized["frame_id"].iloc[0] == 1
    assert normalized["fatigue_score"].iloc[-1] == 80
    assert bool(normalized["alert_triggered"].iloc[2]) is True


def test_summarize_log_returns_session_metrics():
    summary = summarize_log(_sample_log())

    assert summary.frames == 4
    assert summary.duration_seconds == 1.5
    assert summary.alert_count == 1
    assert summary.state_counts["Fadiga"] == 2
    assert summary.fatigue_frame_ratio == 0.5
    assert summary.attention_frame_ratio == 0.25


def test_alert_rows_filters_triggered_alerts():
    alerts = alert_rows(_sample_log())

    assert len(alerts) == 1
    assert alerts.iloc[0]["frame_id"] == 3


def test_report_writers_create_json_and_markdown(tmp_path):
    payload = build_report_payload(_sample_log(), source="video.mp4", log_path="log.csv")

    json_path = write_json_report(payload, tmp_path / "report.json")
    markdown_path = write_markdown_report(payload, tmp_path / "report.md")

    assert json_path.exists()
    assert markdown_path.exists()
    assert "Relatório da Sessão" in markdown_path.read_text(encoding="utf-8")
