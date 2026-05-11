import pandas as pd

from src.plots import build_feature_figure, build_score_figure, build_state_figure, write_html_charts


def _sample_log():
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.5, 1.0],
            "fatigue_score": [5, 30, 70],
            "perclos": [0.0, 0.2, 0.5],
            "ear_mean": [0.30, 0.20, 0.15],
            "mar": [0.20, 0.22, 0.60],
            "state": ["Atento", "Atencao", "Fadiga"],
        }
    )


def test_build_figures_from_log():
    df = _sample_log()

    assert build_score_figure(df) is not None
    assert build_feature_figure(df) is not None
    assert build_state_figure(df) is not None


def test_write_html_charts_creates_files(tmp_path):
    paths = write_html_charts(_sample_log(), tmp_path, prefix="session_test")

    assert len(paths) == 3
    assert all(path.exists() for path in paths)
