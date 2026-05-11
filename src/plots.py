from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from .reporting import normalize_log


def build_score_figure(df: pd.DataFrame):
    normalized = normalize_log(df)
    columns = [column for column in ["fatigue_score", "perclos"] if column in normalized.columns]
    if not columns:
        return None

    return px.line(
        normalized,
        x="timestamp",
        y=columns,
        title="Score de fadiga e PERCLOS",
        labels={"timestamp": "Tempo (s)", "value": "Valor", "variable": "Sinal"},
    )


def build_feature_figure(df: pd.DataFrame):
    normalized = normalize_log(df)
    columns = [column for column in ["ear_mean", "mar"] if column in normalized.columns]
    if not columns:
        return None

    return px.line(
        normalized,
        x="timestamp",
        y=columns,
        title="EAR e MAR ao longo do tempo",
        labels={"timestamp": "Tempo (s)", "value": "Valor", "variable": "Feature"},
    )


def build_state_figure(df: pd.DataFrame):
    normalized = normalize_log(df)
    if "state" not in normalized.columns:
        return None

    state_df = (
        normalized["state"]
        .value_counts()
        .rename_axis("state")
        .reset_index(name="frames")
    )
    return px.bar(
        state_df,
        x="state",
        y="frames",
        color="state",
        title="Distribuicao de estados",
        labels={"state": "Estado", "frames": "Frames"},
    )


def write_html_charts(df: pd.DataFrame, output_dir: str | Path, *, prefix: str) -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    figures = {
        "score_perclos": build_score_figure(df),
        "ear_mar": build_feature_figure(df),
        "states": build_state_figure(df),
    }

    paths: list[Path] = []
    for name, figure in figures.items():
        if figure is None:
            continue

        path = target / f"{prefix}_{name}.html"
        figure.write_html(path, include_plotlyjs="cdn")
        paths.append(path)

    return paths
