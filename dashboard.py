from __future__ import annotations

import copy
import re
import time
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.capture import open_capture
from src.evaluation import LABEL_COLUMNS, evaluate_predictions, load_labels, normalize_labels
from src.fatigue_rules import FatigueEstimator
from src.landmarks import MediaPipeFaceMeshDetector
from src.pipeline import process_frame, process_video
from src.reporting import alert_rows, latest_log, load_log, normalize_log, summarize_log
from src.utils import build_fatigue_config, load_config


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
LOG_DIR = ROOT / "outputs" / "logs"
VIDEO_DIR = ROOT / "outputs" / "videos"
REPORT_DIR = ROOT / "outputs" / "reports"
CHART_DIR = ROOT / "outputs" / "charts"
UPLOAD_DIR = ROOT / "data" / "uploads"
SAMPLE_LOG_DIR = ROOT / "data" / "samples"
LABEL_DIR = ROOT / "data" / "labels"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
LABEL_TEMPLATE = "start_seconds,end_seconds,label\n0,5,Atento\n5,10,Fadiga\n10,15,Atento\n"
STATE_COLORS = {
    "Atento": "#2E9F6E",
    "Atencao": "#D6A700",
    "Fadiga": "#D64545",
    "Rosto ausente": "#8A8F98",
    "Desconhecido": "#7A7A7A",
}


def main() -> None:
    st.set_page_config(page_title="Detector de Fadiga", layout="wide")
    _apply_theme()

    _sidebar_brand()
    mode = _sidebar_mode_selector()

    if mode == "Analisar vídeo":
        _video_mode()
    elif mode == "Webcam ao vivo":
        _webcam_mode()
    else:
        _review_mode()


def _video_mode() -> None:
    _page_header("Analisar vídeo", "Upload, processamento offline e revisão da sessão gerada.")
    config = _sidebar_processing_config()

    _sidebar_section("Vídeo", "Envie um arquivo para análise offline.")
    uploaded = st.sidebar.file_uploader("Arquivo de vídeo", type=VIDEO_TYPES, key="video_upload")
    if uploaded is not None:
        _uploaded_file_card(uploaded)
    save_video = st.sidebar.checkbox("Salvar overlay", value=True)
    max_frames = st.sidebar.number_input("Limite de frames", min_value=0, value=0, step=100)

    process = st.sidebar.button(
        "Processar vídeo",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

    if process and uploaded is not None:
        upload_path = _save_uploaded_video(uploaded)
        progress = st.progress(0, text="Processando vídeo")
        status = st.empty()

        def on_progress(frame_id: int, total_frames: int | None) -> None:
            if total_frames:
                progress.progress(min(frame_id / total_frames, 1.0), text=f"Frame {frame_id}/{total_frames}")
            elif frame_id % 25 == 0:
                status.caption(f"Frames processados: {frame_id}")

        with st.spinner("Analisando vídeo..."):
            artifacts = process_video(
                source=str(upload_path),
                config=config,
                root=ROOT,
                headless=True,
                save_video=save_video,
                sound_enabled=False,
                write_report=True,
                max_frames=max_frames or None,
                timestamp_mode="video",
                progress_callback=on_progress,
            )

        progress.progress(1.0, text="Processamento concluído")
        st.session_state["active_log_path"] = str(artifacts.log_path)
        st.session_state["active_video_path"] = str(artifacts.video_path) if artifacts.video_path else ""
        st.success(f"Sessão {artifacts.session_id} processada")

    active_log = st.session_state.get("active_log_path")
    if active_log and Path(active_log).exists():
        _session_view(load_log(active_log), Path(active_log), source_label="Sessão processada")
    else:
        latest = latest_log(LOG_DIR)
        if latest:
            _session_view(load_log(latest), latest, source_label="Última sessão local")
        else:
            _empty_state()


def _webcam_mode() -> None:
    _page_header("Webcam ao vivo", "Captura local com métricas atualizadas durante a execução.")
    config = _sidebar_processing_config()

    _sidebar_section("Câmera", "Teste local quando houver webcam disponível.")
    camera_index = st.sidebar.number_input("Índice", min_value=0, value=0, step=1)
    seconds = st.sidebar.slider("Duração", min_value=5, max_value=120, value=20, step=5)
    start = st.sidebar.button("Iniciar webcam", type="primary", use_container_width=True)

    frame_slot = st.empty()
    metric_slots = st.columns(5)
    status = st.empty()

    if not start:
        status.info("Webcam pronta para teste local.")
        return

    try:
        _run_webcam_preview(
            config=config,
            camera_index=int(camera_index),
            seconds=int(seconds),
            frame_slot=frame_slot,
            metric_slots=metric_slots,
        )
    except Exception as exc:
        st.error(f"Não foi possível iniciar a webcam: {exc}")


def _review_mode() -> None:
    _page_header("Revisar sessão", "Logs, vídeos processados, alertas, timelines e avaliação opcional.")
    df, log_path, source_label = _load_log_source()
    if df is None:
        _empty_state()
        return

    st.caption(source_label)
    labels_df, labels_source = _load_labels()
    _session_view(df, log_path, source_label=source_label, labels_df=labels_df, labels_source=labels_source)


def _session_view(
    df: pd.DataFrame,
    log_path: Path | None,
    *,
    source_label: str,
    labels_df: pd.DataFrame | None = None,
    labels_source: str = "",
) -> None:
    filtered = _sidebar_filters(df)
    summary = summarize_log(filtered)
    _summary_metrics(summary)

    tab_overview, tab_timeline, tab_alerts, tab_eval, tab_files = st.tabs(
        ["Resumo", "Timeline", "Alertas", "Avaliação", "Arquivos"]
    )

    with tab_overview:
        video_path = _matching_video_path(log_path)
        cols = st.columns([0.42, 0.58])
        with cols[0]:
            if video_path and video_path.exists():
                _video_player(video_path)
            else:
                st.info("Nenhum vídeo processado associado a este log.")
        with cols[1]:
            _state_distribution(summary.state_counts)

    with tab_timeline:
        _timeline(filtered)

    with tab_alerts:
        _alerts(filtered)

    with tab_eval:
        if labels_df is None:
            labels_df, labels_source = _load_labels(inline=True)
        if labels_df is not None:
            _evaluation(df, labels_df, labels_source, log_path)
        else:
            _evaluation_source_note(log_path)
            st.info("Selecione um CSV de labels para calcular métricas supervisionadas.")

    with tab_files:
        _files_panel(log_path, source_label)


def _sidebar_processing_config() -> dict:
    config = copy.deepcopy(load_config(CONFIG_PATH))

    _sidebar_section("Processamento", "Controles visuais do overlay.")
    config["ui"]["show_landmarks"] = st.sidebar.toggle(
        "Mostrar landmarks",
        value=bool(config["ui"]["show_landmarks"]),
    )
    config["ui"]["show_debug_panel"] = st.sidebar.toggle(
        "Painel técnico",
        value=bool(config["ui"]["show_debug_panel"]),
        help="Mostra métricas compactas no rodapé do vídeo, sem cobrir o rosto.",
    )
    config["video"]["preserve_aspect_ratio"] = st.sidebar.toggle(
        "Manter proporção",
        value=bool(config["video"].get("preserve_aspect_ratio", True)),
    )

    _sidebar_section("Thresholds", "Ajuste fino da detecção.")
    config["thresholds"]["ear_closed"] = st.sidebar.slider(
        "EAR fechado",
        min_value=0.05,
        max_value=0.40,
        value=float(config["thresholds"]["ear_closed"]),
        step=0.01,
    )
    config["thresholds"]["perclos_fatigue"] = st.sidebar.slider(
        "PERCLOS fadiga",
        min_value=0.10,
        max_value=0.80,
        value=float(config["thresholds"]["perclos_fatigue"]),
        step=0.05,
    )
    config["thresholds"]["fatigue_score_alarm"] = st.sidebar.slider(
        "Score fadiga",
        min_value=20.0,
        max_value=95.0,
        value=float(config["thresholds"]["fatigue_score_alarm"]),
        step=5.0,
    )
    return config


def _run_webcam_preview(*, config: dict, camera_index: int, seconds: int, frame_slot, metric_slots) -> None:
    import cv2

    capture = open_capture("webcam", camera_index=camera_index)
    detector = MediaPipeFaceMeshDetector(**config["face_mesh"])
    estimator = FatigueEstimator(build_fatigue_config(config))

    start_time = time.perf_counter()
    last_time = start_time
    frame_count = 0

    try:
        while time.perf_counter() - start_time < seconds:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            timestamp = now - start_time

            processed = process_frame(
                frame,
                detector=detector,
                estimator=estimator,
                config=config,
                timestamp=timestamp,
                fps=fps,
            )
            rgb = cv2.cvtColor(processed.frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, channels="RGB", use_container_width=True)

            metric_slots[0].metric("Estado", processed.result.state)
            metric_slots[1].metric("Score", f"{processed.result.score:.1f}")
            metric_slots[2].metric("PERCLOS", f"{processed.result.perclos:.2f}")
            metric_slots[3].metric("FPS", f"{processed.fps:.1f}")
            metric_slots[4].metric("Latência", f"{processed.latency_ms:.1f}ms")
    finally:
        capture.release()
        detector.close()


def _load_log_source() -> tuple[pd.DataFrame | None, Path | None, str]:
    with st.sidebar:
        _sidebar_section("Sessão", "Escolha um log já processado.")
        uploaded = st.file_uploader("CSV de log", type=["csv"], key="log_csv")
        logs = _available_logs()
        selected = None

        if logs:
            labels = {str(log.relative_to(ROOT)): log for log in logs}
            default_index = _default_log_index(list(labels.values()))
            selected_name = st.selectbox("Log local", list(labels.keys()), index=default_index)
            selected = labels[selected_name]

    if uploaded is not None:
        return normalize_log(pd.read_csv(uploaded)), None, f"Arquivo enviado: {uploaded.name}"

    if selected is not None:
        return load_log(selected), selected, f"Log local: {selected}"

    latest = latest_log(LOG_DIR)
    if latest is None:
        return None, None, ""

    return load_log(latest), latest, f"Log local: {latest}"


def _load_labels(inline: bool = False) -> tuple[pd.DataFrame | None, str]:
    target = st.container() if inline else st.sidebar
    with target:
        if not inline:
            _sidebar_section("Labels", "CSV com intervalos anotados.")
        st.caption("Use labels anotados por intervalo. O log da sessão já é carregado automaticamente.")
        st.download_button(
            "Baixar modelo de labels",
            data=LABEL_TEMPLATE,
            file_name="template_labels.csv",
            mime="text/csv",
            use_container_width=not inline,
            key="labels_template_inline" if inline else "labels_template_sidebar",
        )
        uploaded = st.file_uploader(
            "CSV de labels anotados",
            type=["csv"],
            key="labels_csv_inline" if inline else "labels_csv",
            help="Colunas esperadas: start_seconds, end_seconds, label. Não envie o CSV de outputs/logs aqui.",
        )
        label_files = _available_labels()
        selected = None

        if label_files:
            labels = {"Nenhum": None}
            labels.update({str(path.relative_to(ROOT)): path for path in label_files})
            selected_name = st.selectbox(
                "Labels locais",
                list(labels.keys()),
                key="labels_select_inline" if inline else "labels_select",
            )
            selected = labels[selected_name]

    if uploaded is not None:
        labels = _read_labels_csv(pd.read_csv(uploaded), source_name=uploaded.name)
        if labels is not None:
            return labels, f"Labels enviados: {uploaded.name}"
        return None, ""

    if selected is not None:
        try:
            return load_labels(selected), f"Labels locais: {selected}"
        except ValueError as exc:
            st.error(f"CSV de labels inválido: {exc}")
            return None, ""

    return None, ""


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_log(df)

    with st.sidebar:
        _sidebar_section("Filtros", "Refine a sessão carregada.")
        if "state" in normalized.columns:
            states = sorted(normalized["state"].dropna().unique())
            selected_states = st.multiselect("Estados", states, default=states)
            if selected_states:
                normalized = normalized[normalized["state"].isin(selected_states)]

        if "fatigue_score" in normalized.columns:
            min_score, max_score = st.slider(
                "Score",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                step=1.0,
            )
            normalized = normalized[
                (normalized["fatigue_score"] >= min_score)
                & (normalized["fatigue_score"] <= max_score)
            ]

    return normalized


def _read_labels_csv(labels_df: pd.DataFrame, *, source_name: str) -> pd.DataFrame | None:
    missing = [column for column in LABEL_COLUMNS if column not in labels_df.columns]
    if not missing:
        return normalize_labels(labels_df)

    if _looks_like_session_log(labels_df):
        st.warning(
            f"`{source_name}` parece ser um log de sessão, não um CSV de labels. "
            "A sessão já usa esse log automaticamente; para avaliação, envie um CSV com "
            "`start_seconds`, `end_seconds` e `label`."
        )
    else:
        st.error(
            "CSV de labels inválido. Colunas obrigatórias: "
            f"`{', '.join(LABEL_COLUMNS)}`."
        )

    return None


def _looks_like_session_log(df: pd.DataFrame) -> bool:
    log_columns = {"frame_id", "timestamp", "state", "fatigue_score", "perclos", "ear_mean", "mar"}
    return len(log_columns.intersection(df.columns)) >= 3


def _summary_metrics(summary) -> None:
    metrics = [
        ("Frames", f"{summary.frames}"),
        ("Duração", f"{summary.duration_seconds:.1f}s"),
        ("FPS médio", f"{summary.avg_fps:.1f}"),
        ("Latência", f"{summary.avg_latency_ms:.1f}ms"),
        ("Score máx.", f"{summary.max_score:.1f}"),
        ("Alertas", f"{summary.alert_count}"),
        ("Face válida", f"{summary.face_valid_ratio:.0%}"),
        ("Atenção", f"{summary.attention_frame_ratio:.0%}"),
        ("Fadiga", f"{summary.fatigue_frame_ratio:.0%}"),
    ]
    cols = st.columns(3)
    for index, (label, value) in enumerate(metrics):
        with cols[index % 3]:
            st.metric(label, value)


def _state_distribution(state_counts: dict[str, int]) -> None:
    if not state_counts:
        return

    state_df = pd.DataFrame(
        {"state": list(state_counts.keys()), "frames": list(state_counts.values())}
    )
    fig = px.bar(
        state_df,
        x="state",
        y="frames",
        color="state",
        color_discrete_map=STATE_COLORS,
        title="Distribuição de estados",
        labels={"state": "Estado", "frames": "Frames"},
        template="plotly_white",
    )
    _style_figure(fig)
    st.plotly_chart(fig, use_container_width=True)
    _section_note(
        "Leitura: este gráfico mostra quantos frames ficaram em cada estado. "
        "Uma barra alta em fadiga indica que os sinais permaneceram acima dos thresholds por mais tempo."
    )


def _timeline(df: pd.DataFrame) -> None:
    normalized = normalize_log(df)
    if "timestamp" not in normalized.columns:
        return

    score_columns = [column for column in ["fatigue_score", "perclos"] if column in normalized]
    if score_columns:
        fig = px.line(
            normalized,
            x="timestamp",
            y=score_columns,
            title="Score e PERCLOS",
            labels={"timestamp": "Tempo (s)", "value": "Valor", "variable": "Sinal"},
            template="plotly_white",
        )
        _style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

    feature_columns = [column for column in ["ear_mean", "mar"] if column in normalized]
    if feature_columns:
        fig = px.line(
            normalized,
            x="timestamp",
            y=feature_columns,
            title="EAR e MAR",
            labels={"timestamp": "Tempo (s)", "value": "Valor", "variable": "Feature"},
            template="plotly_white",
        )
        _style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

    if score_columns or feature_columns:
        _section_note(
            "Leitura: o score resume os sinais temporais de fadiga; PERCLOS sobe quando os olhos ficam fechados "
            "por mais tempo. EAR baixo costuma indicar fechamento dos olhos, e MAR alto pode sugerir boca aberta "
            "ou bocejo."
        )


def _alerts(df: pd.DataFrame) -> None:
    alerts = alert_rows(df)
    if alerts.empty:
        st.info("Nenhum alerta registrado.")
        return

    columns = [
        column
        for column in ["frame_id", "timestamp", "fatigue_score", "perclos", "state", "reasons"]
        if column in alerts.columns
    ]
    st.dataframe(alerts[columns], use_container_width=True, hide_index=True)


def _evaluation(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    source: str,
    log_path: Path | None = None,
) -> None:
    _evaluation_source_note(log_path)
    result = evaluate_predictions(df, labels_df)
    st.caption(source)
    cols = st.columns(4)
    cols[0].metric("Frames avaliados", f"{result.frames_evaluated}")
    cols[1].metric("Acurácia", f"{result.accuracy:.1%}")
    cols[2].metric("Falsos alertas/min", f"{result.false_alarms_per_minute:.2f}")
    cols[3].metric("Fadiga perdida", f"{result.missed_fatigue_frame_ratio:.1%}")

    if result.confusion_matrix:
        matrix_df = pd.DataFrame(result.confusion_matrix).T
        fig = px.imshow(
            matrix_df,
            text_auto=True,
            title="Matriz de confusão",
            labels={"x": "Previsto", "y": "Real", "color": "Frames"},
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        _style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)
        _section_note(
            "Leitura: linhas são os labels reais e colunas são as previsões do detector. "
            "Quanto mais frames na diagonal principal, melhor a concordância com a anotação."
        )

    if result.class_metrics:
        metrics_df = pd.DataFrame(result.class_metrics).T.reset_index(names="classe")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def _files_panel(log_path: Path | None, source_label: str) -> None:
    paths = _session_artifacts(log_path)

    if not paths:
        st.info(source_label)
        return

    st.subheader("Downloads da sessão")
    st.caption("Arquivos gerados para a sessão atualmente carregada.")

    for label, path, mime in paths:
        if path.exists():
            _artifact_download_row(label, path, mime)

    _section_note(
        "Leitura: o CSV é o log bruto da sessão; o vídeo contém o overlay processado; "
        "os relatórios e gráficos HTML são artefatos prontos para revisar ou anexar ao portfólio."
    )


def _artifact_download_row(label: str, path: Path, mime: str) -> None:
    size = _format_file_size(path.stat().st_size)
    kind = _artifact_kind(path)
    relative_path = _relative_path(path)

    with st.container(border=True):
        cols = st.columns([0.28, 0.48, 0.24])
        with cols[0]:
            st.markdown(
                f"""
                <div class="artifact-title">{escape(label)}</div>
                <div class="artifact-meta">{escape(kind)} · {escape(size)}</div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f"""
                <div class="artifact-path" title="{escape(str(path))}">
                    {escape(relative_path)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.download_button(
                "Baixar",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                use_container_width=True,
                key=f"download_{_safe_key(label)}_{_safe_key(path.name)}",
            )


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    kinds = {
        ".csv": "CSV",
        ".mp4": "Vídeo",
        ".md": "Markdown",
        ".json": "JSON",
        ".html": "HTML",
    }
    return kinds.get(suffix, suffix.upper().lstrip(".") or "Arquivo")


def _format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def _available_logs() -> list[Path]:
    if LOG_DIR.exists():
        logs = sorted(LOG_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if logs:
            return logs

    if not SAMPLE_LOG_DIR.exists():
        return []

    return sorted(SAMPLE_LOG_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)


def _available_labels() -> list[Path]:
    if not LABEL_DIR.exists():
        return []

    return sorted(LABEL_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)


def _save_uploaded_video(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(uploaded.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    path.write_bytes(uploaded.getbuffer())
    return path


def _uploaded_file_card(uploaded) -> None:
    size_mb = uploaded.size / (1024 * 1024)
    st.sidebar.markdown(
        f"""
        <div class="uploaded-file-card">
            <div class="uploaded-file-icon">▶</div>
            <div class="uploaded-file-meta">
                <div class="uploaded-file-name">{uploaded.name}</div>
                <div class="uploaded-file-size">{size_mb:.1f} MB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "video.mp4"


def _matching_video_path(log_path: Path | None) -> Path | None:
    active_video = st.session_state.get("active_video_path")
    if active_video and Path(active_video).exists():
        return Path(active_video)

    if not log_path:
        return None

    for suffix in [".mp4", ".webm"]:
        candidate = VIDEO_DIR / f"{log_path.stem}{suffix}"
        if candidate.exists():
            return candidate

    return None


def _video_player(path: Path) -> None:
    mime = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    try:
        st.video(path.read_bytes(), format=mime)
    except Exception:
        st.video(str(path))


def _matching_report_path(log_path: Path | None) -> Path | None:
    if not log_path:
        return None

    candidate = REPORT_DIR / f"{log_path.stem}.md"
    return candidate if candidate.exists() else None


def _session_artifacts(log_path: Path | None) -> list[tuple[str, Path, str]]:
    if not log_path:
        return []

    artifacts: list[tuple[str, Path, str]] = [("Log CSV", log_path, "text/csv")]
    video_path = _matching_video_path(log_path)
    report_md_path = _matching_report_path(log_path)
    report_json_path = REPORT_DIR / f"{log_path.stem}.json"

    if video_path:
        mime = "video/webm" if video_path.suffix.lower() == ".webm" else "video/mp4"
        artifacts.append(("Vídeo processado", video_path, mime))
    if report_md_path:
        artifacts.append(("Relatório Markdown", report_md_path, "text/markdown"))
    if report_json_path.exists():
        artifacts.append(("Relatório JSON", report_json_path, "application/json"))
    if CHART_DIR.exists():
        for chart_path in sorted(CHART_DIR.glob(f"{log_path.stem}_*.html")):
            chart_name = chart_path.stem.replace(f"{log_path.stem}_", "").replace("_", " ")
            artifacts.append((f"Gráfico {chart_name}", chart_path, "text/html"))

    return artifacts


def _evaluation_source_note(log_path: Path | None) -> None:
    if log_path:
        st.caption(f"Log avaliado: {log_path}")
    else:
        st.caption("Log avaliado: arquivo enviado manualmente.")


def _default_log_index(paths: list[Path]) -> int:
    active = st.session_state.get("active_log_path")
    if not active:
        return 0

    for index, path in enumerate(paths):
        if str(path) == active:
            return index
    return 0


def _sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Detector</div>
            <div class="sidebar-brand-subtitle">Fadiga em vídeo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_mode_selector() -> str:
    options = ["Analisar vídeo", "Webcam ao vivo", "Revisar sessão"]
    if "sidebar_mode" not in st.session_state:
        st.session_state["sidebar_mode"] = options[0]

    st.sidebar.markdown("<div class='sidebar-mode-label'>Modo</div>", unsafe_allow_html=True)

    for option in options:
        selected = st.session_state["sidebar_mode"] == option
        if st.sidebar.button(
            option,
            key=f"mode_button_{option}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state["sidebar_mode"] = option
            selected = True

    return st.session_state["sidebar_mode"]


def _sidebar_section(title: str, caption: str | None = None) -> None:
    caption_html = f"<div class='sidebar-section-caption'>{caption}</div>" if caption else ""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">{title}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1>{title}</h1><p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def _empty_state() -> None:
    st.info("Nenhuma sessão encontrada.")


def _section_note(text: str) -> None:
    st.markdown(f"<div class='section-note'>{text}</div>", unsafe_allow_html=True)


def _style_figure(fig) -> None:
    fig.update_layout(
        margin=dict(l=24, r=24, t=58, b=42),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827", size=14, family="Inter, Segoe UI, Arial, sans-serif"),
        title=dict(font=dict(color="#111827", size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            color="#111827",
            title_font=dict(color="#111827", size=14),
            tickfont=dict(color="#374151", size=12),
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
            linecolor="#d1d5db",
        ),
        yaxis=dict(
            color="#111827",
            title_font=dict(color="#111827", size=14),
            tickfont=dict(color="#374151", size=12),
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
            linecolor="#d1d5db",
        ),
    )


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text: #111827;
            --muted: #4b5563;
            --subtle: #6b7280;
            --line: #e5e7eb;
            --panel: #ffffff;
            --salmon: #ff4f55;
            --salmon-dark: #d83b42;
            --soft-salmon: #fff0f1;
        }
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", Arial, sans-serif !important;
        }
        .stApp {
            background: #ffffff;
            color: var(--text);
        }
        #MainMenu,
        footer,
        header {
            background: #ffffff !important;
            color: var(--text) !important;
        }
        .main,
        .block-container,
        section.main > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: #ffffff !important;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] * {
            color: var(--text);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text) !important;
            font-size: 0.95rem;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1320px;
        }
        .sidebar-brand {
            padding: 1.15rem 0 1.35rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }
        .sidebar-brand-title {
            font-size: 1.85rem;
            line-height: 1.05;
            font-weight: 780;
            color: var(--text);
        }
        .sidebar-brand-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .sidebar-section {
            padding-top: 1.05rem;
            margin-top: 0.6rem;
            border-top: 1px solid var(--line);
        }
        .sidebar-section-title {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em !important;
        }
        .sidebar-section-caption {
            margin-top: 0.18rem;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.3;
        }
        h1, h2, h3, h4, h5, h6,
        p, span, label, div {
            color: var(--text);
            letter-spacing: 0 !important;
        }
        h1 {
            font-size: 2rem !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.2rem !important;
            color: var(--text);
            font-weight: 760;
        }
        h2, h3 {
            color: var(--text);
            font-weight: 720;
        }
        .subtitle {
            color: var(--muted);
            margin: 0 0 1.25rem 0;
            font-size: 1rem;
        }
        .section-note {
            margin: 0.8rem 0 1rem 0;
            padding: 0.75rem 0.85rem;
            background: #f9fafb;
            border: 1px solid var(--line);
            border-left: 3px solid var(--salmon);
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .artifact-title {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 720;
            line-height: 1.25;
            margin-top: 0.08rem;
        }
        .artifact-meta {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.3;
            margin-top: 0.2rem;
        }
        .artifact-path {
            width: 100%;
            min-height: 2.75rem;
            display: flex;
            align-items: center;
            padding: 0 0.78rem;
            background: #f9fafb;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--muted);
            font-family: "Cascadia Mono", "Consolas", monospace;
            font-size: 0.82rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 720;
        }
        div[data-testid="stTabs"] button p {
            font-size: 0.98rem;
            color: var(--text) !important;
            font-weight: 560;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom-color: var(--salmon) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--salmon) !important;
        }
        .sidebar-mode-label {
            margin: 0.75rem 0 0.4rem 0;
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em !important;
        }
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stToggle"] label {
            color: var(--text) !important;
        }
        div[role="radiogroup"] label > div:first-child,
        [data-testid="stCheckbox"] div[data-testid="stCheckbox"] {
            border-color: var(--salmon) !important;
        }
        [data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] section * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] button * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: var(--soft-salmon) !important;
            color: var(--text) !important;
            border-color: var(--salmon) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
        }
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] div:has([data-testid="stFileUploaderFileName"]),
        [data-testid="stSidebar"] div:has([data-testid="stFileUploaderFileName"]) {
            display: none !important;
        }
        .uploaded-file-card {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.45rem 0 0.7rem 0;
            padding: 0.7rem 0.75rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .uploaded-file-icon {
            width: 2rem;
            height: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            border: 1px solid var(--line);
            border-radius: 7px;
            color: var(--text);
            background: #f9fafb;
            font-size: 0.82rem;
        }
        .uploaded-file-meta {
            min-width: 0;
        }
        .uploaded-file-name {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .uploaded-file-size {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.12rem;
        }
        .stButton > button {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px;
            font-weight: 650;
        }
        .stButton > button:hover {
            background: var(--soft-salmon) !important;
            color: var(--text) !important;
            border-color: var(--salmon) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--salmon) !important;
            color: #ffffff !important;
            border-color: var(--salmon) !important;
        }
        .stButton > button[kind="primary"] * {
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--salmon-dark) !important;
            color: #ffffff !important;
            border-color: var(--salmon-dark) !important;
        }
        .stDownloadButton > button {
            background: var(--salmon) !important;
            color: #ffffff !important;
            border: 1px solid var(--salmon) !important;
            border-radius: 8px;
            font-weight: 650;
            min-height: 2.75rem;
        }
        .stDownloadButton > button *,
        .stDownloadButton > button:hover * {
            color: #ffffff !important;
        }
        .stDownloadButton > button:hover {
            background: var(--salmon-dark) !important;
            color: #ffffff !important;
            border-color: var(--salmon-dark) !important;
        }
        [data-baseweb="select"],
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="textarea"] {
            background: #ffffff !important;
            color: var(--text) !important;
        }
        [data-baseweb="select"] {
            min-height: 3rem !important;
            overflow: visible !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] > div,
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"],
        ul[role="listbox"],
        li[role="option"],
        input {
            background: #ffffff !important;
            color: var(--text) !important;
            border-color: var(--line) !important;
            min-height: 2.8rem !important;
            overflow: visible !important;
            padding-left: 0.35rem !important;
        }
        [data-baseweb="select"] > div > div,
        [data-baseweb="select"] div[role="combobox"],
        [data-baseweb="select"] div[aria-haspopup="listbox"] {
            overflow: visible !important;
            min-height: 2.5rem !important;
            padding-left: 0.25rem !important;
        }
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *,
        [role="listbox"] *,
        li[role="option"] * {
            background: #ffffff !important;
            color: var(--text) !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background: var(--soft-salmon) !important;
            color: var(--text) !important;
        }
        [data-baseweb="select"] svg,
        [data-baseweb="input"] svg,
        [data-testid="stNumberInput"] svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }
        [data-testid="stNumberInput"] button {
            background: #ffffff !important;
            color: var(--text) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stNumberInput"] button *,
        [data-testid="stNumberInput"] input {
            color: var(--text) !important;
            background: #ffffff !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background: var(--soft-salmon) !important;
        }
        [data-baseweb="tag"] {
            background-color: var(--salmon) !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            min-height: 1.85rem !important;
            height: auto !important;
            padding: 0.28rem 0.5rem !important;
            margin: 0.2rem 0.2rem 0.2rem 0.15rem !important;
            align-items: center !important;
            overflow: visible !important;
        }
        [data-baseweb="tag"] * {
            color: #ffffff !important;
            line-height: 1.1rem !important;
            overflow: visible !important;
        }
        [data-baseweb="tag"] svg,
        [data-baseweb="tag"] path {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-baseweb="select"] [data-baseweb="tag"] span {
            color: #ffffff !important;
        }
        .stSlider [data-baseweb="slider"] div {
            color: var(--text) !important;
        }
        .stSlider [role="slider"] {
            background-color: var(--salmon) !important;
            border-color: var(--salmon) !important;
        }
        .stSlider [data-testid="stTickBar"] {
            color: var(--muted) !important;
        }
        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"],
        [data-testid="stCheckbox"] div[aria-checked="true"] {
            background-color: var(--salmon) !important;
            border-color: var(--salmon) !important;
        }
        [data-baseweb="switch"] div[aria-checked="true"],
        button[role="switch"][aria-checked="true"] {
            background-color: var(--salmon) !important;
            border-color: var(--salmon) !important;
        }
        .stDataFrame,
        [data-testid="stDataFrame"] {
            color: var(--text);
        }
        .stAlert {
            color: var(--text);
        }
        code, pre {
            color: var(--text) !important;
            background: #f9fafb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
