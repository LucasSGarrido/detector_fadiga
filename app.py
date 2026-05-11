from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import process_video
from src.utils import load_config


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector de fadiga em tempo real")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--source", default=None, help="Use 'webcam' or a video path")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--timestamp-mode",
        choices=["auto", "video", "real"],
        default="auto",
        help="Use video timestamps for files or real elapsed time for webcam",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    source = args.source or config["video"]["source"]
    camera_index = args.camera_index
    if camera_index is None:
        camera_index = int(config["video"]["camera_index"])

    artifacts = process_video(
        source=source,
        config=config,
        root=ROOT,
        camera_index=camera_index,
        headless=args.headless,
        save_video=args.save_video,
        sound_enabled=bool(config["alert"]["sound_enabled"]) and not args.no_sound,
        write_report=not args.no_report,
        max_frames=args.max_frames,
        timestamp_mode=args.timestamp_mode,
    )

    print(f"Log salvo em: {artifacts.log_path}")
    if artifacts.video_path:
        print(f"Video salvo em: {artifacts.video_path}")
    if artifacts.report_json_path:
        print(f"Relatorio JSON salvo em: {artifacts.report_json_path}")
    if artifacts.report_md_path:
        print(f"Relatorio Markdown salvo em: {artifacts.report_md_path}")
    for chart_path in artifacts.chart_paths:
        print(f"Grafico HTML salvo em: {chart_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
