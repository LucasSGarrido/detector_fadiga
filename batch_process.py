from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class BatchItem:
    path: Path
    return_code: int
    stdout: str
    stderr: str

    @property
    def status(self) -> str:
        return "ok" if self.return_code == 0 else "erro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processa uma pasta de videos offline")
    parser.add_argument("--input-dir", required=True, help="Pasta com videos para analisar")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--extensions", nargs="*", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    videos = list(iter_videos(input_dir, args.extensions))

    if not videos:
        print(f"Nenhum video encontrado em: {input_dir}")
        return 1

    results: list[BatchItem] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Processando: {video}")
        command = build_command(
            video,
            config=Path(args.config),
            save_video=args.save_video,
            max_frames=args.max_frames,
        )
        completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        item = BatchItem(
            path=video,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        results.append(item)

        if completed.returncode != 0:
            print(f"Erro ao processar {video}")
            print(completed.stderr)
            if not args.continue_on_error:
                break

    summary_path = write_batch_summary(results)
    print(f"Resumo do lote salvo em: {summary_path}")

    return 0 if all(item.return_code == 0 for item in results) else 1


def iter_videos(input_dir: Path, extensions: list[str] | tuple[str, ...]):
    normalized = {extension.lower() for extension in extensions}
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized:
            yield path


def build_command(
    video: Path,
    *,
    config: Path,
    save_video: bool = False,
    max_frames: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "app.py"),
        "--source",
        str(video),
        "--config",
        str(config),
        "--headless",
        "--no-sound",
        "--timestamp-mode",
        "video",
    ]

    if save_video:
        command.append("--save-video")
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])

    return command


def write_batch_summary(results: list[BatchItem]) -> Path:
    output_dir = ROOT / "outputs" / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video", "status", "return_code", "stdout", "stderr"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "video": str(item.path),
                    "status": item.status,
                    "return_code": item.return_code,
                    "stdout": item.stdout.strip(),
                    "stderr": item.stderr.strip(),
                }
            )

    return path


if __name__ == "__main__":
    raise SystemExit(main())
