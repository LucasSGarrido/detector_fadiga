from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.evaluation import (
    evaluate_files,
    write_evaluation_json,
    write_evaluation_markdown,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um log com labels manuais")
    parser.add_argument("--log", required=True, help="CSV gerado em outputs/logs/")
    parser.add_argument("--labels", required=True, help="CSV com start_seconds,end_seconds,label")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "evaluations"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_files(args.log, args.labels)

    output_dir = Path(args.output_dir)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"evaluation_{session_id}.json"
    markdown_path = output_dir / f"evaluation_{session_id}.md"

    write_evaluation_json(result, json_path)
    write_evaluation_markdown(result, markdown_path)

    print(f"Frames avaliados: {result.frames_evaluated}")
    print(f"Acuracia: {result.accuracy:.2%}")
    print(f"Falsos alertas/min: {result.false_alarms_per_minute:.2f}")
    print(f"Avaliacao JSON salva em: {json_path}")
    print(f"Avaliacao Markdown salva em: {markdown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
