#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.router_persona_eval import build_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 10-persona repeated NanoResearch experiment manifest.")
    parser.add_argument("--questions", required=True, help="Input questions file (.json array or .jsonl)")
    parser.add_argument("--output", required=True, help="Output manifest jsonl path")
    parser.add_argument("--summary-output", default="", help="Optional summary json path")
    parser.add_argument("--persona", action="append", default=[], help="Optional persona override; may be passed multiple times")
    parser.add_argument(
        "--include-appendix-baseline",
        dest="include_appendix_baseline",
        action="store_true",
        help="Include the context-informed appendix baseline in the manifest",
    )
    parser.add_argument(
        "--no-include-appendix-baseline",
        dest="include_appendix_baseline",
        action="store_false",
        help="Exclude the context-informed appendix baseline",
    )
    parser.set_defaults(include_appendix_baseline=True)
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError("Questions file must be a JSON array or JSONL file")


def main() -> None:
    args = parse_args()
    questions = load_questions(Path(args.questions))
    manifest = build_experiment_manifest(
        questions,
        personas=args.persona or None,
        include_appendix_baseline=args.include_appendix_baseline,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(manifest),
        "question_count": len(questions),
        "personas": sorted({row["persona_id"] for row in manifest}),
        "variants": sorted({row["variant_name"] for row in manifest}),
        "include_appendix_baseline": args.include_appendix_baseline,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
