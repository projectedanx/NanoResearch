#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.router_persona_eval import aggregate_experiment_results


JSON_EXTENSIONS = {".json", ".jsonl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 10-persona NanoResearch experiment results.")
    parser.add_argument("--input", action="append", required=True, help="Input result file or directory; may be repeated")
    parser.add_argument("--output-dir", required=True, help="Directory to write summary artifacts")
    return parser.parse_args()


def iter_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(child for child in path.iterdir() if child.suffix in JSON_EXTENSIONS))
        else:
            files.append(path)
    return files


def load_records(path: Path) -> list[dict[str, Any]]:
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
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"Unsupported JSON payload in {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = iter_input_files(args.input)
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(load_records(path))

    summary = aggregate_experiment_results(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "main_table.json", summary["main_table"])
    write_json(output_dir / "efficiency_table.json", summary["efficiency_table"])
    write_json(output_dir / "persona_breakdown.json", summary["per_persona"])
    write_json(output_dir / "appendix_baselines.json", summary["appendix_baselines"])
    write_json(output_dir / "ablation_contributions.json", summary["ablation_contributions"])
    (output_dir / "main_table.tex").write_text(summary["main_table"]["latex"], encoding="utf-8")
    (output_dir / "efficiency_table.tex").write_text(summary["efficiency_table"]["latex"], encoding="utf-8")

    print(
        json.dumps(
            {
                "record_count": summary["record_count"],
                "question_count": summary["question_count"],
                "personas": summary["personas"],
                "main_table_methods": [row["method"] for row in summary["main_table"]["rows"]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
