#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.canonical_baselines import lookup_canonical_baseline
from nanoresearch.experiments.deep_persona_runner import _compute_delta, _normalize_metric_scale_pair, _to_optional_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute shared canonical baselines and deltas for router-persona experiment records.")
    parser.add_argument("--input-dir", required=True, help="Batch output directory that contains shard_*/results.jsonl and per-assignment result.json files")
    parser.add_argument("--output", required=True, help="Output JSONL file with canonical baseline and delta fields applied")
    parser.add_argument("--summary-output", default="", help="Optional JSON summary path")
    return parser.parse_args()


def iter_result_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("shard_*/results.jsonl")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def recompute_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    metadata = dict(output.get("metadata") or {})
    question_id = str(output.get("question_id") or "").strip()
    primary_metric_name = str(output.get("primary_metric_name") or "").strip() or None
    canonical = lookup_canonical_baseline(question_id, primary_metric_name)

    output["raw_baseline_performance"] = output.get("baseline_performance")
    output["raw_delta_over_baseline"] = output.get("delta_over_baseline")

    if canonical is None:
        metadata["canonical_baseline_applied"] = False
        output["metadata"] = metadata
        return output

    final_performance = _to_optional_float(output.get("final_performance"))
    baseline_performance = _to_optional_float(canonical.get("baseline_value"))
    final_performance, baseline_performance = _normalize_metric_scale_pair(final_performance, baseline_performance)
    delta = _compute_delta(
        final_performance,
        baseline_performance,
        bool(canonical.get("higher_is_better", True)),
    )

    output["baseline_performance"] = baseline_performance
    output["delta_over_baseline"] = delta
    metadata["canonical_baseline_applied"] = True
    metadata["canonical_baseline_name"] = canonical.get("baseline_name")
    metadata["canonical_baseline_metric_name"] = canonical.get("metric_name")
    metadata["canonical_baseline_provenance"] = canonical.get("provenance_uri")
    output["metadata"] = metadata
    return output


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    records = iter_result_records(input_dir)
    recomputed = [recompute_record(record) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in recomputed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "record_count": len(recomputed),
        "canonical_baseline_applied_count": sum(1 for row in recomputed if (row.get("metadata") or {}).get("canonical_baseline_applied")),
        "delta_available_count": sum(1 for row in recomputed if row.get("delta_over_baseline") is not None),
        "questions": sorted({str(row.get("question_id") or "") for row in recomputed}),
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
