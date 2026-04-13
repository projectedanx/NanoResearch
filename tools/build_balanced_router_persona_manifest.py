#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.router_persona_eval import DEFAULT_PERSONA_IDS, build_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced router-persona manifest where each topic is assigned to two personas in a ring pairing."
    )
    parser.add_argument("--questions", required=True, help="Input questions file (.json array or .jsonl)")
    parser.add_argument("--output", required=True, help="Output manifest jsonl path")
    parser.add_argument("--summary-output", default="", help="Optional summary json path")
    parser.add_argument(
        "--pairings-output",
        default="",
        help="Optional pairing summary json path",
    )
    parser.add_argument(
        "--persona",
        action="append",
        default=[],
        help="Optional ordered persona override; may be passed multiple times",
    )
    parser.add_argument(
        "--evolution-rounds",
        type=int,
        default=1,
        help="How many sequential evolution rounds to generate per persona x variant x question chain",
    )
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
    parser.set_defaults(include_appendix_baseline=False)
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
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


def ordered_personas(raw_personas: list[str]) -> list[str]:
    personas = [str(persona).strip() for persona in (raw_personas or DEFAULT_PERSONA_IDS) if str(persona).strip()]
    if not personas:
        raise ValueError("At least one persona is required")
    if len(set(personas)) != len(personas):
        raise ValueError("Persona list contains duplicates; provide each persona at most once")
    return personas


def build_ring_pairings(questions: list[dict[str, Any]], personas: list[str]) -> list[dict[str, Any]]:
    if len(questions) != len(personas):
        raise ValueError(
            f"Balanced ring pairing requires question_count == persona_count, got {len(questions)} questions and {len(personas)} personas"
        )
    pairings: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        pairings.append(
            {
                "question_id": str(question.get("question_id") or ""),
                "personas": [
                    personas[index],
                    personas[(index + 1) % len(personas)],
                ],
            }
        )
    return pairings


def main() -> None:
    args = parse_args()
    questions = load_questions(Path(args.questions))
    personas = ordered_personas(args.persona)
    pairings = build_ring_pairings(questions, personas)

    question_map = {str(question["question_id"]): question for question in questions}
    manifest: list[dict[str, Any]] = []
    for pairing in pairings:
        question = question_map[pairing["question_id"]]
        manifest.extend(
            build_experiment_manifest(
                [question],
                personas=pairing["personas"],
                include_appendix_baseline=args.include_appendix_baseline,
                evolution_rounds=max(1, args.evolution_rounds),
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(manifest),
        "question_count": len(questions),
        "persona_count": len(personas),
        "personas": personas,
        "question_ids": [str(question["question_id"]) for question in questions],
        "pairings": pairings,
        "variants": sorted({row["variant_name"] for row in manifest}),
        "evolution_rounds": max(1, args.evolution_rounds),
        "include_appendix_baseline": bool(args.include_appendix_baseline),
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.pairings_output:
        pairings_path = Path(args.pairings_output)
        pairings_path.parent.mkdir(parents=True, exist_ok=True)
        pairings_path.write_text(json.dumps(pairings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
