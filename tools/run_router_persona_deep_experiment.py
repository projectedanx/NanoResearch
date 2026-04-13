from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.deep_persona_runner import run_manifest
from nanoresearch.experiments.router_persona_eval import VARIANT_BY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run deep-pipeline persona experiments from a manifest.')
    parser.add_argument('--manifest', required=True, help='Path to a JSON or JSONL experiment manifest.')
    parser.add_argument('--output-dir', required=True, help='Output directory for workspaces and result records.')
    parser.add_argument('--config', default=None, help='Optional ResearchConfig JSON path.')
    parser.add_argument('--persona', action='append', default=[], help='Limit to one or more persona_id values.')
    parser.add_argument('--variant', action='append', default=[], help='Limit to one or more variant_name values.')
    parser.add_argument('--question', action='append', default=[], help='Limit to one or more question_id values.')
    parser.add_argument('--limit', type=int, default=0, help='Optional max number of assignments to run after filtering.')
    parser.add_argument('--max-alignment-retries', type=int, default=1, help='Maximum additional full-pipeline retries after a failed alignment judgment.')
    parser.add_argument('--skip-sdpo-variants', action='store_true', help='Skip variants whose semantics require same-router hindsight SDPO.')
    parser.add_argument(
        '--disable-ideation-retrieval',
        action='store_true',
        help='Run IDEATION in eval-fast mode using only the manifest/topic context, without online literature or GitHub retrieval.',
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'Manifest not found: {path}')
    if path.suffix == '.jsonl':
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get('assignments'), list):
        return list(data['assignments'])
    raise ValueError(f'Unsupported manifest structure in {path}')


def filter_assignments(
    rows: list[dict[str, Any]],
    *,
    personas: list[str],
    variants: list[str],
    questions: list[str],
    skip_sdpo_variants: bool,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    persona_set = {item.strip() for item in personas if item.strip()}
    variant_set = {item.strip() for item in variants if item.strip()}
    question_set = {item.strip() for item in questions if item.strip()}
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in rows:
        if persona_set and str(row.get('persona_id')) not in persona_set:
            continue
        if variant_set and str(row.get('variant_name')) not in variant_set:
            continue
        question_id = str((row.get('question') or {}).get('question_id') or row.get('question_id') or '')
        if question_set and question_id not in question_set:
            continue
        variant_name = str(row.get('variant_name') or '')
        variant = VARIANT_BY_NAME.get(variant_name)
        if skip_sdpo_variants and variant and variant.same_router_hindsight_sdpo:
            skipped.append({
                'assignment_id': row.get('assignment_id'),
                'persona_id': row.get('persona_id'),
                'variant_name': variant_name,
                'question_id': question_id,
                'reason': 'requires_same_router_hindsight_sdpo',
            })
            continue
        kept.append(row)
        if limit and len(kept) >= limit:
            break
    return kept, skipped


async def _main_async(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    selected, skipped = filter_assignments(
        rows,
        personas=args.persona,
        variants=args.variant,
        questions=args.question,
        skip_sdpo_variants=args.skip_sdpo_variants,
        limit=max(0, int(args.limit or 0)),
    )
    if not selected:
        raise RuntimeError('No assignments remain after filtering.')

    (output_dir / 'selected_assignments.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'skipped_assignments.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding='utf-8')

    records = await run_manifest(
        selected,
        output_dir=output_dir,
        config_path=args.config,
        max_alignment_retries=max(0, int(args.max_alignment_retries)),
        disable_ideation_retrieval=bool(args.disable_ideation_retrieval),
    )
    summary = {
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
        'selected_count': len(selected),
        'skipped_count': len(skipped),
        'result_count': len(records),
        'personas': sorted({str(row.get('persona_id') or '') for row in selected}),
        'variants': sorted({str(row.get('variant_name') or '') for row in selected}),
        'question_ids': sorted({str((row.get('question') or {}).get('question_id') or '') for row in selected}),
        'max_alignment_retries': int(args.max_alignment_retries),
        'disable_ideation_retrieval': bool(args.disable_ideation_retrieval),
    }
    (output_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
