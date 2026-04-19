from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from nanoresearch.experiments.deep_persona_runner import run_manifest
from nanoresearch.experiments.elastic_scheduler import default_worker_id, run_elastic_manifest
from nanoresearch.experiments.router_persona_eval import VARIANT_BY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run deep-pipeline persona experiments from a manifest.')
    parser.add_argument('--manifest', required=True, help='Path to a JSON or JSONL experiment manifest.')
    parser.add_argument('--output-dir', required=True, help='Output directory for workspaces and result records.')
    parser.add_argument('--config', default=None, help='Optional ResearchConfig JSON path.')
    parser.add_argument(
        '--skip-completed-under',
        action='append',
        default=[],
        help='Skip assignments whose assignment_id already has a result.json anywhere under the given output root. Repeatable.',
    )
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
    parser.add_argument(
        '--elastic',
        action='store_true',
        help='Use the shared elastic scheduler instead of a static one-shot assignment list.',
    )
    parser.add_argument(
        '--scheduler-root',
        default='',
        help='Shared scheduler directory for elastic mode. Defaults to <output-dir>/_scheduler.',
    )
    parser.add_argument(
        '--worker-id',
        default='',
        help='Stable worker id for elastic mode. Defaults to hostname + pid / SLURM job id.',
    )
    parser.add_argument(
        '--heartbeat-seconds',
        type=int,
        default=60,
        help='Elastic mode heartbeat interval while a worker owns a claim.',
    )
    parser.add_argument(
        '--claim-stale-seconds',
        type=int,
        default=1800,
        help='Elastic mode stale-claim timeout used when a worker disappears mid-assignment.',
    )
    parser.add_argument(
        '--poll-seconds',
        type=int,
        default=30,
        help='Elastic mode idle polling interval when no assignment is immediately claimable.',
    )
    parser.add_argument(
        '--failure-cooldown-seconds',
        type=int,
        default=600,
        help='Elastic mode cooldown after an assignment raises before it may be claimed again.',
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


def load_completed_assignment_ids(roots: list[str]) -> set[str]:
    completed: set[str] = set()
    for root_str in roots:
        root = Path(root_str).expanduser()
        if not root.exists():
            continue
        indexed = False
        direct_results = root / 'results.jsonl'
        if direct_results.is_file():
            indexed = True
            try:
                for line in direct_results.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    assignment_id = str(payload.get('assignment_id') or '').strip()
                    if assignment_id:
                        completed.add(assignment_id)
            except Exception:
                pass
        for child in root.iterdir():
            if not child.is_dir():
                continue
            results_jsonl = child / 'results.jsonl'
            if not results_jsonl.is_file():
                continue
            indexed = True
            try:
                for line in results_jsonl.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    assignment_id = str(payload.get('assignment_id') or '').strip()
                    if assignment_id:
                        completed.add(assignment_id)
            except Exception:
                continue
        if indexed:
            continue
        for result_path in root.glob('**/result.json'):
            try:
                payload = json.loads(result_path.read_text(encoding='utf-8'))
            except Exception:
                continue
            assignment_id = str(payload.get('assignment_id') or '').strip()
            if assignment_id:
                completed.add(assignment_id)
    return completed


def filter_assignments(
    rows: list[dict[str, Any]],
    *,
    completed_assignment_ids: set[str],
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
        assignment_id = str(row.get('assignment_id') or '').strip()
        if assignment_id and assignment_id in completed_assignment_ids:
            skipped.append({
                'assignment_id': assignment_id,
                'persona_id': row.get('persona_id'),
                'variant_name': row.get('variant_name'),
                'question_id': str((row.get('question') or {}).get('question_id') or row.get('question_id') or ''),
                'reason': 'already_completed',
            })
            continue
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
    scheduler_root = Path(args.scheduler_root) if args.scheduler_root else output_dir / '_scheduler'
    existing_scheduler_selected = scheduler_root / 'selected_assignments.json'
    existing_scheduler_skipped = scheduler_root / 'skipped_assignments.json'

    rows = load_manifest(manifest_path)
    completed_roots = list(dict.fromkeys([str(output_dir), *(list(args.skip_completed_under or []))]))
    completed_assignment_ids = load_completed_assignment_ids(completed_roots)
    if args.elastic and existing_scheduler_selected.is_file():
        selected = load_manifest(existing_scheduler_selected)
        skipped = load_manifest(existing_scheduler_skipped) if existing_scheduler_skipped.is_file() else []
    else:
        selected, skipped = filter_assignments(
            rows,
            completed_assignment_ids=completed_assignment_ids,
            personas=args.persona,
            variants=args.variant,
            questions=args.question,
            skip_sdpo_variants=args.skip_sdpo_variants,
            limit=max(0, int(args.limit or 0)),
        )
    if not selected:
        raise RuntimeError('No assignments remain after filtering.')

    if args.elastic:
        worker_id = str(args.worker_id or default_worker_id())
        summary = await run_elastic_manifest(
            selected,
            output_dir=output_dir,
            skipped_assignments=skipped,
            seed_completed_assignment_ids=completed_assignment_ids,
            scheduler_root=scheduler_root,
            worker_id=worker_id,
            config_path=args.config,
            manifest_path=manifest_path,
            max_alignment_retries=max(0, int(args.max_alignment_retries)),
            disable_ideation_retrieval=bool(args.disable_ideation_retrieval),
            heartbeat_seconds=max(1, int(args.heartbeat_seconds)),
            claim_stale_seconds=max(60, int(args.claim_stale_seconds)),
            poll_seconds=max(1, int(args.poll_seconds)),
            failure_cooldown_seconds=max(1, int(args.failure_cooldown_seconds)),
        )
        scheduler_root.mkdir(parents=True, exist_ok=True)
        worker_summary = {
            'mode': 'elastic',
            'manifest': str(manifest_path),
            'output_dir': str(output_dir),
            'scheduler_root': str(scheduler_root),
            'worker_id': worker_id,
            'selected_count': len(selected),
            'skipped_count': len(skipped),
            'completed_assignment_count': len(completed_assignment_ids),
            'status': summary,
            'max_alignment_retries': int(args.max_alignment_retries),
            'disable_ideation_retrieval': bool(args.disable_ideation_retrieval),
            'skip_completed_under': completed_roots,
        }
        worker_summary_path = scheduler_root / f'worker_summary_{worker_id}.json'
        worker_summary_path.write_text(json.dumps(worker_summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(worker_summary, ensure_ascii=False, indent=2))
        return 0

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
        'mode': 'static',
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
        'selected_count': len(selected),
        'skipped_count': len(skipped),
        'result_count': len(records),
        'completed_assignment_count': len(completed_assignment_ids),
        'personas': sorted({str(row.get('persona_id') or '') for row in selected}),
        'variants': sorted({str(row.get('variant_name') or '') for row in selected}),
        'question_ids': sorted({str((row.get('question') or {}).get('question_id') or '') for row in selected}),
        'max_alignment_retries': int(args.max_alignment_retries),
        'disable_ideation_retrieval': bool(args.disable_ideation_retrieval),
        'skip_completed_under': completed_roots,
    }
    (output_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
