#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare shared elastic SLURM workers for router-persona deep experiments."
    )
    parser.add_argument("--manifest", required=True, help="Full experiment manifest JSON/JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Batch output directory shared by all workers.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional research config JSON shared by all workers. Leave empty to use assignment-level config_path from the manifest.",
    )
    parser.add_argument("--worker-count", type=int, default=6, help="How many worker sbatch scripts to generate.")
    parser.add_argument(
        "--worker-start-index",
        type=int,
        default=1,
        help="1-based index for the first worker. Use this to append more workers later.",
    )
    parser.add_argument("--job-prefix", default="nr6", help="Short SLURM job-name prefix.")
    parser.add_argument("--scheduler-root", default="", help="Shared scheduler root. Defaults to <output-dir>/_scheduler.")
    parser.add_argument("--python-bin", default="/mnt/petrelfs/xujinhang/anaconda3/bin/python")
    parser.add_argument("--repo-root", default="/mnt/petrelfs/xujinhang/nanoresearch_eval")
    parser.add_argument("--partition", default="belt_road")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="48G")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--quotatype", default="auto")
    parser.add_argument("--cuda-home", default="/mnt/petrelfs/share/test-cuda/cuda-12.1")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--claim-stale-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--failure-cooldown-seconds", type=int, default=600)
    parser.add_argument("--max-alignment-retries", type=int, default=1)
    parser.add_argument(
        "--disable-ideation-retrieval",
        action="store_true",
        help="Disable online retrieval during ideation.",
    )
    parser.add_argument(
        "--skip-completed-under",
        action="append",
        default=[],
        help="Extra completed-result roots to pass through to the worker command.",
    )
    parser.add_argument("--http-proxy", default="")
    parser.add_argument("--https-proxy", default="")
    parser.add_argument("--openalex-api-key", default="")
    return parser.parse_args()


def shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def render_worker_script(args: argparse.Namespace, *, worker_index: int, job_name: str, out_log: Path, err_log: Path) -> str:
    scheduler_root = Path(args.scheduler_root) if args.scheduler_root else Path(args.output_dir) / "_scheduler"
    cmd = [
        shell_quote(args.python_bin),
        "tools/run_router_persona_deep_experiment.py",
        "--manifest",
        shell_quote(str(Path(args.manifest))),
        "--output-dir",
        shell_quote(str(Path(args.output_dir))),
        "--elastic",
        "--scheduler-root",
        shell_quote(str(scheduler_root)),
        "--worker-id",
        shell_quote(f"{args.job_prefix}-worker{worker_index:02d}"),
        "--heartbeat-seconds",
        str(max(1, int(args.heartbeat_seconds))),
        "--claim-stale-seconds",
        str(max(60, int(args.claim_stale_seconds))),
        "--poll-seconds",
        str(max(1, int(args.poll_seconds))),
        "--failure-cooldown-seconds",
        str(max(1, int(args.failure_cooldown_seconds))),
        "--max-alignment-retries",
        str(max(0, int(args.max_alignment_retries))),
    ]
    if str(args.config or "").strip():
        cmd.extend(["--config", shell_quote(str(Path(args.config)))])
    if args.disable_ideation_retrieval:
        cmd.append("--disable-ideation-retrieval")
    for root in list(args.skip_completed_under or []):
        if str(root).strip():
            cmd.extend(["--skip-completed-under", shell_quote(str(Path(root)))])

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(args.cpus_per_task)}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --gres={args.gres}",
        f"#SBATCH --quotatype={args.quotatype}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
        "",
        "set -euo pipefail",
        f"cd {shell_quote(str(Path(args.repo_root)))}",
        "export PYTHONPATH=.",
        f"export CUDA_HOME={shell_quote(args.cuda_home)}",
        'export PATH="$CUDA_HOME/bin:$PATH"',
        'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"',
    ]
    if str(args.http_proxy or "").strip():
        lines.extend(
            [
                f"export http_proxy={shell_quote(args.http_proxy)}",
                'export HTTP_PROXY="$http_proxy"',
            ]
        )
    if str(args.https_proxy or "").strip():
        lines.extend(
            [
                f"export https_proxy={shell_quote(args.https_proxy)}",
                'export HTTPS_PROXY="$https_proxy"',
            ]
        )
    if str(args.openalex_api_key or "").strip():
        lines.append(f"export OPENALEX_API_KEY={shell_quote(args.openalex_api_key)}")
    lines.extend(
        [
            f'echo "===== START {job_name} ====="',
            " ".join(cmd),
            f'echo "===== END {job_name} ====="',
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    scheduler_root = Path(args.scheduler_root) if args.scheduler_root else output_dir / "_scheduler"
    sbatch_dir = output_dir / "sbatch"
    slurm_logs_dir = output_dir / "slurm_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_root.mkdir(parents=True, exist_ok=True)
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)

    submit_plan_path = output_dir / "submit_plan.json"
    existing_plan = []
    if submit_plan_path.exists():
        try:
            existing_plan = json.loads(submit_plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_plan = []
    if not isinstance(existing_plan, list):
        existing_plan = []

    new_entries = []
    for offset in range(max(0, int(args.worker_count))):
        worker_index = int(args.worker_start_index) + offset
        job_name = f"{args.job_prefix}_w{worker_index:02d}"
        sbatch_path = sbatch_dir / f"{worker_index:02d}_{job_name}.sbatch"
        out_log = slurm_logs_dir / f"{worker_index:02d}_{job_name}_%j.out"
        err_log = slurm_logs_dir / f"{worker_index:02d}_{job_name}_%j.err"
        sbatch_path.write_text(
            render_worker_script(args, worker_index=worker_index, job_name=job_name, out_log=out_log, err_log=err_log),
            encoding="utf-8",
        )
        new_entries.append(
            {
                "worker_index": worker_index,
                "job_name": job_name,
                "scheduler_root": str(scheduler_root),
                "sbatch": str(sbatch_path),
                "stdout": str(out_log),
                "stderr": str(err_log),
            }
        )

    plan_by_index = {int(item.get("worker_index") or 0): item for item in existing_plan if isinstance(item, dict)}
    for item in new_entries:
        plan_by_index[int(item["worker_index"])] = item
    merged_plan = [plan_by_index[index] for index in sorted(plan_by_index)]
    submit_plan_path.write_text(json.dumps(merged_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    batch_summary = {
        "manifest": str(Path(args.manifest)),
        "output_dir": str(output_dir),
        "scheduler_root": str(scheduler_root),
        "config": str(Path(args.config)) if str(args.config or "").strip() else "",
        "worker_count_total": len(merged_plan),
        "worker_count_generated_now": len(new_entries),
        "worker_start_index": int(args.worker_start_index),
        "job_prefix": args.job_prefix,
        "partition": args.partition,
        "cpus_per_task": int(args.cpus_per_task),
        "mem": args.mem,
        "gres": args.gres,
        "quotatype": args.quotatype,
        "heartbeat_seconds": int(args.heartbeat_seconds),
        "claim_stale_seconds": int(args.claim_stale_seconds),
        "poll_seconds": int(args.poll_seconds),
        "failure_cooldown_seconds": int(args.failure_cooldown_seconds),
        "disable_ideation_retrieval": bool(args.disable_ideation_retrieval),
        "skip_completed_under": list(args.skip_completed_under or []),
        "openalex_api_key_configured": bool(str(args.openalex_api_key or "").strip()),
    }
    (output_dir / "elastic_batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
