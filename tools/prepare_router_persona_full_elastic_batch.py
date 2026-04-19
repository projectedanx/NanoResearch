#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_TEMPLATE_BY_PERSONA = {
    "nlp_conference_exploratory": "nlp_conference_pragmatic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a full router-persona elastic SLURM batch with assignment-level persona configs."
    )
    parser.add_argument("--manifest", required=True, help="Full experiment manifest JSON/JSONL path.")
    parser.add_argument("--batch-root", required=True, help="Root directory for prepared configs, manifest, and run scripts.")
    parser.add_argument("--source-config-dir", required=True, help="Directory containing per-persona config JSON files.")
    parser.add_argument(
        "--router-model-root",
        default="/mnt/dhwfile/raise/user/xujinhang/nanoresearch/tmp/router_sdpo_offpolicy_runs/per_persona",
        help="Root directory containing per-persona router SDPO checkpoints.",
    )
    parser.add_argument("--worker-count", type=int, default=6, help="How many worker sbatch scripts to generate now.")
    parser.add_argument(
        "--worker-start-index",
        type=int,
        default=1,
        help="1-based worker index for this generation wave. Use >1 when appending more workers later.",
    )
    parser.add_argument("--job-prefix", default="nre6", help="Short job-name prefix for SLURM workers.")
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
        help="Force ideation retrieval off in copied configs and worker launch.",
    )
    parser.add_argument(
        "--enable-ideation-retrieval",
        action="store_true",
        help="Force ideation retrieval on in copied configs, overriding any source config default.",
    )
    parser.add_argument(
        "--skip-completed-under",
        action="append",
        default=[],
        help="Existing result roots to skip when workers run. Repeatable.",
    )
    parser.add_argument("--http-proxy", default="")
    parser.add_argument("--https-proxy", default="")
    parser.add_argument("--openalex-api-key", default="")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("assignments"), list):
        return list(data["assignments"])
    raise ValueError(f"Unsupported manifest structure in {path}")


def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_router_model_path(router_model_root: Path, persona_id: str) -> Path:
    persona_root = router_model_root / persona_id
    if not persona_root.exists():
        raise FileNotFoundError(f"Router model directory not found for persona {persona_id}: {persona_root}")

    checkpoints = sorted(persona_root.glob("*/train/epoch-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No router SDPO checkpoint found for persona {persona_id} under {persona_root}")
    return checkpoints[-1]


def prepare_persona_config(
    *,
    persona_id: str,
    source_config_dir: Path,
    output_config_dir: Path,
    router_model_root: Path,
    disable_ideation_retrieval: bool,
    enable_ideation_retrieval: bool,
) -> dict[str, Any]:
    source_path = source_config_dir / f"{persona_id}.json"
    template_persona = ""
    synthesized = False

    if source_path.is_file():
        config_data = load_json(source_path)
    else:
        template_persona = DEFAULT_TEMPLATE_BY_PERSONA.get(persona_id, "")
        if not template_persona:
            raise FileNotFoundError(
                f"Missing config for persona {persona_id} in {source_config_dir}, and no fallback template is defined."
            )
        template_path = source_config_dir / f"{template_persona}.json"
        if not template_path.is_file():
            raise FileNotFoundError(f"Fallback template missing for persona {persona_id}: {template_path}")
        config_data = load_json(template_path)
        synthesized = True

    research = dict(config_data.get("research") or {})
    research["slurm_default_time"] = ""
    if disable_ideation_retrieval and enable_ideation_retrieval:
        raise ValueError("Cannot set both disable_ideation_retrieval and enable_ideation_retrieval.")
    if enable_ideation_retrieval:
        research["ideation_disable_retrieval"] = False
    elif disable_ideation_retrieval:
        research["ideation_disable_retrieval"] = True

    if synthesized:
        research["router_sdpo_model_path"] = str(resolve_router_model_path(router_model_root, persona_id))

    config_data["research"] = research
    output_path = output_config_dir / f"{persona_id}.json"
    write_json(output_path, config_data)

    return {
        "persona_id": persona_id,
        "config_path": str(output_path),
        "source_path": str(source_path if source_path.is_file() else (source_config_dir / f"{template_persona}.json")),
        "source_persona": template_persona or persona_id,
        "synthesized_from_template": synthesized,
        "router_sdpo_model_path": str(research.get("router_sdpo_model_path") or ""),
        "ideation_disable_retrieval": bool(research.get("ideation_disable_retrieval")),
        "slurm_default_time": str(research.get("slurm_default_time") or ""),
    }


def build_assignment_manifest(rows: list[dict[str, Any]], config_by_persona: dict[str, Path]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        persona_id = str(row.get("persona_id") or "").strip()
        if not persona_id:
            raise ValueError(f"Assignment missing persona_id: {row}")
        config_path = config_by_persona.get(persona_id)
        if config_path is None:
            raise KeyError(f"No prepared config for persona {persona_id}")
        enriched = dict(row)
        enriched["config_path"] = str(config_path)
        prepared.append(enriched)
    return prepared


def build_prepare_command(args: argparse.Namespace, prepared_manifest_path: Path, run_dir: Path) -> list[str]:
    cmd = [
        str(Path(args.python_bin)),
        str(Path(args.repo_root) / "tools" / "prepare_router_persona_elastic_slurm_batch.py"),
        "--manifest",
        str(prepared_manifest_path),
        "--output-dir",
        str(run_dir),
        "--worker-count",
        str(max(0, int(args.worker_count))),
        "--worker-start-index",
        str(int(args.worker_start_index)),
        "--job-prefix",
        args.job_prefix,
        "--python-bin",
        str(Path(args.python_bin)),
        "--repo-root",
        str(Path(args.repo_root)),
        "--partition",
        args.partition,
        "--cpus-per-task",
        str(int(args.cpus_per_task)),
        "--mem",
        args.mem,
        "--gres",
        args.gres,
        "--quotatype",
        args.quotatype,
        "--cuda-home",
        args.cuda_home,
        "--heartbeat-seconds",
        str(int(args.heartbeat_seconds)),
        "--claim-stale-seconds",
        str(int(args.claim_stale_seconds)),
        "--poll-seconds",
        str(int(args.poll_seconds)),
        "--failure-cooldown-seconds",
        str(int(args.failure_cooldown_seconds)),
        "--max-alignment-retries",
        str(int(args.max_alignment_retries)),
    ]
    if args.disable_ideation_retrieval:
        cmd.append("--disable-ideation-retrieval")
    for root in list(args.skip_completed_under or []):
        if str(root).strip():
            cmd.extend(["--skip-completed-under", str(Path(root))])
    if str(args.http_proxy or "").strip():
        cmd.extend(["--http-proxy", args.http_proxy])
    if str(args.https_proxy or "").strip():
        cmd.extend(["--https-proxy", args.https_proxy])
    if str(args.openalex_api_key or "").strip():
        cmd.extend(["--openalex-api-key", args.openalex_api_key])
    return cmd


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    batch_root = Path(args.batch_root)
    source_config_dir = Path(args.source_config_dir)
    router_model_root = Path(args.router_model_root)
    repo_root = Path(args.repo_root)

    rows = load_manifest(manifest_path)
    personas = sorted({str(row.get("persona_id") or "").strip() for row in rows if str(row.get("persona_id") or "").strip()})
    if not personas:
        raise RuntimeError("No persona_id values found in manifest.")

    configs_dir = batch_root / "configs"
    manifests_dir = batch_root / "manifests"
    run_dir = batch_root / "run"
    configs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_summaries: list[dict[str, Any]] = []
    config_by_persona: dict[str, Path] = {}
    for persona_id in personas:
        summary = prepare_persona_config(
            persona_id=persona_id,
            source_config_dir=source_config_dir,
            output_config_dir=configs_dir,
            router_model_root=router_model_root,
            disable_ideation_retrieval=bool(args.disable_ideation_retrieval),
            enable_ideation_retrieval=bool(args.enable_ideation_retrieval),
        )
        config_summaries.append(summary)
        config_by_persona[persona_id] = Path(summary["config_path"])

    prepared_rows = build_assignment_manifest(rows, config_by_persona)
    prepared_manifest_path = manifests_dir / "manifest_with_assignment_configs.jsonl"
    write_manifest_jsonl(prepared_manifest_path, prepared_rows)
    write_manifest_jsonl(manifests_dir / "manifest_source_copy.jsonl", rows)

    prepare_cmd = build_prepare_command(args, prepared_manifest_path, run_dir)
    subprocess.run(prepare_cmd, check=True)

    submit_plan_path = run_dir / "submit_plan.json"
    submit_plan = json.loads(submit_plan_path.read_text(encoding="utf-8")) if submit_plan_path.is_file() else []
    submit_lines = ["#!/bin/bash", "set -euo pipefail"]
    for item in submit_plan:
        sbatch_path = str(item.get("sbatch") or "").strip()
        if sbatch_path:
            submit_lines.append(f"sbatch {sbatch_path}")
    (batch_root / "submit_all.sh").write_text("\n".join(submit_lines) + "\n", encoding="utf-8")

    assignment_count_by_persona = Counter(str(row.get("persona_id") or "") for row in rows)
    assignment_count_by_variant = Counter(str(row.get("variant_name") or "") for row in rows)
    assignment_count_by_round = Counter(int(row.get("evolution_round") or 0) for row in rows)
    question_ids = sorted({str((row.get("question") or {}).get("question_id") or row.get("question_id") or "") for row in rows})

    summary = {
        "source_manifest": str(manifest_path),
        "prepared_manifest": str(prepared_manifest_path),
        "batch_root": str(batch_root),
        "run_dir": str(run_dir),
        "source_config_dir": str(source_config_dir),
        "router_model_root": str(router_model_root),
        "worker_count_generated_now": int(args.worker_count),
        "worker_start_index": int(args.worker_start_index),
        "job_prefix": args.job_prefix,
        "persona_count": len(personas),
        "question_count": len(question_ids),
        "assignment_count": len(rows),
        "assignment_count_by_persona": dict(sorted(assignment_count_by_persona.items())),
        "assignment_count_by_variant": dict(sorted(assignment_count_by_variant.items())),
        "assignment_count_by_round": {str(k): v for k, v in sorted(assignment_count_by_round.items())},
        "question_ids": question_ids,
        "config_summaries": config_summaries,
        "submit_plan_path": str(submit_plan_path),
        "submit_script": str(batch_root / "submit_all.sh"),
        "disable_ideation_retrieval": bool(args.disable_ideation_retrieval),
        "enable_ideation_retrieval": bool(args.enable_ideation_retrieval),
        "skip_completed_under": list(args.skip_completed_under or []),
        "openalex_api_key_configured": bool(str(args.openalex_api_key or "").strip()),
    }
    write_json(batch_root / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
