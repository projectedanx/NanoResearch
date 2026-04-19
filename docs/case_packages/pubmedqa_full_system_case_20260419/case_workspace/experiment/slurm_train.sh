#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/datasets"
DEFAULT_MODELS_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models"
DEFAULT_PUBMEDQA_JSON = (
    "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/"
    "ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/"
    "datasets/pubmedqa/data/ori_pqal.json"
)
DEFAULT_TEST_GROUND_TRUTH_JSON = (
    "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/"
    "ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/"
    "datasets/pubmedqa/data/test_ground_truth.json"
)
DEFAULT_PUBMEDBERT_DIR = (
    "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/"
    "ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/"
    "models/PubMedBERT-base"
)
DEFAULT_T5_SMALL_DIR = (
    "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/"
    "ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/"
    "models/T5-small"
)
DEFAULT_TRAIN_CMD = (
    "python train.py --run_name pubmedqa --mode cct --epochs 5 --batch_size 16 "
    "--lr 2e-5 --lambda_contrastive 0.5 --fp16"
)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM script for 1xA100 CCT training."
    )
    parser.add_argument("--job-name", type=str, default="pubmedqa_cct")
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--account", type=str, default="")
    parser.add_argument("--qos", type=str, default="")
    parser.add_argument("--time", type=str, default="48:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", type=str, default="80G")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-type", type=str, default="a100")
    parser.add_argument("--output-dir", type=str, default=str(Path.cwd() / "results"))
    parser.add_argument("--script-path", type=str, default=str(Path.cwd() / "slurm_train.sh"))
    parser.add_argument("--workdir", type=str, default=str(Path.cwd()))
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--pubmedqa-json", type=str, default=DEFAULT_PUBMEDQA_JSON)
    parser.add_argument("--test-ground-truth-json", type=str, default=DEFAULT_TEST_GROUND_TRUTH_JSON)
    parser.add_argument("--pubmedbert-dir", type=str, default=DEFAULT_PUBMEDBERT_DIR)
    parser.add_argument("--t5-small-dir", type=str, default=DEFAULT_T5_SMALL_DIR)
    parser.add_argument("--train-cmd", type=str, default=DEFAULT_TRAIN_CMD)
    parser.add_argument("--conda-env", type=str, default="")
    parser.add_argument("--module-load", type=str, default="", help="Comma-separated modules to load.")
    parser.add_argument("--submit", action="store_true", help="Submit the generated script with sbatch.")
    parser.add_argument("--dry-run", action="store_true", help="Inject --dry-run into train command.")
    parser.add_argument("--quick-eval", action="store_true", help="Inject --quick-eval into train command.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    required_paths = {
        "workdir": args.workdir,
        "data_dir": args.data_dir,
        "models_dir": args.models_dir,
        "pubmedqa_json": args.pubmedqa_json,
        "test_ground_truth_json": args.test_ground_truth_json,
        "pubmedbert_dir": args.pubmedbert_dir,
        "t5_small_dir": args.t5_small_dir,
    }
    missing = []
    for name, p in required_paths.items():
        if not Path(p).exists():
            missing.append(f"{name}: {p}")
    if missing:
        raise FileNotFoundError(
            "Required paths are missing:\n" + "\n".join(missing)
        )


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    slurm_dir = output_dir / "slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, slurm_dir


def _append_flag_if_missing(cmd: str, flag: str) -> str:
    tokens = shlex.split(cmd)
    if flag not in tokens:
        tokens.append(flag)
    return " ".join(shlex.quote(tok) for tok in tokens)


def _append_kv_if_missing(cmd: str, key: str, value: str) -> str:
    tokens = shlex.split(cmd)
    if key not in tokens:
        tokens.extend([key, value])
    return " ".join(shlex.quote(tok) for tok in tokens)


def prepare_train_cmd(args: argparse.Namespace) -> str:
    cmd = args.train_cmd.strip()
    if args.dry_run:
        cmd = _append_flag_if_missing(cmd, "--dry-run")
    if args.quick_eval:
        cmd = _append_flag_if_missing(cmd, "--quick-eval")

    cmd = _append_kv_if_missing(cmd, "--data_dir", args.data_dir)
    cmd = _append_kv_if_missing(cmd, "--model_name_or_path", args.pubmedbert_dir)
    cmd = _append_kv_if_missing(cmd, "--paraphraser_name_or_path", args.t5_small_dir)
    cmd = _append_kv_if_missing(cmd, "--pubmedqa_json", args.pubmedqa_json)
    cmd = _append_kv_if_missing(cmd, "--test_ground_truth_json", args.test_ground_truth_json)

    return cmd


def build_sbatch_script(args: argparse.Namespace, logs_dir: Path, slurm_dir: Path) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = logs_dir / f"{args.job_name}_{timestamp}.out"
    err_log = logs_dir / f"{args.job_name}_{timestamp}.err"

    train_cmd = prepare_train_cmd(args)

    module_lines = ""
    if args.module_load.strip():
        modules = [m.strip() for m in args.module_load.split(",") if m.strip()]
        if modules:
            module_lines = "\n".join([f"module load {m}" for m in modules])

    sbatch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --ntasks-per-node={args.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --gres=gpu:{args.gpu_type}:{args.gpus}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]

    if args.account.strip():
        sbatch_lines.append(f"#SBATCH --account={args.account}")
    if args.qos.strip():
        sbatch_lines.append(f"#SBATCH --qos={args.qos}")

    body_lines = [
        "",
        'echo "========== SLURM JOB START =========="',
        'echo "Host: $(hostname)"',
        'echo "Date: $(date)"',
        'echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"',
        'echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"',
        'echo "Working directory: ' + shlex.quote(args.workdir) + '"',
        "",
        f"cd {shlex.quote(args.workdir)}",
        "",
        "export TOKENIZERS_PARALLELISM=false",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"export HF_HOME={shlex.quote(str(slurm_dir / 'hf_cache'))}",
        f"export TRANSFORMERS_CACHE={shlex.quote(str(slurm_dir / 'hf_cache' / 'transformers'))}",
        f"export HF_DATASETS_CACHE={shlex.quote(str(slurm_dir / 'hf_cache' / 'datasets'))}",
        f"export PYTHONUNBUFFERED=1",
        "",
    ]

    if module_lines:
        body_lines.extend([module_lines, ""])

    if args.conda_env.strip():
        body_lines.extend(
            [
                'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then',
                '  source "$HOME/miniconda3/etc/profile.d/conda.sh"',
                "elif [ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then",
                '  source "$HOME/anaconda3/etc/profile.d/conda.sh"',
                "else",
                '  echo "Conda init script not found." >&2',
                "  exit 1",
                "fi",
                f"conda activate {shlex.quote(args.conda_env)}",
                "",
            ]
        )

    body_lines.extend(
        [
            "python -c \"import torch; print('Torch:', torch.__version__); "
            "print('CUDA available:', torch.cuda.is_available()); "
            "print('Device count:', torch.cuda.device_count())\"",
            "",
            f"mkdir -p {shlex.quote(args.output_dir)}",
            'echo "Training command:"',
            f"echo {shlex.quote(train_cmd)}",
            "",
            f"{train_cmd}",
            'exit_code=$?',
            'echo "Train exit code: ${exit_code}"',
            'echo "========== SLURM JOB END =========="',
            "exit ${exit_code}",
            "",
        ]
    )

    return "\n".join(sbatch_lines + body_lines)


def write_script(script_path: Path, content: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content, encoding="utf-8")
    os.chmod(script_path, 0o750)


def submit_script(script_path: Path) -> None:
    completed = subprocess.run(
        ["sbatch", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    logging.info("Submission success: %s", completed.stdout.strip())


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        validate_paths(args)
        output_dir = Path(args.output_dir)
        logs_dir, slurm_dir = ensure_output_dirs(output_dir)

        script_path = Path(args.script_path).resolve()
        content = build_sbatch_script(args, logs_dir, slurm_dir)
        write_script(script_path, content)

        logging.info("SLURM script written to: %s", script_path)
        logging.info("Logs directory: %s", logs_dir)

        if args.submit:
            submit_script(script_path)
        else:
            logging.info("Dry generation only. Use --submit to submit via sbatch.")
    except Exception:
        logging.exception("Fatal error while generating/submitting SLURM script.")
        sys.exit(1)


if __name__ == "__main__":
    main()