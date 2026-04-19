#!/bin/bash
#SBATCH --job-name=pubmedqa_cct_pu
#SBATCH --partition=belt_road
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --mem=64G
#SBATCH --export=ALL,LANG=C,LC_ALL=C
#SBATCH --output=/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment/logs/slurm_%j.out
#SBATCH --error=/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment/logs/slurm_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================"

PYTHON_BIN="/mnt/petrelfs/xujinhang/anaconda3/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "[ERROR] Could not find a usable python executable." >&2
        exit 1
    fi
fi

PIP_BIN="$(dirname "$PYTHON_BIN")/pip"
if [ ! -x "$PIP_BIN" ]; then
    PIP_BIN="$PYTHON_BIN -m pip"
fi
echo "Using Python: $PYTHON_BIN"


# Enable proxy for downloading models/data (read from environment)
export https_proxy="${HTTPS_PROXY:-}"
export http_proxy="${HTTP_PROXY:-}"
export CUDA_HOME=/mnt/petrelfs/share/test-cuda/cuda-12.1
export PATH=/mnt/petrelfs/share/test-cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/test-cuda/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

# Create output directories
CANONICAL_RESULTS_DIR=/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/results
EXPERIMENT_RESULTS_DIR=/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment/results

mkdir -p "$CANONICAL_RESULTS_DIR"
mkdir -p /mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment/checkpoints
mkdir -p /mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment/logs

# Keep both legacy and experiment paths valid via symlink to one canonical results root.
if [ -e "$EXPERIMENT_RESULTS_DIR" ] && [ ! -L "$EXPERIMENT_RESULTS_DIR" ]; then
    rm -rf "$EXPERIMENT_RESULTS_DIR"
fi
ln -sfn "$CANONICAL_RESULTS_DIR" "$EXPERIMENT_RESULTS_DIR"

export NANORESEARCH_RESULTS_DIR="$CANONICAL_RESULTS_DIR"
export NANORESEARCH_LEGACY_RESULTS_DIR="$EXPERIMENT_RESULTS_DIR"

# Install requirements only when a manifest exists
cd /mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/experiment
if [ -f requirements.txt ]; then
    if [ "$PIP_BIN" = "$PYTHON_BIN -m pip" ]; then
        "$PYTHON_BIN" -m pip install -r requirements.txt --quiet 2>/dev/null || true
    else
        "$PIP_BIN" install -r requirements.txt --quiet 2>/dev/null || true
    fi
fi

python() {
    "$PYTHON_BIN" "$@"
}

# Run training
echo "Starting training..."
python nanoresearch_runner.py 2>&1

EXIT_CODE=$?

echo "========================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
