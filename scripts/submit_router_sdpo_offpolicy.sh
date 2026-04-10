#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARTITION="${PARTITION:-belt_road}"
JOB_NAME="${JOB_NAME:-router-sdpo-offpolicy}"
RUN_NAME="${RUN_NAME:-router_sdpo_offpolicy_$(date +%Y%m%d_%H%M%S)}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
SETUP_ENV="${SETUP_ENV:-1}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/mnt/petrelfs/xujinhang/nanoresearch/tmp/router_sdpo_offpolicy_runs}"
SBATCH_LOG_DIR="${SBATCH_LOG_DIR:-${OUTPUT_ROOT_BASE}/slurm_logs}"

mkdir -p "${SBATCH_LOG_DIR}"

echo "[SDPO] Submitting exact off-policy router training to Slurm"
echo "[SDPO] partition=${PARTITION} run_name=${RUN_NAME} setup_env=${SETUP_ENV}"

sbatch \
  --job-name "${JOB_NAME}" \
  --partition "${PARTITION}" \
  --nodes 1 \
  --ntasks 1 \
  --gres gpu:8 \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --time "${TIME_LIMIT}" \
  --chdir "${REPO_ROOT}" \
  --output "${SBATCH_LOG_DIR}/${RUN_NAME}.%j.out" \
  --error "${SBATCH_LOG_DIR}/${RUN_NAME}.%j.err" \
  --export "ALL,RUN_NAME=${RUN_NAME},SETUP_ENV=${SETUP_ENV},OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE}" \
  --wrap "bash ${REPO_ROOT}/scripts/run_router_sdpo_offpolicy.sh"
