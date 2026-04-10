#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/dhwfile/raise/user/xujinhang/data/modelscope/models/Qwen/Qwen3-8B}"
CONDA_BASE="${CONDA_BASE:-/mnt/petrelfs/xujinhang/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/mnt/petrelfs/xujinhang/nanoresearch/tmp/router_sdpo_offpolicy_runs}"
RUN_NAME="${RUN_NAME:-router_sdpo_offpolicy_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT_BASE}/${RUN_NAME}"
MANIFEST_DIR="${RUN_DIR}/manifest"
GATE_DIR="${RUN_DIR}/gate"
TRAIN_DIR="${RUN_DIR}/train"
LOG_DIR="${RUN_DIR}/logs"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SETUP_ENV="${SETUP_ENV:-0}"

INPUT_DIRS=(
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_seeded_shard1of4_batch9_r2"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_seeded_shard2of4_batch9_r2"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_seeded_shard3of4_batch6_r2"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_seeded_shard4of4_batch6_r2"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_fill600_shard1"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_fill600_shard2"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_fill600_shard3"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_fill600_shard4"
  "/mnt/petrelfs/xujinhang/nanoresearch/tmp/live_router_multiturn_seeded_qwen_20260409_fill600_task3fix"
)

mkdir -p "${MANIFEST_DIR}" "${GATE_DIR}" "${TRAIN_DIR}" "${LOG_DIR}"

# shellcheck disable=SC1091
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

if [[ "${SETUP_ENV}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install \
    "transformers==4.51.3" \
    "accelerate==1.6.0" \
    "datasets==3.5.0" \
    "peft==0.15.1" \
    "trl==0.16.1" \
    "bitsandbytes==0.42.0"
fi

BNB_DIR="$("${PYTHON_BIN}" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
spec = find_spec("bitsandbytes")
print(Path(spec.origin).resolve().parent if spec and spec.origin else "")
PY
)"
if [[ -n "${BNB_DIR}" ]]; then
  if [[ -e "${BNB_DIR}/libbitsandbytes_cuda123.so" && ! -e "${BNB_DIR}/libbitsandbytes_cuda124.so" ]]; then
    ln -sf "${BNB_DIR}/libbitsandbytes_cuda123.so" "${BNB_DIR}/libbitsandbytes_cuda124.so"
  fi
  if [[ -e "${BNB_DIR}/libbitsandbytes_cuda123_nocublaslt.so" && ! -e "${BNB_DIR}/libbitsandbytes_cuda124_nocublaslt.so" ]]; then
    ln -sf "${BNB_DIR}/libbitsandbytes_cuda123_nocublaslt.so" "${BNB_DIR}/libbitsandbytes_cuda124_nocublaslt.so"
  fi
fi

EXPORT_ARGS=()
for input_dir in "${INPUT_DIRS[@]}"; do
  EXPORT_ARGS+=(--input-dir "${input_dir}")
done

echo "[SDPO] Exporting clean off-policy manifest into ${MANIFEST_DIR}"
"${PYTHON_BIN}" "${REPO_ROOT}/tools/export_router_sdpo_offpolicy.py" \
  "${EXPORT_ARGS[@]}" \
  --tokenizer-path "${MODEL_PATH}" \
  --output "${MANIFEST_DIR}/train_manifest.jsonl" \
  --stats-output "${MANIFEST_DIR}/export_stats.json" \
  --drop-report-output "${MANIFEST_DIR}/drop_report.json" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  | tee "${LOG_DIR}/export.log"

echo "[SDPO] Running 8-GPU launch gate"
torchrun --nproc_per_node=8 "${REPO_ROOT}/tools/train_router_sdpo_offpolicy.py" \
  --model-path "${MODEL_PATH}" \
  --manifest "${MANIFEST_DIR}/train_manifest.jsonl" \
  --output-dir "${GATE_DIR}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --num-epochs 1 \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --gate-only \
  --max-optimizer-steps 1 \
  | tee "${LOG_DIR}/gate.log"

echo "[SDPO] Launching full 2-epoch training"
torchrun --nproc_per_node=8 "${REPO_ROOT}/tools/train_router_sdpo_offpolicy.py" \
  --model-path "${MODEL_PATH}" \
  --manifest "${MANIFEST_DIR}/train_manifest.jsonl" \
  --output-dir "${TRAIN_DIR}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --num-epochs "${NUM_EPOCHS}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  | tee "${LOG_DIR}/train.log"

echo "[SDPO] Completed. Outputs in ${RUN_DIR}"
