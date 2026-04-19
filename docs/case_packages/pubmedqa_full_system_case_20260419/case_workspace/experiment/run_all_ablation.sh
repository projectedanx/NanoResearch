#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
TRAIN_SCRIPT="train.py"

DATA_DIR="/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/datasets"
MODEL_PATH="/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models/PubMedBERT-base"
RESULTS_ROOT="/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/results"

RUN_EPOCHS=5
RUN_BATCH_SIZE=16
RUN_LR="2e-5"
RUN_SEED=42

ENABLE_QUICK_EVAL=0
ENABLE_DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Runs full CCT and two ablations with matched hyperparameters.

Options:
  --python-bin PATH       Python executable (default: ${PYTHON_BIN})
  --train-script PATH     Training script path (default: ${TRAIN_SCRIPT})
  --data-dir PATH         Dataset directory (default: ${DATA_DIR})
  --model-path PATH       PubMedBERT path (default: ${MODEL_PATH})
  --results-root PATH     Root directory for ablation outputs (default: ${RESULTS_ROOT})
  --epochs INT            Epochs per run (default: ${RUN_EPOCHS})
  --batch-size INT        Batch size (default: ${RUN_BATCH_SIZE})
  --lr FLOAT              Learning rate (default: ${RUN_LR})
  --seed INT              Random seed (default: ${RUN_SEED})
  --quick-eval            Pass --quick-eval to train.py
  --dry-run               Pass --dry-run to train.py
  -h, --help              Show this help

Examples:
  $0
  $0 --quick-eval
  $0 --python-bin /usr/bin/python3 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --epochs)
      RUN_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      RUN_BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      RUN_LR="$2"
      shift 2
      ;;
    --seed)
      RUN_SEED="$2"
      shift 2
      ;;
    --quick-eval)
      ENABLE_QUICK_EVAL=1
      shift
      ;;
    --dry-run)
      ENABLE_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] train.py not found at: ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Data directory not found: ${DATA_DIR}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model path not found: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "${RESULTS_ROOT}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
EXP_DIR="${RESULTS_ROOT}/ablation_cct_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

SUMMARY_CSV="${EXP_DIR}/ablation_summary.csv"
SUMMARY_JSON="${EXP_DIR}/ablation_summary.json"
LOG_FILE="${EXP_DIR}/run.log"

echo "run_name,lambda_contrastive,status,final_val_accuracy,best_val_accuracy,metrics_file" > "${SUMMARY_CSV}"

declare -a EXTRA_FLAGS=()
if [[ ${ENABLE_QUICK_EVAL} -eq 1 ]]; then
  EXTRA_FLAGS+=("--quick-eval")
fi
if [[ ${ENABLE_DRY_RUN} -eq 1 ]]; then
  EXTRA_FLAGS+=("--dry-run")
fi

run_one() {
  local run_name="$1"
  local lambda_value="$2"

  local run_dir="${EXP_DIR}/${run_name}"
  mkdir -p "${run_dir}"

  echo "[$(date '+%F %T')] Starting run: ${run_name} (lambda=${lambda_value})" | tee -a "${LOG_FILE}"

  rm -rf results
  mkdir -p results

  set +e
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --run_name "${run_name}" \
    --mode cct \
    --epochs "${RUN_EPOCHS}" \
    --batch_size "${RUN_BATCH_SIZE}" \
    --lr "${RUN_LR}" \
    --lambda_contrastive "${lambda_value}" \
    --seed "${RUN_SEED}" \
    --fp16 \
    --data_dir "${DATA_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
  local rc=${PIPESTATUS[0]}
  set -e

  local status="success"
  if [[ ${rc} -ne 0 ]]; then
    status="failed"
    echo "[$(date '+%F %T')] Run failed: ${run_name} (exit=${rc})" | tee -a "${LOG_FILE}"
  else
    echo "[$(date '+%F %T')] Run completed: ${run_name}" | tee -a "${LOG_FILE}"
  fi

  local metrics_src="results/metrics.json"
  local metrics_dst="${run_dir}/metrics.json"
  local final_val_acc=""
  local best_val_acc=""

  if [[ -f "${metrics_src}" ]]; then
    cp "${metrics_src}" "${metrics_dst}"
    read -r final_val_acc best_val_acc < <(
      "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

p = Path("${metrics_dst}")
final_acc = ""
best_acc = ""

if p.exists():
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        if isinstance(data.get("history"), list) and data["history"]:
            vals = [x.get("val_accuracy") for x in data["history"] if isinstance(x, dict)]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                final_acc = vals[-1]
                best_acc = max(vals)
        if final_acc == "" and isinstance(data.get("val_accuracy"), (int, float)):
            final_acc = data["val_accuracy"]
            best_acc = data["val_accuracy"]
        if best_acc == "" and isinstance(data.get("best_val_accuracy"), (int, float)):
            best_acc = data["best_val_accuracy"]
        if final_acc == "" and isinstance(data.get("final_val_accuracy"), (int, float)):
            final_acc = data["final_val_accuracy"]
    elif isinstance(data, list) and data:
        vals = [x.get("val_accuracy") for x in data if isinstance(x, dict)]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            final_acc = vals[-1]
            best_acc = max(vals)

print(final_acc, best_acc)
PY
    )
  else
    echo "[$(date '+%F %T')] WARNING: metrics.json not found for ${run_name}" | tee -a "${LOG_FILE}"
    status="failed"
  fi

  if compgen -G "results/*" > /dev/null; then
    cp -r results/* "${run_dir}/" || true
  fi

  echo "${run_name},${lambda_value},${status},${final_val_acc},${best_val_acc},${metrics_dst}" >> "${SUMMARY_CSV}"

  if [[ "${status}" != "success" ]]; then
    return 1
  fi
  return 0
}

overall_rc=0

run_one "pubmedqa_cct_full" "0.5" || overall_rc=1
run_one "pubmedqa_ablation_no_contrastive" "0.0" || overall_rc=1
run_one "pubmedqa_ablation_lambda_0p2" "0.2" || overall_rc=1

"${PYTHON_BIN}" - <<PY
import csv
import json
from pathlib import Path

csv_path = Path("${SUMMARY_CSV}")
json_path = Path("${SUMMARY_JSON}")

rows = []
with csv_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

summary = {
    "experiment_dir": str(Path("${EXP_DIR}")),
    "runs": rows,
    "num_runs": len(rows),
    "num_success": sum(1 for r in rows if r.get("status") == "success"),
    "num_failed": sum(1 for r in rows if r.get("status") != "success"),
}
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[INFO] Wrote summary JSON: {json_path}")
PY

echo "[$(date '+%F %T')] All ablation runs finished. Summary: ${SUMMARY_CSV}" | tee -a "${LOG_FILE}"
echo "[$(date '+%F %T')] JSON summary: ${SUMMARY_JSON}" | tee -a "${LOG_FILE}"

if [[ ${overall_rc} -ne 0 ]]; then
  echo "[ERROR] One or more ablation runs failed."
  exit 1
fi

exit 0