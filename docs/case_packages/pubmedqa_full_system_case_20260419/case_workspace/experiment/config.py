import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


# =========================
# Absolute project paths
# =========================
WORKSPACE_ROOT = Path(
    "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/"
    "router_plan_fullsystem_20260419_10proc_r2/run/"
    "ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/"
    "workspaces/attempt-01"
).resolve()

DATA_DIR = (WORKSPACE_ROOT / "datasets").resolve()
MODELS_DIR = (WORKSPACE_ROOT / "models").resolve()
RESULTS_DIR = (WORKSPACE_ROOT / "results").resolve()
CHECKPOINT_DIR = (WORKSPACE_ROOT / "checkpoints").resolve()
CACHE_DIR = (WORKSPACE_ROOT / "cache").resolve()
LOG_DIR = (WORKSPACE_ROOT / "logs").resolve()

# Dataset absolute paths
PUBMEDQA_ROOT = (DATA_DIR / "pubmedqa").resolve()
PUBMEDQA_DATA_DIR = (PUBMEDQA_ROOT / "data").resolve()
PUBMEDQA_TRAIN_JSON = (PUBMEDQA_DATA_DIR / "ori_pqal.json").resolve()
PUBMEDQA_TEST_GT_JSON = (PUBMEDQA_DATA_DIR / "test_ground_truth.json").resolve()

# Model absolute paths (ONLY available models)
PUBMEDBERT_MODEL_PATH = (MODELS_DIR / "PubMedBERT-base").resolve()
T5_SMALL_MODEL_PATH = (MODELS_DIR / "T5-small").resolve()
BIOBERT_MODEL_PATH = (MODELS_DIR / "BioBERT").resolve()

# Label mapping
LABEL2ID: Dict[str, int] = {"no": 0, "maybe": 1, "yes": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Default training parameters
DEFAULTS = {
    "run_name": "pubmedqa",
    "mode": "cct",  # one of: baseline, cct, cct_ce_only, cct_para_only
    "epochs": 5,
    "batch_size": 16,
    "eval_batch_size": 32,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_length": 512,
    "max_question_length": 96,
    "max_paraphrase_length": 96,
    "max_t5_gen_length": 64,
    "gradient_accumulation_steps": 1,
    "lambda_contrastive": 0.5,
    "lambda_kl": 0.1,
    "dropout": 0.1,
    "seed": 42,
    "num_workers": 4,
    "logging_steps": 20,
    "save_total_limit": 2,
    "fp16": False,
    "bf16": False,
    "dry_run": False,
    "quick_eval": False,
    "quick_train_samples": 64,
    "quick_eval_samples": 64,
}


@dataclass
class ExperimentConfig:
    run_name: str = DEFAULTS["run_name"]
    mode: str = DEFAULTS["mode"]
    epochs: int = DEFAULTS["epochs"]
    batch_size: int = DEFAULTS["batch_size"]
    eval_batch_size: int = DEFAULTS["eval_batch_size"]
    lr: float = DEFAULTS["lr"]
    weight_decay: float = DEFAULTS["weight_decay"]
    warmup_ratio: float = DEFAULTS["warmup_ratio"]
    max_seq_length: int = DEFAULTS["max_seq_length"]
    max_question_length: int = DEFAULTS["max_question_length"]
    max_paraphrase_length: int = DEFAULTS["max_paraphrase_length"]
    max_t5_gen_length: int = DEFAULTS["max_t5_gen_length"]
    gradient_accumulation_steps: int = DEFAULTS["gradient_accumulation_steps"]
    lambda_contrastive: float = DEFAULTS["lambda_contrastive"]
    lambda_kl: float = DEFAULTS["lambda_kl"]
    dropout: float = DEFAULTS["dropout"]
    seed: int = DEFAULTS["seed"]
    num_workers: int = DEFAULTS["num_workers"]
    logging_steps: int = DEFAULTS["logging_steps"]
    save_total_limit: int = DEFAULTS["save_total_limit"]
    fp16: bool = DEFAULTS["fp16"]
    bf16: bool = DEFAULTS["bf16"]
    dry_run: bool = DEFAULTS["dry_run"]
    quick_eval: bool = DEFAULTS["quick_eval"]
    quick_train_samples: int = DEFAULTS["quick_train_samples"]
    quick_eval_samples: int = DEFAULTS["quick_eval_samples"]

    # Paths
    workspace_root: str = str(WORKSPACE_ROOT)
    data_dir: str = str(DATA_DIR)
    models_dir: str = str(MODELS_DIR)
    results_dir: str = str(RESULTS_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    cache_dir: str = str(CACHE_DIR)
    log_dir: str = str(LOG_DIR)

    # Dataset files
    pubmedqa_root: str = str(PUBMEDQA_ROOT)
    pubmedqa_data_dir: str = str(PUBMEDQA_DATA_DIR)
    train_json_path: str = str(PUBMEDQA_TRAIN_JSON)
    test_ground_truth_path: str = str(PUBMEDQA_TEST_GT_JSON)

    # Models
    pubmedbert_model_path: str = str(PUBMEDBERT_MODEL_PATH)
    t5_small_model_path: str = str(T5_SMALL_MODEL_PATH)
    biobert_model_path: str = str(BIOBERT_MODEL_PATH)

    # Labels
    num_labels: int = 3
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        if self.label2id is None:
            self.label2id = dict(LABEL2ID)
        if self.id2label is None:
            self.id2label = dict(ID2LABEL)

    def validate(self, strict: bool = True) -> None:
        allowed_modes = {"baseline", "cct", "cct_ce_only", "cct_para_only"}
        if self.mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {sorted(allowed_modes)}")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch_size and eval_batch_size must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.quick_train_samples <= 0 or self.quick_eval_samples <= 0:
            raise ValueError("quick_*_samples must be > 0")
        if self.num_labels != 3:
            raise ValueError("num_labels must be 3 for PubMedQA yes/no/maybe classification")
        if self.label2id is None or self.id2label is None:
            raise ValueError("label mappings must not be None")
        if strict:
            required_paths = [
                Path(self.workspace_root),
                Path(self.data_dir),
                Path(self.models_dir),
                Path(self.train_json_path),
                Path(self.test_ground_truth_path),
                Path(self.pubmedbert_model_path),
                Path(self.t5_small_model_path),
            ]
            missing = [str(p) for p in required_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing required files/directories: {missing}")

    def ensure_dirs(self) -> None:
        for p in [self.results_dir, self.checkpoint_dir, self.cache_dir, self.log_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configuration parser for PubMedQA Contrastive Consistency Tuning experiments."
    )
    parser.add_argument("--run_name", type=str, default=DEFAULTS["run_name"])
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULTS["mode"],
        choices=["baseline", "cct", "cct_ce_only", "cct_para_only"],
        help="Training mode/ablation setting.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULTS["eval_batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--max_question_length", type=int, default=DEFAULTS["max_question_length"])
    parser.add_argument("--max_paraphrase_length", type=int, default=DEFAULTS["max_paraphrase_length"])
    parser.add_argument("--max_t5_gen_length", type=int, default=DEFAULTS["max_t5_gen_length"])
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULTS["gradient_accumulation_steps"],
    )
    parser.add_argument("--lambda_contrastive", type=float, default=DEFAULTS["lambda_contrastive"])
    parser.add_argument("--lambda_kl", type=float, default=DEFAULTS["lambda_kl"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULTS["logging_steps"])
    parser.add_argument("--save_total_limit", type=int, default=DEFAULTS["save_total_limit"])
    parser.add_argument("--fp16", action="store_true", default=DEFAULTS["fp16"])
    parser.add_argument("--bf16", action="store_true", default=DEFAULTS["bf16"])
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=DEFAULTS["dry_run"])
    parser.add_argument("--quick-eval", dest="quick_eval", action="store_true", default=DEFAULTS["quick_eval"])
    parser.add_argument("--quick_train_samples", type=int, default=DEFAULTS["quick_train_samples"])
    parser.add_argument("--quick_eval_samples", type=int, default=DEFAULTS["quick_eval_samples"])

    # Absolute paths
    parser.add_argument("--workspace_root", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--log_dir", type=str, default=str(LOG_DIR))

    # Dataset paths
    parser.add_argument("--pubmedqa_root", type=str, default=str(PUBMEDQA_ROOT))
    parser.add_argument("--pubmedqa_data_dir", type=str, default=str(PUBMEDQA_DATA_DIR))
    parser.add_argument("--train_json_path", type=str, default=str(PUBMEDQA_TRAIN_JSON))
    parser.add_argument("--test_ground_truth_path", type=str, default=str(PUBMEDQA_TEST_GT_JSON))

    # Model paths
    parser.add_argument("--pubmedbert_model_path", type=str, default=str(PUBMEDBERT_MODEL_PATH))
    parser.add_argument("--t5_small_model_path", type=str, default=str(T5_SMALL_MODEL_PATH))
    parser.add_argument("--biobert_model_path", type=str, default=str(BIOBERT_MODEL_PATH))
    return parser


def parse_args_to_config(args: Optional[list] = None, strict_validate: bool = False) -> ExperimentConfig:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)
    cfg = ExperimentConfig(**vars(parsed))
    cfg.ensure_dirs()
    cfg.validate(strict=strict_validate)
    return cfg


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("config")
    cfg = parse_args_to_config(strict_validate=False)
    logger.info("Loaded configuration for run_name=%s mode=%s", cfg.run_name, cfg.mode)
    config_dump_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_config.json")
    cfg.save_json(config_dump_path)
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))