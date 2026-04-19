import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import tarfile
import zipfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from typing import Dict, List, Tuple, Any, Optional

# Suppress known non-fatal warnings *before importing torch*, because
# torch can emit warnings at import time.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0\.",
    category=FutureWarning,
    module=r"huggingface_hub\.file_download",
)

# Keep Hugging Face/tokenizers auxiliary output off to avoid non-error STDERR noise in strict runners.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
    hf_disable_progress_bars()
except Exception:
    pass


DEFAULT_DATA_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/datasets"
DEFAULT_MODELS_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models"
DEFAULT_PUBMEDBERT_PATH = os.path.join(DEFAULT_MODELS_DIR, "PubMedBERT-base")
DEFAULT_T5_PATH = os.path.join(DEFAULT_MODELS_DIR, "T5-small")
DEFAULT_ORI_PATH = os.path.join(DEFAULT_DATA_DIR, "pubmedqa", "data", "ori_pqal.json")
DEFAULT_TEST_GT_PATH = os.path.join(DEFAULT_DATA_DIR, "pubmedqa", "data", "test_ground_truth.json")

LABEL2ID = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class Example:
    qid: str
    question: str
    context: str
    label: int


class PubMedQADataset(Dataset):
    def __init__(self, examples: List[Example], paraphrases: Dict[str, str]):
        self.examples = examples
        self.paraphrases = paraphrases

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        para = self.paraphrases.get(ex.qid, ex.question)
        return {
            "qid": ex.qid,
            "question": ex.question,
            "paraphrase": para,
            "context": ex.context,
            "label": ex.label,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "qid": [b["qid"] for b in batch],
        "question": [b["question"] for b in batch],
        "paraphrase": [b["paraphrase"] for b in batch],
        "context": [b["context"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def setup_logging(results_dir: str, run_name: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f"{run_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_decompress_archives(data_dir: str) -> None:
    for root, _, files in os.walk(data_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                if fn.endswith(".zip"):
                    target_dir = fp[:-4]
                    if not os.path.isdir(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                        with zipfile.ZipFile(fp, "r") as zf:
                            zf.extractall(target_dir)
                elif fn.endswith(".tar.gz") or fn.endswith(".tgz") or fn.endswith(".tar"):
                    target_dir = fp.rsplit(".", 1)[0]
                    if not os.path.isdir(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                        with tarfile.open(fp, "r:*") as tf:
                            tf.extractall(target_dir)
            except Exception as e:
                logging.warning(f"Failed to decompress {fp}: {e}")


def normalize_label(label: str) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in LABEL2ID:
        return s
    if s in {"true", "y"}:
        return "yes"
    if s in {"false", "n"}:
        return "no"
    return None


def extract_question(item: Dict[str, Any]) -> str:
    for k in ["QUESTION", "question", "Question", "query"]:
        if k in item and item[k]:
            return str(item[k]).strip()
    return ""


def extract_context(item: Dict[str, Any]) -> str:
    for k in ["CONTEXTS", "contexts", "CONTEXT", "context", "abstract"]:
        if k in item and item[k] is not None:
            c = item[k]
            if isinstance(c, dict):
                if "contexts" in c and isinstance(c["contexts"], list):
                    return " ".join([str(x) for x in c["contexts"]]).strip()
                return " ".join([str(v) for v in c.values()]).strip()
            if isinstance(c, list):
                return " ".join([str(x) for x in c]).strip()
            return str(c).strip()
    long_answer = item.get("LONG_ANSWER", item.get("long_answer", ""))
    if long_answer:
        return str(long_answer).strip()
    return ""


def extract_label(item: Dict[str, Any]) -> Optional[str]:
    for k in ["final_decision", "FINAL_DECISION", "label", "answer", "decision"]:
        if k in item:
            v = normalize_label(item[k])
            if v is not None:
                return v
    return None


def load_local_pubmedqa(ori_path: str, test_gt_path: str, seed: int, val_ratio: float) -> Tuple[List[Example], List[Example], List[Example]]:
    with open(ori_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples_all: List[Example] = []
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = [(str(i), x) for i, x in enumerate(raw)]
    else:
        raise ValueError(f"Unsupported format in {ori_path}")

    for qid, item in iterable:
        if not isinstance(item, dict):
            continue
        q = extract_question(item)
        c = extract_context(item)
        l = extract_label(item)
        if not q or not c or l is None:
            continue
        examples_all.append(Example(qid=str(qid), question=q, context=c, label=LABEL2ID[l]))

    if len(examples_all) == 0:
        raise ValueError(f"No valid examples parsed from {ori_path}")

    test_ids_to_label: Dict[str, int] = {}
    if os.path.isfile(test_gt_path):
        with open(test_gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        if isinstance(gt, dict):
            for k, v in gt.items():
                lv = normalize_label(v)
                if lv is not None:
                    test_ids_to_label[str(k)] = LABEL2ID[lv]

    examples_by_id = {e.qid: e for e in examples_all}
    test_examples: List[Example] = []
    if test_ids_to_label:
        for qid, lid in test_ids_to_label.items():
            if qid in examples_by_id:
                e = examples_by_id[qid]
                test_examples.append(Example(qid=e.qid, question=e.question, context=e.context, label=lid))

    test_id_set = {e.qid for e in test_examples}
    remaining = [e for e in examples_all if e.qid not in test_id_set]

    rng = random.Random(seed)
    rng.shuffle(remaining)

    if len(remaining) < 2:
        raise ValueError("Not enough remaining examples to create train/validation splits.")

    val_size = max(1, int(len(remaining) * val_ratio))
    if val_size >= len(remaining):
        val_size = max(1, len(remaining) - 1)

    val_examples = remaining[:val_size]
    train_examples = remaining[val_size:]

    if len(test_examples) == 0:
        # Fallback: split from remaining for a local test set
        test_size = max(1, min(val_size, len(train_examples) // 2))
        test_examples = train_examples[:test_size]
        train_examples = train_examples[test_size:]

    if len(train_examples) == 0 or len(val_examples) == 0:
        raise ValueError("Empty train or validation split after processing.")

    return train_examples, val_examples, test_examples


def load_hf_pubmedqa(data_dir: str, seed: int, val_ratio: float) -> Tuple[List[Example], List[Example], List[Example]]:
    if load_dataset is None:
        raise RuntimeError("datasets package is not available for HuggingFace fallback.")

    ds = None
    tried = []
    for cfg in [("pubmed_qa", "pqa_labeled"), ("pubmed_qa", None)]:
        try:
            if cfg[1] is None:
                ds = load_dataset(cfg[0], cache_dir=data_dir)
            else:
                ds = load_dataset(cfg[0], cfg[1], cache_dir=data_dir)
            break
        except Exception as e:
            tried.append((cfg, str(e)))

    if ds is None:
        raise RuntimeError(f"Unable to load PubMedQA from HF. Tried: {tried}")

    if "train" not in ds:
        raise RuntimeError("HF dataset has no train split.")

    def hf_to_examples(split_data, start_idx: int = 0) -> List[Example]:
        out: List[Example] = []
        for i, row in enumerate(split_data):
            q = str(row.get("question", "")).strip()
            context_obj = row.get("contexts", row.get("context", ""))
            if isinstance(context_obj, dict) and "contexts" in context_obj:
                c = " ".join([str(x) for x in context_obj["contexts"]]).strip()
            elif isinstance(context_obj, list):
                c = " ".join([str(x) for x in context_obj]).strip()
            else:
                c = str(context_obj).strip()
            ans = normalize_label(row.get("final_decision", row.get("answer", None)))
            if q and c and ans in LABEL2ID:
                qid = str(row.get("pubid", row.get("id", start_idx + i)))
                out.append(Example(qid=qid, question=q, context=c, label=LABEL2ID[ans]))
        return out

    train_raw = hf_to_examples(ds["train"], 0)
    if len(train_raw) == 0:
        raise RuntimeError("HF train split has no parseable examples.")

    rng = random.Random(seed)
    rng.shuffle(train_raw)

    val_size = max(1, int(len(train_raw) * val_ratio))
    test_size = max(1, val_size)
    if val_size + test_size >= len(train_raw):
        test_size = max(1, len(train_raw) // 4)
        val_size = max(1, len(train_raw) // 4)

    val_examples = train_raw[:val_size]
    test_examples = train_raw[val_size:val_size + test_size]
    train_examples = train_raw[val_size + test_size:]

    if len(train_examples) == 0:
        raise RuntimeError("HF split resulted in empty train set.")

    return train_examples, val_examples, test_examples


def load_pubmedqa_data(args) -> Tuple[List[Example], List[Example], List[Example]]:
    maybe_decompress_archives(args.data_dir)

    ori_exists = os.path.isfile(args.ori_path)
    gt_exists = os.path.isfile(args.test_gt_path)
    if ori_exists:
        logging.info(f"Loading local PubMedQA from {args.ori_path}")
        return load_local_pubmedqa(args.ori_path, args.test_gt_path if gt_exists else "", args.seed, args.val_ratio)

    logging.warning("Local PubMedQA not found; trying HuggingFace download fallback.")
    return load_hf_pubmedqa(args.data_dir, args.seed, args.val_ratio)


def subsample_examples(examples: List[Example], n: Optional[int], seed: int) -> List[Example]:
    if n is None or n <= 0 or n >= len(examples):
        return examples
    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    return [examples[i] for i in idxs]


def generate_or_load_paraphrases(
    train_examples: List[Example],
    args,
    device: torch.device,
) -> Dict[str, str]:
    cache_file = args.paraphrase_cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    cached: Dict[str, str] = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load paraphrase cache; will regenerate. Error: {e}")
            cached = {}

    needed = [ex for ex in train_examples if ex.qid not in cached]
    if len(needed) == 0:
        return cached

    if args.dry_run:
        for ex in needed:
            cached[ex.qid] = ex.question
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
        return cached

    logging.info(f"Generating paraphrases for {len(needed)} training examples using T5...")
    try:
        t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path, local_files_only=True)
        t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path, local_files_only=True).to(device)
    except Exception:
        t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path)
        t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path).to(device)

    t5_model.eval()
    batch_size = args.paraphrase_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(needed), batch_size), desc="Paraphrasing", file=sys.stdout):
            batch = needed[i:i + batch_size]
            prompts = [f"paraphrase: {ex.question} </s>" for ex in batch]
            enc = t5_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            outs = t5_model.generate(
                **enc,
                max_length=64,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
            texts = t5_tokenizer.batch_decode(outs, skip_special_tokens=True)
            for ex, txt in zip(batch, texts):
                txt = txt.strip() if txt and txt.strip() else ex.question
                cached[ex.qid] = txt

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    del t5_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cached


def build_model_and_tokenizer(args, device: torch.device):
    def _has_hf_weights(path: str) -> bool:
        return any(
            os.path.isfile(os.path.join(path, name))
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )

    attempted = []

    candidate_local_paths = [args.model_path]
    biobert_fallback = os.path.join(os.path.dirname(args.model_path), "BioBERT")
    if biobert_fallback not in candidate_local_paths:
        candidate_local_paths.append(biobert_fallback)

    for local_path in candidate_local_paths:
        if not os.path.isdir(local_path):
            continue
        if not _has_hf_weights(local_path):
            logging.warning(
                "Skipping local model directory without checkpoint weights: %s",
                local_path,
            )
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                local_path,
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
            model.to(device)
            logging.info("Loaded model/tokenizer from local path: %s", local_path)
            return model, tokenizer
        except Exception as e:
            attempted.append(f"local:{local_path} -> {e}")

    for hf_id in [
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dmis-lab/biobert-base-cased-v1.1",
    ]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_id,
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
                local_files_only=False,
            )
            model.to(device)
            logging.info("Loaded model/tokenizer from HF id: %s", hf_id)
            return model, tokenizer
        except Exception as e:
            attempted.append(f"hf:{hf_id} -> {e}")

    raise RuntimeError(
        "Failed to load any usable encoder checkpoint. Attempts: " + " | ".join(attempted)
    )


def compute_per_class_accuracy(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metrics = {}
    for cls_name, cls_id in LABEL2ID.items():
        mask = labels == cls_id
        if mask.sum() == 0:
            metrics[f"{cls_name}_accuracy"] = 0.0
        else:
            metrics[f"{cls_name}_accuracy"] = float((preds[mask] == labels[mask]).mean())
    return metrics


def evaluate_model(model, tokenizer, dataloader, device, max_length: int, quick_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False, file=sys.stdout)):
            enc = tokenizer(
                batch["context"],
                batch["question"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            labels = batch["label"].to(device)
            outputs = model(**enc, labels=labels, output_hidden_states=False)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            total_loss += float(loss.item())
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            if quick_batches is not None and (bi + 1) >= quick_batches:
                break

    preds_np = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    labels_np = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)

    if len(labels_np) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "yes_accuracy": 0.0, "no_accuracy": 0.0, "maybe_accuracy": 0.0}

    acc = float((preds_np == labels_np).mean())
    class_metrics = compute_per_class_accuracy(preds_np, labels_np)
    return {
        "loss": total_loss / max(1, len(all_labels)),
        "accuracy": acc,
        **class_metrics,
    }


def save_metrics(metrics_payload: Dict[str, Any], json_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    history = metrics_payload.get("history", [])
    if len(history) == 0:
        return

    fieldnames = sorted({k for row in history for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def mirror_results_if_needed(primary_results_dir: str) -> None:
    legacy_dir = os.environ.get("NANORESEARCH_LEGACY_RESULTS_DIR", "").strip()
    if not legacy_dir:
        return

    primary_abs = os.path.abspath(primary_results_dir)
    legacy_abs = os.path.abspath(legacy_dir)
    if primary_abs == legacy_abs:
        return

    try:
        os.makedirs(legacy_abs, exist_ok=True)
        for name in os.listdir(primary_abs):
            src = os.path.join(primary_abs, name)
            dst = os.path.join(legacy_abs, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)
    except Exception as e:
        logging.warning("Failed to mirror results to legacy path %s: %s", legacy_abs, str(e))


def get_mode_weights(args) -> Tuple[float, float]:
    mode = args.mode.lower()
    if mode == "baseline":
        return 0.0, 0.0
    if mode == "cct":
        return args.lambda_consistency, args.lambda_contrastive
    if mode == "cct_consistency":
        return args.lambda_consistency, 0.0
    if mode == "cct_contrastive":
        return 0.0, args.lambda_contrastive
    raise ValueError(f"Unknown mode: {args.mode}")


def train(args):
    set_seed(args.seed)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id:
        logging.info(f"Detected SLURM_JOB_ID={slurm_job_id}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    train_examples, val_examples, test_examples = load_pubmedqa_data(args)

    if args.quick_eval:
        logging.info("Quick-eval enabled: forcing tiny subset and short training.")
        args.epochs = min(args.epochs, 1)
        args.max_train_samples = 64 if args.max_train_samples is None else min(args.max_train_samples, 64)
        args.max_val_samples = 32 if args.max_val_samples is None else min(args.max_val_samples, 32)
        args.max_test_samples = 32 if args.max_test_samples is None else min(args.max_test_samples, 32)

    train_examples = subsample_examples(train_examples, args.max_train_samples, args.seed)
    val_examples = subsample_examples(val_examples, args.max_val_samples, args.seed + 1)
    test_examples = subsample_examples(test_examples, args.max_test_samples, args.seed + 2)

    logging.info(f"Split sizes -> train: {len(train_examples)}, val: {len(val_examples)}, test: {len(test_examples)}")

    paraphrases = generate_or_load_paraphrases(train_examples, args, device)
    # For val/test we do identity paraphrase (not used in eval)
    val_paraphrases = {e.qid: e.question for e in val_examples}
    test_paraphrases = {e.qid: e.question for e in test_examples}

    train_ds = PubMedQADataset(train_examples, paraphrases)
    val_ds = PubMedQADataset(val_examples, val_paraphrases)
    test_ds = PubMedQADataset(test_examples, test_paraphrases)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model, tokenizer = build_model_and_tokenizer(args, device)

    if args.eval_only:
        val_metrics = evaluate_model(model, tokenizer, val_loader, device, args.max_length, quick_batches=2 if args.dry_run else None)
        test_metrics = evaluate_model(model, tokenizer, test_loader, device, args.max_length, quick_batches=2 if args.dry_run else None)
        payload = {
            "run_name": args.run_name,
            "mode": args.mode,
            "args": vars(args),
            "history": [],
            "eval_only": True,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        save_metrics(payload, args.metrics_json, args.metrics_csv)
        mirror_results_if_needed(args.results_dir)
        logging.info(f"Eval-only complete. Val acc={val_metrics['accuracy']:.4f} Test acc={test_metrics['accuracy']:.4f}")
        return

    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = int(np.ceil(len(train_loader) / args.grad_accum_steps)) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    use_amp = bool(args.fp16 and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def autocast_context():
            return torch.amp.autocast("cuda", enabled=use_amp)
    else:
        def autocast_context():
            return torch.cuda.amp.autocast(enabled=use_amp)

    lambda_consistency, lambda_contrastive = get_mode_weights(args)
    best_val_acc = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        ce_loss_sum = 0.0
        consistency_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stdout)
        for step, batch in enumerate(pbar, start=1):
            labels = batch["label"].to(device)

            contexts = batch["context"]
            questions = batch["question"]
            paraphrases_batch = batch["paraphrase"]

            neg_questions = questions[1:] + questions[:1] if len(questions) > 1 else questions

            with autocast_context():
                enc_orig = tokenizer(
                    contexts,
                    questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                out_orig = model(**enc_orig, labels=labels, output_hidden_states=True)
                ce_loss = out_orig.loss
                logits_orig = out_orig.logits
                cls_orig = out_orig.hidden_states[-1][:, 0, :]

                total_loss = ce_loss
                consistency_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)

                if lambda_consistency > 0.0 or lambda_contrastive > 0.0:
                    enc_para = tokenizer(
                        contexts,
                        paraphrases_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                    ).to(device)
                    out_para = model(**enc_para, output_hidden_states=True)
                    logits_para = out_para.logits
                    cls_para = out_para.hidden_states[-1][:, 0, :]

                    if lambda_consistency > 0.0:
                        t = args.consistency_temperature
                        kl1 = F.kl_div(
                            F.log_softmax(logits_orig / t, dim=-1),
                            F.softmax(logits_para / t, dim=-1),
                            reduction="batchmean",
                        )
                        kl2 = F.kl_div(
                            F.log_softmax(logits_para / t, dim=-1),
                            F.softmax(logits_orig / t, dim=-1),
                            reduction="batchmean",
                        )
                        consistency_loss = 0.5 * (kl1 + kl2) * (t ** 2)
                        total_loss = total_loss + lambda_consistency * consistency_loss

                    if lambda_contrastive > 0.0:
                        enc_neg = tokenizer(
                            contexts,
                            neg_questions,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=args.max_length,
                        ).to(device)
                        out_neg = model(**enc_neg, output_hidden_states=True)
                        cls_neg = out_neg.hidden_states[-1][:, 0, :]

                        cos_pos = F.cosine_similarity(cls_orig, cls_para, dim=-1)
                        cos_neg = F.cosine_similarity(cls_orig, cls_neg, dim=-1)
                        contrastive_loss = F.relu(args.contrastive_margin - cos_pos + cos_neg).mean()
                        total_loss = total_loss + lambda_contrastive * contrastive_loss

                loss = total_loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(total_loss.item())
            ce_loss_sum += float(ce_loss.item())
            consistency_loss_sum += float(consistency_loss.item())
            contrastive_loss_sum += float(contrastive_loss.item())
            step_count += 1

            pbar.set_postfix(
                loss=f"{epoch_loss / max(1, step_count):.4f}",
                ce=f"{ce_loss_sum / max(1, step_count):.4f}",
                cons=f"{consistency_loss_sum / max(1, step_count):.4f}",
                ctr=f"{contrastive_loss_sum / max(1, step_count):.4f}",
            )

            if args.dry_run and step >= 1:
                break

        val_metrics = evaluate_model(
            model,
            tokenizer,
            val_loader,
            device,
            args.max_length,
            quick_batches=2 if args.dry_run else None
        )

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, step_count),
            "train_ce_loss": ce_loss_sum / max(1, step_count),
            "train_consistency_loss": consistency_loss_sum / max(1, step_count),
            "train_contrastive_loss": contrastive_loss_sum / max(1, step_count),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_yes_accuracy": val_metrics["yes_accuracy"],
            "val_no_accuracy": val_metrics["no_accuracy"],
            "val_maybe_accuracy": val_metrics["maybe_accuracy"],
            "lr": scheduler.get_last_lr()[0] if len(scheduler.get_last_lr()) > 0 else args.lr,
        }
        history.append(record)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            os.makedirs(args.best_ckpt_dir, exist_ok=True)
            model.save_pretrained(args.best_ckpt_dir)
            tokenizer.save_pretrained(args.best_ckpt_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                },
                os.path.join(args.best_ckpt_dir, "trainer_state.pt"),
            )

        payload = {
            "run_name": args.run_name,
            "mode": args.mode,
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        save_metrics(payload, args.metrics_json, args.metrics_csv)

        logging.info(
            f"Epoch {epoch} | "
            f"train_loss={record['train_loss']:.4f} "
            f"val_acc={record['val_accuracy']:.4f} "
            f"yes={record['val_yes_accuracy']:.4f} "
            f"no={record['val_no_accuracy']:.4f} "
            f"maybe={record['val_maybe_accuracy']:.4f}"
        )

        if args.dry_run:
            break

    # Final evaluation using best checkpoint
    logging.info("Loading best checkpoint for final validation/test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(args.best_ckpt_dir).to(device)
    best_tokenizer = AutoTokenizer.from_pretrained(args.best_ckpt_dir)

    val_metrics = evaluate_model(best_model, best_tokenizer, val_loader, device, args.max_length)
    test_metrics = evaluate_model(best_model, best_tokenizer, test_loader, device, args.max_length)

    # Save test predictions
    pred_file = os.path.join(args.results_dir, "test_predictions.jsonl")
    best_model.eval()
    with open(pred_file, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Writing test predictions", leave=False, file=sys.stdout):
                enc = best_tokenizer(
                    batch["context"],
                    batch["question"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                logits = best_model(**enc).logits
                preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                for qid, p in zip(batch["qid"], preds):
                    rec = {"qid": qid, "prediction": ID2LABEL[int(p)]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    final_payload = {
        "run_name": args.run_name,
        "mode": args.mode,
        "args": vars(args),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "final_test_accuracy": float(test_metrics.get("accuracy", 0.0)),
        "final_val_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "timestamp": datetime.now(UTC).isoformat(),
        "best_checkpoint": args.best_ckpt_dir,
        "test_predictions": pred_file,
    }
    save_metrics(final_payload, args.metrics_json, args.metrics_csv)
    mirror_results_if_needed(args.results_dir)

    logging.info(
        f"Training complete | best_epoch={best_epoch} "
        f"best_val_acc={best_val_acc:.4f} "
        f"final_test_acc={test_metrics['accuracy']:.4f}"
    )


def build_args():
    parser = argparse.ArgumentParser(description="Train PubMedQA with baseline/CCT/ablations")

    parser.add_argument("--run_name", type=str, default="pubmedqa")
    parser.add_argument("--mode", type=str, default="cct", choices=["baseline", "cct", "cct_consistency", "cct_contrastive"])

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ori_path", type=str, default=DEFAULT_ORI_PATH)
    parser.add_argument("--test_gt_path", type=str, default=DEFAULT_TEST_GT_PATH)

    parser.add_argument("--model_path", type=str, default=DEFAULT_PUBMEDBERT_PATH)
    parser.add_argument("--t5_model_path", type=str, default=DEFAULT_T5_PATH)

    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.abspath(os.environ.get("NANORESEARCH_RESULTS_DIR", "results")),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lambda_consistency", type=float, default=0.5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    parser.add_argument("--contrastive_margin", type=float, default=0.2)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--paraphrase_batch_size", type=int, default=16)
    parser.add_argument("--paraphrase_cache", type=str, default=os.path.join(DEFAULT_DATA_DIR, "pubmedqa", "data", "paraphrase_cache_t5_small.json"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--quick-eval", dest="quick_eval", action="store_true")
    parser.add_argument("--eval-only", dest="eval_only", action="store_true")

    return parser.parse_args()


def main():
    args = build_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.metrics_json = os.path.join(args.results_dir, "metrics.json")
    args.metrics_csv = os.path.join(args.results_dir, "metrics.csv")
    args.best_ckpt_dir = os.path.join(args.results_dir, "best_model")

    setup_logging(args.results_dir, args.run_name)

    logging.info("Starting training script with arguments:")
    logging.info(json.dumps(vars(args), indent=2))

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.isdir(args.model_path):
        logging.warning(
            "Model path is not a local directory: %s. Will treat it as a model id or use fallbacks.",
            args.model_path,
        )
    if not os.path.isdir(args.t5_model_path) and not args.dry_run:
        raise FileNotFoundError(f"T5 model path not found: {args.t5_model_path}")

    train(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Unhandled error in train.py")
        sys.exit(1)