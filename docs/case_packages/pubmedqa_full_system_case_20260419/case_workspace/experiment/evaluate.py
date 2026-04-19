import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_DATA_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/datasets"
DEFAULT_MODEL_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models"
DEFAULT_EVAL_FILE = os.path.join(DEFAULT_DATA_DIR, "pubmedqa", "data", "test_ground_truth.json")
DEFAULT_SOURCE_FILE = os.path.join(DEFAULT_DATA_DIR, "pubmedqa", "data", "ori_pqal.json")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "PubMedBERT-base")

LABEL2ID = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class QAExample:
    qid: str
    question: str
    context: str
    label: Optional[int] = None
    raw_label: Optional[str] = None


class PubMedQAEvalDataset(Dataset):
    def __init__(self, examples: List[QAExample]):
        if len(examples) == 0:
            raise ValueError("No examples provided to PubMedQAEvalDataset.")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> QAExample:
        return self.examples[idx]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, int):
        if label in ID2LABEL:
            return label
        return None
    if isinstance(label, str):
        key = label.strip().lower()
        if key in LABEL2ID:
            return LABEL2ID[key]
    return None


def _find_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lower_map and d[lower_map[lk]] is not None:
            return d[lower_map[lk]]
    return default


def _normalize_context(context_obj: Any) -> str:
    if context_obj is None:
        return ""
    if isinstance(context_obj, str):
        return context_obj
    if isinstance(context_obj, list):
        vals = [str(x).strip() for x in context_obj if x is not None and str(x).strip()]
        return " ".join(vals)
    if isinstance(context_obj, dict):
        # Common pubmedqa keys
        keys_priority = ["CONTEXTS", "contexts", "LONG_ANSWER", "long_answer", "context", "abstract"]
        for k in keys_priority:
            if k in context_obj:
                return _normalize_context(context_obj[k])
        vals = [str(v).strip() for v in context_obj.values() if v is not None and str(v).strip()]
        return " ".join(vals)
    return str(context_obj)


def parse_examples_from_obj(obj: Any) -> Dict[str, QAExample]:
    parsed: Dict[str, QAExample] = {}

    def build_example(qid: str, item: Any) -> Optional[QAExample]:
        if item is None:
            return None

        # Case: label-only entry
        if isinstance(item, str):
            lbl = normalize_label(item)
            return QAExample(qid=qid, question="", context="", label=lbl, raw_label=item)

        if not isinstance(item, dict):
            return None

        question = _find_first(item, ["QUESTION", "question", "Question", "query", "q"], "")
        context_raw = _find_first(
            item,
            ["CONTEXTS", "contexts", "context", "CONTEXT", "LONG_ANSWER", "long_answer", "abstract"],
            "",
        )
        context = _normalize_context(context_raw)

        raw_label = _find_first(
            item,
            ["final_decision", "answer", "label", "LABEL", "A", "gold_label", "ground_truth"],
            None,
        )
        label = normalize_label(raw_label)

        # Some schemas include nested structures
        if not question:
            nested_q = _find_first(item, ["question_info", "qa", "data"], None)
            if isinstance(nested_q, dict):
                question = _find_first(nested_q, ["QUESTION", "question"], question)
                if not context:
                    context = _normalize_context(
                        _find_first(nested_q, ["CONTEXTS", "contexts", "context"], "")
                    )
                if raw_label is None:
                    raw_label = _find_first(nested_q, ["final_decision", "answer", "label"], None)
                    label = normalize_label(raw_label)

        return QAExample(
            qid=str(qid),
            question=str(question) if question is not None else "",
            context=str(context) if context is not None else "",
            label=label,
            raw_label=str(raw_label) if raw_label is not None else None,
        )

    if isinstance(obj, dict):
        for k, v in obj.items():
            ex = build_example(str(k), v)
            if ex is not None:
                parsed[ex.qid] = ex
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                qid = _find_first(item, ["id", "qid", "pubid", "PMID"], str(i))
                ex = build_example(str(qid), item)
                if ex is not None:
                    parsed[ex.qid] = ex
            elif isinstance(item, str):
                ex = build_example(str(i), item)
                if ex is not None:
                    parsed[ex.qid] = ex
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

    return parsed


def merge_eval_and_source(eval_data: Dict[str, QAExample], source_data: Dict[str, QAExample]) -> List[QAExample]:
    merged: List[QAExample] = []
    all_ids = sorted(set(eval_data.keys()) | set(source_data.keys()))

    for qid in all_ids:
        e = eval_data.get(qid)
        s = source_data.get(qid)

        question = ""
        context = ""
        label = None
        raw_label = None

        if s is not None:
            question = s.question or question
            context = s.context or context
            label = s.label if s.label is not None else label
            raw_label = s.raw_label or raw_label

        if e is not None:
            if e.question:
                question = e.question
            if e.context:
                context = e.context
            if e.label is not None:
                label = e.label
            if e.raw_label is not None:
                raw_label = e.raw_label

        if not question and not context and label is None:
            continue

        merged.append(QAExample(qid=qid, question=question, context=context, label=label, raw_label=raw_label))

    return merged


def build_examples(eval_file: str, source_file: Optional[str] = None) -> List[QAExample]:
    eval_obj = load_json(eval_file)
    eval_data = parse_examples_from_obj(eval_obj)

    if source_file is not None and os.path.exists(source_file):
        source_obj = load_json(source_file)
        source_data = parse_examples_from_obj(source_obj)
        examples = merge_eval_and_source(eval_data, source_data)
    else:
        examples = list(eval_data.values())

    examples = [x for x in examples if x.question.strip() or x.context.strip()]
    if len(examples) == 0:
        raise ValueError("No valid examples after parsing/merging files.")
    return examples


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: int):
    def collate_fn(batch: List[QAExample]) -> Dict[str, Any]:
        questions = [x.question for x in batch]
        contexts = [x.context for x in batch]
        labels = [x.label if x.label is not None else -100 for x in batch]
        qids = [x.qid for x in batch]

        enc = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["qids"] = qids
        return enc

    return collate_fn


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if len(y_true) == 0:
        metrics["overall_accuracy"] = None
        for name in LABEL2ID.keys():
            metrics[f"{name}_accuracy"] = None
            metrics[f"{name}_total"] = 0
        metrics["n_samples_with_labels"] = 0
        return metrics

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    overall = float((y_true_np == y_pred_np).mean())
    metrics["overall_accuracy"] = overall
    metrics["n_samples_with_labels"] = int(len(y_true_np))

    for label_name, label_id in LABEL2ID.items():
        mask = y_true_np == label_id
        total = int(mask.sum())
        if total == 0:
            acc = None
        else:
            acc = float((y_pred_np[mask] == y_true_np[mask]).mean())
        metrics[f"{label_name}_accuracy"] = acc
        metrics[f"{label_name}_total"] = total

    # Confusion matrix [true, pred]
    conf = np.zeros((len(LABEL2ID), len(LABEL2ID)), dtype=int)
    for t, p in zip(y_true_np, y_pred_np):
        if t in ID2LABEL and p in ID2LABEL:
            conf[t, p] += 1
    metrics["confusion_matrix"] = conf.tolist()
    metrics["label_order"] = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    return metrics


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_metrics_csv(path: str, metrics: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, (list, dict)):
                writer.writerow([k, json.dumps(v, ensure_ascii=False)])
            else:
                writer.writerow([k, v])


def load_model_and_tokenizer(checkpoint_path: str, num_labels: int, local_files_only: bool = True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=local_files_only, use_fast=True)
    except Exception as e:
        if local_files_only:
            logging.warning("Tokenizer local load failed (%s). Retrying with local_files_only=False.", str(e))
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=False, use_fast=True)
        else:
            raise

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
    except Exception as e:
        if local_files_only:
            logging.warning("Model local load failed (%s). Retrying with local_files_only=False.", str(e))
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=False,
            )
        else:
            raise

    return model, tokenizer


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", leave=False)
        for step, batch in enumerate(progress):
            qids = batch.pop("qids")
            labels = batch["labels"].clone()
            batch = {k: v.to(device) for k, v in batch.items() if k != "qids"}

            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            preds_cpu = preds.detach().cpu().numpy().tolist()
            probs_cpu = probs.detach().cpu().numpy().tolist()
            labels_cpu = labels.detach().cpu().numpy().tolist()

            for qid, pred_id, prob_row, true_id in zip(qids, preds_cpu, probs_cpu, labels_cpu):
                row: Dict[str, Any] = {
                    "id": qid,
                    "pred_id": int(pred_id),
                    "pred_label": ID2LABEL.get(int(pred_id), str(pred_id)),
                    "prob_yes": float(prob_row[LABEL2ID["yes"]]) if len(prob_row) > LABEL2ID["yes"] else None,
                    "prob_no": float(prob_row[LABEL2ID["no"]]) if len(prob_row) > LABEL2ID["no"] else None,
                    "prob_maybe": float(prob_row[LABEL2ID["maybe"]]) if len(prob_row) > LABEL2ID["maybe"] else None,
                }

                if true_id != -100:
                    true_id_int = int(true_id)
                    row["true_id"] = true_id_int
                    row["true_label"] = ID2LABEL.get(true_id_int, str(true_id_int))
                    row["correct"] = int(true_id_int == int(pred_id))
                    y_true.append(true_id_int)
                    y_pred.append(int(pred_id))
                else:
                    row["true_id"] = None
                    row["true_label"] = None
                    row["correct"] = None

                prediction_rows.append(row)

            if dry_run:
                logging.info("Dry-run enabled: stopping evaluation after 1 batch.")
                break

    metrics = compute_metrics(y_true, y_pred)
    return metrics, prediction_rows


def write_predictions(predictions_json_path: str, predictions_csv_path: str, rows: List[Dict[str, Any]]) -> None:
    save_json(predictions_json_path, rows)
    if len(rows) == 0:
        with open(predictions_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "pred_id", "pred_label", "true_id", "true_label", "correct"])
        return

    fieldnames = sorted(rows[0].keys())
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PubMedQA classifier with overall/per-class accuracy.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval_file", type=str, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--source_file", type=str, default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--metrics_out", type=str, default=None, help="Path to metrics.json")
    parser.add_argument("--metrics_csv_out", type=str, default=None, help="Path to metrics.csv")
    parser.add_argument("--predictions_json", type=str, default=None, help="Path to predictions JSON")
    parser.add_argument("--predictions_csv", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for evaluation samples.")
    parser.add_argument("--quick-eval", action="store_true", help="Run fast evaluation on a tiny subset.")
    parser.add_argument("--dry-run", action="store_true", help="Sanity-check pipeline with a single batch.")
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint/model path not found: {args.checkpoint_path}")
    if args.source_file is not None and not os.path.exists(args.source_file):
        logging.warning("Source file does not exist and will be ignored: %s", args.source_file)
        args.source_file = None
    os.makedirs(args.results_dir, exist_ok=True)

    if args.metrics_out is None:
        args.metrics_out = os.path.join(args.results_dir, "metrics.json")
    if args.metrics_csv_out is None:
        args.metrics_csv_out = os.path.join(args.results_dir, "metrics.csv")
    if args.predictions_json is None:
        args.predictions_json = os.path.join(args.results_dir, "predictions.json")
    if args.predictions_csv is None:
        args.predictions_csv = os.path.join(args.results_dir, "predictions.csv")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    setup_logging()
    args = parse_args()

    try:
        ensure_paths(args)
        set_seed(args.seed)

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        logging.info("Using device: %s", device)
        logging.info("Loading evaluation examples from: %s", args.eval_file)
        examples = build_examples(args.eval_file, args.source_file)
        logging.info("Loaded %d total examples before sampling.", len(examples))

        if args.quick_eval:
            limit = 64
            examples = examples[:limit]
            logging.info("Quick-eval enabled: using first %d examples.", len(examples))

        if args.max_samples is not None:
            examples = examples[: args.max_samples]
            logging.info("max_samples applied: using first %d examples.", len(examples))

        if len(examples) == 0:
            raise ValueError("No examples available for evaluation after sampling.")

        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint_path,
            num_labels=args.num_labels,
            local_files_only=True,
        )
        model.to(device)

        dataset = PubMedQAEvalDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_builder(tokenizer, args.max_length),
        )

        metrics, pred_rows = evaluate(model, dataloader, device, dry_run=args.dry_run)

        # enrich prediction rows with source text for traceability
        ex_map = {e.qid: e for e in examples}
        for r in pred_rows:
            qid = r["id"]
            ex = ex_map.get(qid)
            if ex is not None:
                r["question"] = ex.question
                r["context"] = ex.context
                if r.get("true_label") is None and ex.label is not None:
                    r["true_id"] = ex.label
                    r["true_label"] = ID2LABEL.get(ex.label, str(ex.label))
                    r["correct"] = int(ex.label == r["pred_id"])

        write_predictions(args.predictions_json, args.predictions_csv, pred_rows)
        save_json(args.metrics_out, metrics)
        save_metrics_csv(args.metrics_csv_out, metrics)

        logging.info("Evaluation complete.")
        logging.info("Metrics: %s", json.dumps(metrics, indent=2))
        logging.info("Saved metrics JSON to: %s", args.metrics_out)
        logging.info("Saved metrics CSV to: %s", args.metrics_csv_out)
        logging.info("Saved predictions JSON to: %s", args.predictions_json)
        logging.info("Saved predictions CSV to: %s", args.predictions_csv)

    except Exception as e:
        logging.exception("Fatal error during evaluation: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()