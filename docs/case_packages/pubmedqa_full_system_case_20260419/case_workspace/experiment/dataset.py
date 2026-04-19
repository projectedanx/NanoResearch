import os
import re
import json
import math
import random
import logging
import zipfile
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_DATA_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/datasets"
DEFAULT_T5_MODEL_DIR = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models/T5-small"

LABEL2ID = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    return LABEL2ID.copy(), ID2LABEL.copy()


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _decompress_archives(root_dir: str) -> None:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Data root directory not found: {root_dir}")

    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(current_root, filename)
            lower_name = filename.lower()

            try:
                if lower_name.endswith(".zip"):
                    extract_dir = file_path[:-4]
                    if not os.path.exists(extract_dir):
                        logger.info("Decompressing zip archive: %s", file_path)
                        os.makedirs(extract_dir, exist_ok=True)
                        with zipfile.ZipFile(file_path, "r") as zf:
                            zf.extractall(extract_dir)

                elif lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
                    extract_dir = file_path.rsplit(".", 2)[0]
                    if not os.path.exists(extract_dir):
                        logger.info("Decompressing tar.gz archive: %s", file_path)
                        os.makedirs(extract_dir, exist_ok=True)
                        with tarfile.open(file_path, "r:gz") as tf:
                            tf.extractall(extract_dir)

                elif lower_name.endswith(".tar"):
                    extract_dir = file_path[:-4]
                    if not os.path.exists(extract_dir):
                        logger.info("Decompressing tar archive: %s", file_path)
                        os.makedirs(extract_dir, exist_ok=True)
                        with tarfile.open(file_path, "r:") as tf:
                            tf.extractall(extract_dir)
            except Exception as e:
                logger.warning("Failed to decompress %s due to: %s", file_path, str(e))


def _find_pubmedqa_files(data_dir: str) -> Tuple[str, Optional[str]]:
    _decompress_archives(data_dir)

    ori_candidate = os.path.join(data_dir, "pubmedqa", "data", "ori_pqal.json")
    test_candidate = os.path.join(data_dir, "pubmedqa", "data", "test_ground_truth.json")

    if os.path.isfile(ori_candidate):
        test_path = test_candidate if os.path.isfile(test_candidate) else None
        return ori_candidate, test_path

    # fallback scan
    ori_path = None
    test_path = None
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f == "ori_pqal.json":
                ori_path = os.path.join(root, f)
            elif f == "test_ground_truth.json":
                test_path = os.path.join(root, f)

    if ori_path is None:
        raise FileNotFoundError(
            f"Could not find ori_pqal.json under data directory: {data_dir}"
        )

    return ori_path, test_path


def _extract_question(entry: Dict[str, Any]) -> str:
    for k in ["QUESTION", "question", "Question", "query", "q"]:
        if k in entry:
            q = _normalize_text(entry[k])
            if q:
                return q
    return ""


def _extract_context(entry: Dict[str, Any]) -> str:
    if "LONG_ANSWER" in entry and _normalize_text(entry["LONG_ANSWER"]):
        return _normalize_text(entry["LONG_ANSWER"])
    if "long_answer" in entry and _normalize_text(entry["long_answer"]):
        return _normalize_text(entry["long_answer"])

    contexts = None
    for k in ["CONTEXTS", "contexts", "context_list"]:
        if k in entry:
            contexts = entry[k]
            break
    if isinstance(contexts, list):
        contexts = [_normalize_text(x) for x in contexts if _normalize_text(x)]
        if contexts:
            return " ".join(contexts)

    for k in ["CONTEXT", "context", "abstract", "passage"]:
        if k in entry and _normalize_text(entry[k]):
            return _normalize_text(entry[k])

    return ""


def _extract_label(entry: Dict[str, Any]) -> Optional[str]:
    candidate_keys = [
        "final_decision",
        "FINAL_DECISION",
        "label",
        "LABEL",
        "answer",
        "ANSWER",
        "decision",
    ]
    for k in candidate_keys:
        if k in entry:
            val = _normalize_text(entry[k]).lower()
            if val in LABEL2ID:
                return val

    # Sometimes decisions can be in "yes/no/maybe" variants
    for k in candidate_keys:
        if k in entry:
            val = _normalize_text(entry[k]).lower()
            if "yes" in val:
                return "yes"
            if "no" in val:
                return "no"
            if "maybe" in val or "unknown" in val:
                return "maybe"
    return None


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_samples_from_ori(ori_data: Any) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    if isinstance(ori_data, dict):
        iterator = ori_data.items()
    elif isinstance(ori_data, list):
        iterator = [(str(i), row) for i, row in enumerate(ori_data)]
    else:
        raise ValueError(f"Unsupported ori_pqal format type: {type(ori_data)}")

    skipped = 0
    for sid, entry in iterator:
        if not isinstance(entry, dict):
            skipped += 1
            continue

        q = _extract_question(entry)
        c = _extract_context(entry)
        y = _extract_label(entry)

        if not q or not c or y is None:
            skipped += 1
            continue

        samples.append(
            {
                "id": str(sid),
                "question": q,
                "context": c,
                "label": y,
                "label_id": LABEL2ID[y],
            }
        )

    logger.info("Built %d valid samples from ori_pqal (skipped %d).", len(samples), skipped)
    return samples


def _deterministic_split(
    samples: List[Dict[str, Any]],
    val_size: int = 500,
    test_size: int = 0,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    items = list(samples)
    rng.shuffle(items)

    n = len(items)
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative")
    if val_size + test_size >= n:
        # keep at least one training sample
        val_size = max(1, min(val_size, n // 5))
        test_size = max(0, min(test_size, n // 10))
        if val_size + test_size >= n:
            test_size = 0
            val_size = max(1, n - 1)

    test = items[:test_size] if test_size > 0 else []
    val = items[test_size:test_size + val_size]
    train = items[test_size + val_size:]

    return {"train": train, "validation": val, "test": test}


def load_pubmedqa_splits(
    data_dir: str = DEFAULT_DATA_DIR,
    seed: int = 42,
    val_size: int = 500,
) -> Dict[str, List[Dict[str, Any]]]:
    ori_path, test_gt_path = _find_pubmedqa_files(data_dir)
    logger.info("Loading PubMedQA ori file from: %s", ori_path)
    ori_data = _load_json(ori_path)
    all_samples = _build_samples_from_ori(ori_data)

    if len(all_samples) < 2:
        raise RuntimeError("Too few valid samples parsed from ori_pqal.json")

    # If ground truth file exists and contains IDs overlapping ori samples, use as test IDs.
    test_ids = set()
    if test_gt_path and os.path.isfile(test_gt_path):
        try:
            gt_data = _load_json(test_gt_path)
            if isinstance(gt_data, dict):
                test_ids = set(str(k) for k in gt_data.keys())
                logger.info("Loaded %d test IDs from ground truth file.", len(test_ids))
        except Exception as e:
            logger.warning("Failed to parse test_ground_truth.json: %s", str(e))

    if test_ids:
        test_samples = [s for s in all_samples if s["id"] in test_ids]
        remain_samples = [s for s in all_samples if s["id"] not in test_ids]

        if len(remain_samples) < 2:
            # fallback to deterministic split across all
            logger.warning("Insufficient remain samples after test split; fallback to deterministic split.")
            splits = _deterministic_split(all_samples, val_size=val_size, test_size=0, seed=seed)
        else:
            # Validation split from remain only
            shuffled = list(remain_samples)
            random.Random(seed).shuffle(shuffled)
            v = min(val_size, max(1, len(shuffled) // 3))
            val_samples = shuffled[:v]
            train_samples = shuffled[v:]
            splits = {"train": train_samples, "validation": val_samples, "test": test_samples}
    else:
        # No reliable official split available locally; deterministic split for reproducibility.
        logger.warning("No usable official test IDs found. Using deterministic train/validation split from ori_pqal.")
        splits = _deterministic_split(all_samples, val_size=val_size, test_size=0, seed=seed)

    for split_name, split_data in splits.items():
        logger.info("Split '%s': %d samples", split_name, len(split_data))
    return splits


class T5Paraphraser:
    def __init__(
        self,
        model_dir: str = DEFAULT_T5_MODEL_DIR,
        device: Optional[str] = None,
        max_input_length: int = 128,
        max_output_length: int = 48,
        num_beams: int = 4,
    ) -> None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"T5 model directory not found: {model_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams

        logger.info("Loading T5 paraphraser from %s on %s", model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, questions: List[str]) -> List[str]:
        prompts = [f"paraphrase: {q}" for q in questions]
        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **model_inputs,
            max_length=self.max_output_length,
            num_beams=self.num_beams,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        cleaned: List[str] = []
        for src_q, para in zip(questions, decoded):
            para = _normalize_text(para)
            if not para:
                para = src_q
            cleaned.append(para)
        return cleaned


def build_paraphrase_cache(
    train_samples: List[Dict[str, Any]],
    cache_path: str,
    t5_model_dir: str = DEFAULT_T5_MODEL_DIR,
    batch_size: int = 16,
    device: Optional[str] = None,
    force_rebuild: bool = False,
) -> Dict[str, str]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache: Dict[str, str] = {}
    if os.path.isfile(cache_path) and not force_rebuild:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cache = {str(k): _normalize_text(v) for k, v in raw.items()}
            logger.info("Loaded paraphrase cache: %s (%d entries)", cache_path, len(cache))
        except Exception as e:
            logger.warning("Failed to load existing paraphrase cache. Rebuilding. Error: %s", str(e))
            cache = {}

    missing = [s for s in train_samples if str(s["id"]) not in cache]
    if not missing and cache:
        logger.info("Paraphrase cache already complete.")
        return cache

    paraphraser = T5Paraphraser(model_dir=t5_model_dir, device=device)

    logger.info("Generating paraphrases for %d missing training samples...", len(missing))
    for i in tqdm(range(0, len(missing), batch_size), desc="Paraphrasing"):
        batch = missing[i:i + batch_size]
        questions = [s["question"] for s in batch]
        ids = [str(s["id"]) for s in batch]

        paras = paraphraser.generate_batch(questions)
        for sid, p, q in zip(ids, paras, questions):
            cache[sid] = p if p else q

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("Saved paraphrase cache to: %s (%d entries)", cache_path, len(cache))
    return cache


class PubMedQADataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        split: str,
        mode: str = "baseline",
        paraphrase_cache: Optional[Dict[str, str]] = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{split}'")
        if mode not in {"baseline", "cct", "paraphrase_only"}:
            raise ValueError(f"Invalid mode '{mode}'")

        self.samples = samples
        self.split = split
        self.mode = mode
        self.paraphrase_cache = paraphrase_cache or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sid = str(sample["id"])

        para = self.paraphrase_cache.get(sid, "")
        if not para:
            para = sample["question"]

        return {
            "id": sid,
            "question": sample["question"],
            "context": sample["context"],
            "paraphrase": para,
            "label_id": int(sample["label_id"]),
            "label": sample["label"],
        }


class PubMedQACollator:
    def __init__(
        self,
        tokenizer: Any,
        mode: str = "baseline",
        max_length: int = 512,
        add_negative_pairs: bool = True,
    ) -> None:
        if mode not in {"baseline", "cct", "paraphrase_only"}:
            raise ValueError(f"Invalid mode '{mode}'")
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.add_negative_pairs = add_negative_pairs

    def _encode_pairs(self, questions: List[str], contexts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            questions,
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Received empty batch in PubMedQACollator")

        ids = [x["id"] for x in batch]
        labels = torch.tensor([int(x["label_id"]) for x in batch], dtype=torch.long)
        questions = [x["question"] for x in batch]
        contexts = [x["context"] for x in batch]
        paraphrases = [x.get("paraphrase", q) or q for x, q in zip(batch, questions)]

        if self.mode == "paraphrase_only":
            main_q = paraphrases
        else:
            main_q = questions

        main_enc = self._encode_pairs(main_q, contexts)
        output: Dict[str, Any] = {
            "ids": ids,
            "labels": labels,
            "input_ids": main_enc["input_ids"],
            "attention_mask": main_enc["attention_mask"],
        }

        if "token_type_ids" in main_enc:
            output["token_type_ids"] = main_enc["token_type_ids"]

        if self.mode == "cct":
            para_enc = self._encode_pairs(paraphrases, contexts)
            output["para_input_ids"] = para_enc["input_ids"]
            output["para_attention_mask"] = para_enc["attention_mask"]
            if "token_type_ids" in para_enc:
                output["para_token_type_ids"] = para_enc["token_type_ids"]

            if self.add_negative_pairs and len(batch) > 1:
                # Deterministic in-batch shift for negatives (irrelevant question/context pair)
                neg_questions = questions[1:] + questions[:1]
                neg_enc = self._encode_pairs(neg_questions, contexts)
                output["neg_input_ids"] = neg_enc["input_ids"]
                output["neg_attention_mask"] = neg_enc["attention_mask"]
                if "token_type_ids" in neg_enc:
                    output["neg_token_type_ids"] = neg_enc["token_type_ids"]

        return output


def create_pubmedqa_datasets(
    data_dir: str = DEFAULT_DATA_DIR,
    t5_model_dir: str = DEFAULT_T5_MODEL_DIR,
    mode: str = "baseline",
    seed: int = 42,
    val_size: int = 500,
    paraphrase_batch_size: int = 16,
    paraphrase_device: Optional[str] = None,
    force_rebuild_paraphrase_cache: bool = False,
    quick_eval: bool = False,
    dry_run: bool = False,
) -> Dict[str, PubMedQADataset]:
    splits = load_pubmedqa_splits(data_dir=data_dir, seed=seed, val_size=val_size)

    if quick_eval:
        # tiny subset for fast end-to-end checks
        splits["train"] = splits["train"][: min(64, len(splits["train"]))]
        splits["validation"] = splits["validation"][: min(64, len(splits["validation"]))]
        if "test" in splits:
            splits["test"] = splits["test"][: min(64, len(splits["test"]))]

    if dry_run:
        splits["train"] = splits["train"][: min(16, len(splits["train"]))]
        splits["validation"] = splits["validation"][: min(16, len(splits["validation"]))]
        if "test" in splits:
            splits["test"] = splits["test"][: min(16, len(splits["test"]))]

    paraphrase_cache: Dict[str, str] = {}
    if mode in {"cct", "paraphrase_only"}:
        cache_dir = os.path.join(data_dir, "pubmedqa", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "paraphrases_t5_small.json")
        paraphrase_cache = build_paraphrase_cache(
            train_samples=splits["train"],
            cache_path=cache_file,
            t5_model_dir=t5_model_dir,
            batch_size=paraphrase_batch_size,
            device=paraphrase_device,
            force_rebuild=force_rebuild_paraphrase_cache,
        )

    datasets_dict = {
        "train": PubMedQADataset(
            samples=splits["train"],
            split="train",
            mode=mode,
            paraphrase_cache=paraphrase_cache,
        ),
        "validation": PubMedQADataset(
            samples=splits["validation"],
            split="validation",
            mode=mode,
            paraphrase_cache=paraphrase_cache,
        ),
    }

    if "test" in splits and len(splits["test"]) > 0:
        datasets_dict["test"] = PubMedQADataset(
            samples=splits["test"],
            split="test",
            mode=mode,
            paraphrase_cache=paraphrase_cache,
        )

    return datasets_dict


def create_dataloaders(
    tokenizer: Any,
    data_dir: str = DEFAULT_DATA_DIR,
    t5_model_dir: str = DEFAULT_T5_MODEL_DIR,
    mode: str = "baseline",
    seed: int = 42,
    val_size: int = 500,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    max_length: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
    paraphrase_batch_size: int = 16,
    paraphrase_device: Optional[str] = None,
    force_rebuild_paraphrase_cache: bool = False,
    quick_eval: bool = False,
    dry_run: bool = False,
) -> Dict[str, DataLoader]:
    if eval_batch_size is None:
        eval_batch_size = batch_size

    datasets_dict = create_pubmedqa_datasets(
        data_dir=data_dir,
        t5_model_dir=t5_model_dir,
        mode=mode,
        seed=seed,
        val_size=val_size,
        paraphrase_batch_size=paraphrase_batch_size,
        paraphrase_device=paraphrase_device,
        force_rebuild_paraphrase_cache=force_rebuild_paraphrase_cache,
        quick_eval=quick_eval,
        dry_run=dry_run,
    )

    collator = PubMedQACollator(
        tokenizer=tokenizer,
        mode=mode,
        max_length=max_length,
        add_negative_pairs=True,
    )

    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets_dict.items():
        is_train = split == "train"
        bsz = batch_size if is_train else eval_batch_size
        if len(ds) == 0:
            logger.warning("Skipping DataLoader creation for empty split: %s", split)
            continue

        loaders[split] = DataLoader(
            ds,
            batch_size=bsz,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collator,
            drop_last=False,
        )

    return loaders


if __name__ == "__main__":
    # Minimal smoke test
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Running dataset.py smoke test...")

    data_dir = DEFAULT_DATA_DIR
    t5_dir = DEFAULT_T5_MODEL_DIR

    # use local PubMedBERT tokenizer path if available; fallback to T5 tokenizer for shape sanity.
    pubmedbert_dir = "/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01/models/PubMedBERT-base"
    tok_dir = pubmedbert_dir if os.path.isdir(pubmedbert_dir) else t5_dir

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    loaders = create_dataloaders(
        tokenizer=tokenizer,
        data_dir=data_dir,
        t5_model_dir=t5_dir,
        mode="cct",
        batch_size=4,
        eval_batch_size=4,
        max_length=256,
        quick_eval=True,
        dry_run=True,
    )

    for split, loader in loaders.items():
        batch = next(iter(loader))
        logger.info(
            "Split=%s batch keys=%s batch_size=%d",
            split,
            list(batch.keys()),
            batch["input_ids"].shape[0],
        )
        break