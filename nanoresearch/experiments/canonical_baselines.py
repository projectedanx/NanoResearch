from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any


_BASELINE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiments"
    / "lightweight_router_persona_canonical_baselines_v1.json"
)


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


@lru_cache(maxsize=1)
def load_canonical_baseline_registry() -> dict[str, dict[str, Any]]:
    rows = json.loads(_BASELINE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Baseline registry must be a JSON array: {_BASELINE_REGISTRY_PATH}")

    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        metrics = []
        for metric in row.get("metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            normalized_aliases = {
                _normalize_metric_name(str(metric.get("metric_name") or ""))
            }
            for alias in metric.get("metric_aliases", []) or []:
                normalized_aliases.add(_normalize_metric_name(str(alias)))
            normalized_aliases.discard("")
            metric_copy = dict(metric)
            metric_copy["normalized_aliases"] = sorted(normalized_aliases)
            metrics.append(metric_copy)
        row_copy = dict(row)
        row_copy["metrics"] = metrics
        registry[question_id] = row_copy
    return registry


def lookup_canonical_baseline(question_id: str, primary_metric_name: str | None) -> dict[str, Any] | None:
    registry = load_canonical_baseline_registry()
    question_key = str(question_id or "").strip()
    if not question_key:
        return None
    topic_entry = registry.get(question_key)
    if not topic_entry:
        return None

    normalized_target = _normalize_metric_name(primary_metric_name or "")
    metrics = list(topic_entry.get("metrics") or [])
    if not metrics:
        return None

    if normalized_target:
        for metric in metrics:
            aliases = set(metric.get("normalized_aliases") or [])
            if normalized_target in aliases:
                return metric

    return metrics[0]
