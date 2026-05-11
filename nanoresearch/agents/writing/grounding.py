"""Evidence grounding: experiment normalization, grounding packet construction."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ._types import GroundingPacket
from . import _escape_latex_text
from .grounding_tables import (
    _GroundingTablesMixin,
    _format_paper_number,
    _metric_priority,
    _short_metric_name,
)

logger = logging.getLogger(__name__)


class _GroundingMixin(_GroundingTablesMixin):
    """Mixin — grounding and table methods."""

    @staticmethod
    def _entry_display_name(entry: dict, *, default: str = "") -> str:
        """Return a readable method/variant name from real artifact fields."""
        for key in ("method_name", "variant_name", "method", "model_name", "name", "run_id"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != "?":
                text = value.strip()
                if key == "run_id":
                    text = text.replace("_", " ").replace("-", " ").strip().title()
                return text
        role = entry.get("role")
        if isinstance(role, str) and role.strip():
            return role.strip().replace("_", " ").title()
        return default

    @classmethod
    def _normalize_result_entries(
        cls,
        entries: Any,
        *,
        kind: str,
        blueprint: dict,
    ) -> list[dict]:
        """Normalize real result rows without inventing metrics."""
        if not isinstance(entries, list):
            return []
        normalized_entries: list[dict] = []
        seen: set[tuple] = set()
        datasets = blueprint.get("datasets", [])
        fallback_dataset = "Unknown Dataset"
        if isinstance(datasets, list) and datasets:
            first = datasets[0]
            if isinstance(first, dict):
                fallback_dataset = str(first.get("name") or fallback_dataset)
            elif isinstance(first, str):
                fallback_dataset = first
        proposed_default = (
            (blueprint.get("proposed_method") or {}).get("name")
            or blueprint.get("method_name")
            or "Proposed Method"
        )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            copied = dict(entry)
            name = cls._entry_display_name(
                copied,
                default=proposed_default if kind == "main" and copied.get("role") == "proposed" else "",
            )
            if not name:
                continue
            if kind == "main":
                copied["method_name"] = name
                copied["is_proposed"] = bool(
                    copied.get("is_proposed")
                    or str(copied.get("role", "")).lower() == "proposed"
                )
            else:
                copied["variant_name"] = name
            copied["dataset"] = str(copied.get("dataset") or fallback_dataset)
            raw_run_id = str(copied.get("run_id") or "")
            core_run_id = re.sub(r"^(baseline|ablation)_\d+_", r"\1_", raw_run_id)
            name_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            dedupe_key = (kind, copied.get("role"), name_key, copied.get("dataset"))
            run_key = (kind, core_run_id) if core_run_id else None
            if dedupe_key in seen or (run_key is not None and run_key in seen):
                continue
            seen.add(dedupe_key)
            if run_key is not None:
                seen.add(run_key)
            normalized_entries.append(copied)
        return normalized_entries

    @classmethod
    def _normalize_experiment_results(
        cls,
        experiment_results: dict,
        blueprint: dict,
        experiment_analysis: dict,
    ) -> dict:
        """Coerce raw execution/analysis metrics into the main_results schema."""
        normalized = dict(experiment_results) if isinstance(experiment_results, dict) else {}
        analysis_payload = experiment_analysis if isinstance(experiment_analysis, dict) else {}
        main_results = normalized.get("main_results")
        if isinstance(main_results, list) and main_results:
            normalized["main_results"] = cls._normalize_result_entries(
                main_results, kind="main", blueprint=blueprint,
            )
            if not normalized.get("ablation_results") and isinstance(
                analysis_payload.get("ablation_results"), list
            ):
                normalized["ablation_results"] = analysis_payload.get("ablation_results", [])
            normalized["ablation_results"] = cls._normalize_result_entries(
                normalized.get("ablation_results"), kind="ablation", blueprint=blueprint,
            )
            return normalized

        metric_snapshot = analysis_payload.get("final_metrics", {})
        if not isinstance(metric_snapshot, dict) or not metric_snapshot:
            metric_snapshot = {
                key: value
                for key, value in normalized.items()
                if isinstance(value, (int, float, str, bool))
            }
        if not metric_snapshot:
            return normalized

        datasets = blueprint.get("datasets", [])
        dataset_name = "Unknown Dataset"
        if isinstance(datasets, list) and datasets:
            first_dataset = datasets[0]
            if isinstance(first_dataset, dict):
                dataset_name = str(first_dataset.get("name", dataset_name)) or dataset_name
            else:
                dataset_name = str(first_dataset) or dataset_name

        method_name = (
            (blueprint.get("proposed_method") or {}).get("name")
            or "Proposed Method"
        )
        normalized["main_results"] = [
            {
                "method_name": method_name,
                "dataset": dataset_name,
                "is_proposed": True,
                "metrics": [
                    {"metric_name": key, "value": value}
                    for key, value in metric_snapshot.items()
                ],
            }
        ]
        if not normalized.get("ablation_results") and isinstance(
            analysis_payload.get("ablation_results"), list
        ):
            normalized["ablation_results"] = analysis_payload.get("ablation_results", [])
        normalized["main_results"] = cls._normalize_result_entries(
            normalized.get("main_results"), kind="main", blueprint=blueprint,
        )
        normalized["ablation_results"] = cls._normalize_result_entries(
            normalized.get("ablation_results"), kind="ablation", blueprint=blueprint,
        )
        return normalized

    # ---- grounding packet construction ----------------------------------------

    @classmethod
    def _classify_completeness(
        cls,
        experiment_status: str,
        main_results: list[dict],
        experiment_analysis: dict,
    ) -> ResultCompleteness:
        """Classify how complete the experiment results are."""
        status_lower = (experiment_status or "").lower()
        if status_lower in ("pending", "failed", "error", "unknown", ""):
            return "none"
        if not main_results:
            return "none"
        # Check for quick-eval markers
        is_quick = (
            "quick" in status_lower
            or experiment_analysis.get("is_quick_eval", False)
            or "quick-eval" in experiment_analysis.get("summary", "").lower()
            or "quick_eval" in status_lower
        )
        if is_quick:
            return "quick_eval"
        # Check for partial results (e.g., only 1 dataset out of planned N)
        converged = experiment_analysis.get("converged")
        if converged is False:
            return "partial"
        return "full"

    @classmethod
    def _build_grounding_packet(
        cls,
        experiment_results: dict,
        experiment_status: str,
        experiment_analysis: dict,
        experiment_summary: str,
        blueprint: dict,
    ) -> GroundingPacket:
        """Build a GroundingPacket from all available evidence sources."""
        normalized = cls._normalize_experiment_results(
            experiment_results or {}, blueprint, experiment_analysis or {}
        )
        analysis = experiment_analysis if isinstance(experiment_analysis, dict) else {}
        main_results = normalized.get("main_results", [])
        if not isinstance(main_results, list):
            main_results = []
        ablation_results = normalized.get("ablation_results", [])
        if not isinstance(ablation_results, list):
            ablation_results = []
        comparison = analysis.get("comparison_with_baselines", {})
        if not isinstance(comparison, dict):
            comparison = {}
        final_metrics = analysis.get("final_metrics", {})
        if not isinstance(final_metrics, dict):
            final_metrics = {}

        completeness = cls._classify_completeness(
            experiment_status, main_results, analysis,
        )

        contract_gaps = cls._artifact_contract_gaps(
            blueprint, main_results, ablation_results, normalized
        )
        if completeness != "none" and contract_gaps:
            completeness = "partial"

        # Identify evidence gaps
        gaps: list[str] = []
        if completeness == "none":
            gaps.append("No experiment results available")
        elif completeness == "quick_eval":
            gaps.append("Results are from quick-eval only (limited epochs/data)")
        gaps.extend(contract_gaps)
        if not ablation_results:
            gaps.append("No ablation study results")
        if not comparison:
            gaps.append("No baseline comparison data from analysis")

        packet = GroundingPacket(
            experiment_status=experiment_status,
            result_completeness=completeness,
            main_results=main_results,
            ablation_results=ablation_results,
            comparison_with_baselines=comparison,
            final_metrics=final_metrics,
            key_findings=analysis.get("key_findings", []) or [],
            limitations=analysis.get("limitations", []) or [],
            training_dynamics=str(analysis.get("training_dynamics", "")),
            analysis_summary=str(analysis.get("summary", "")),
            experiment_summary_md=experiment_summary or "",
            evidence_gaps=gaps,
        )

        # Pre-build deterministic tables when data is available
        if packet.has_real_results:
            packet.main_table_latex = cls._build_main_table_latex(
                main_results, comparison, blueprint,
            )
            if ablation_results:
                packet.ablation_table_latex = cls._build_ablation_table_latex(
                    ablation_results, blueprint,
                )
        else:
            # No verified measured results: do not create result-looking tables.
            # Do not create result-looking tables without verified metrics.
            # Missing categories should be omitted from main results; if they
            # matter for interpretation, discuss scope in paper-facing prose.
            packet.main_table_latex = ""
            packet.ablation_table_latex = ""

        return packet

    @staticmethod
    def _artifact_contract_gaps(
        blueprint: dict,
        main_results: list[dict],
        ablation_results: list[dict],
        normalized: dict,
    ) -> list[str]:
        """Validate that paper-facing result artifacts satisfy the blueprint contract."""
        criteria = blueprint.get("minimum_success_criteria", {}) if isinstance(blueprint, dict) else {}
        if not isinstance(criteria, dict):
            criteria = {}

        proposed_count = 0
        measured_baseline_count = 0
        for entry in main_results:
            if not isinstance(entry, dict) or not entry.get("metrics"):
                continue
            role = str(entry.get("role") or "").lower()
            is_proposed = bool(entry.get("is_proposed")) or role == "proposed"
            if is_proposed:
                proposed_count += 1
            else:
                measured_baseline_count += 1

        measured_ablation_count = sum(
            1 for entry in ablation_results
            if isinstance(entry, dict) and bool(entry.get("metrics"))
        )
        gaps: list[str] = []
        if criteria.get("require_proposed", True) and proposed_count < 1:
            gaps.append("Missing measured proposed-method result")
        min_baselines = int(criteria.get("min_measured_baselines", 0) or 0)
        if measured_baseline_count < min_baselines:
            gaps.append(
                f"Missing measured baselines: expected >= {min_baselines}, got {measured_baseline_count}"
            )
        min_ablations = int(criteria.get("min_ablation_runs", 0) or 0)
        if measured_ablation_count < min_ablations:
            gaps.append(
                f"Missing measured ablations: expected >= {min_ablations}, got {measured_ablation_count}"
            )
        if criteria.get("require_optimization_history") and not (
            normalized.get("optimization_history") or normalized.get("optimization_history_path")
        ):
            gaps.append("Missing measured optimization history")
        if criteria.get("require_complexity") and not normalized.get("complexity_metrics"):
            gaps.append("Missing measured complexity metrics")
        return gaps

    @staticmethod
    def _build_main_table_latex(
        main_results: list[dict],
        comparison: dict,
        blueprint: dict,
    ) -> str:
        """Build a deterministic LaTeX main-results table from structured data.

        Returns empty string if data is insufficient.
        """
        if not main_results:
            return ""

        # Collect all metric names across all entries
        MAX_TABLE_COLS = 7
        all_metrics: list[str] = []
        seen: set[str] = set()
        for entry in main_results:
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                if name and name not in seen:
                    all_metrics.append(name)
                    seen.add(name)
        if not all_metrics:
            return ""
        # Prioritize paper-facing metrics and cap columns to prevent overflow.
        all_metrics = sorted(all_metrics, key=_metric_priority)
        if len(all_metrics) > MAX_TABLE_COLS:
            all_metrics = all_metrics[:MAX_TABLE_COLS]

        # Build rows: first from comparison_with_baselines, then main_results
        rows: list[tuple[str, bool, dict[str, str]]] = []  # (method, is_proposed, {metric: val_str})
        proposed_name = ""

        # Rows from main_results
        for entry in main_results:
            method = entry.get("method_name") or _GroundingMixin._entry_display_name(entry)
            if not method:
                continue
            is_proposed = entry.get("is_proposed", False)
            if is_proposed:
                proposed_name = method
            metric_vals: dict[str, str] = {}
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                val = m.get("value")
                std = m.get("std")
                if val is not None:
                    val_str = _format_paper_number(val)
                    if std is not None:
                        val_str += f" $\\pm$ {_format_paper_number(std)}"
                    metric_vals[name] = val_str
            rows.append((method, is_proposed, metric_vals))

        # Add baseline rows from comparison_with_baselines that aren't already in rows
        existing_methods = {r[0].lower() for r in rows}
        for method_name, method_metrics in comparison.items():
            if method_name.lower() in existing_methods:
                continue
            if method_name.lower() in ("our_method", "proposed", "ours"):
                continue
            if not isinstance(method_metrics, dict):
                continue
            metric_vals = {}
            for metric_name in all_metrics:
                val = method_metrics.get(metric_name)
                if val is not None:
                    metric_vals[metric_name] = _format_paper_number(val)
            if metric_vals:  # only add if has any values
                rows.append((method_name, False, metric_vals))

        if len(rows) < 1:
            return ""

        # Sort: baselines first, proposed method last
        baseline_rows = [r for r in rows if not r[1]]
        proposed_rows = [r for r in rows if r[1]]
        sorted_rows = baseline_rows + proposed_rows

        # Build LaTeX
        n_metrics = len(all_metrics)
        col_spec = "@{}l" + "c" * n_metrics + "@{}"
        header_cells = " & ".join(_escape_latex_text(_short_metric_name(m)) for m in all_metrics)
        use_resizebox = n_metrics >= 4

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            f"\\caption{{Main experimental results. Best results are in \\textbf{{bold}}.}}",
            "\\label{tab:main_results}",
        ]
        if use_resizebox:
            lines.append("\\resizebox{\\linewidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Method & {header_cells} \\\\",
            "\\midrule",
        ])

        # Determine which metrics are lower-is-better
        _LOWER_KW = (
            "loss", "error", "perplexity", "mse", "mae", "rmse", "cer", "wer",
            "fid", "distance", "divergence", "latency", "regret",
            "miss_rate", "false_positive", "eer",
        )
        lower_is_better_metrics: set[str] = {
            mn for mn in all_metrics
            if any(kw in mn.lower().replace(" ", "_").replace("-", "_")
                   for kw in _LOWER_KW)
        }

        # Find best value per metric (for bolding)
        _NUM_RE = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)')

        def _extract_leading_number(s: str) -> float | None:
            """Extract leading numeric value from a metric string like '87.58 +/- 2.99'."""
            m = _NUM_RE.match(s.strip())
            return float(m.group(0)) if m else None

        best_vals: dict[str, float] = {}
        for _, _, mv in sorted_rows:
            for metric_name in all_metrics:
                val_str = mv.get(metric_name, "")
                val_num = _extract_leading_number(val_str)
                if val_num is None:
                    continue
                lower = metric_name in lower_is_better_metrics
                if metric_name not in best_vals:
                    best_vals[metric_name] = val_num
                elif lower and val_num < best_vals[metric_name]:
                    best_vals[metric_name] = val_num
                elif not lower and val_num > best_vals[metric_name]:
                    best_vals[metric_name] = val_num

        for method, is_proposed, metric_vals in sorted_rows:
            cells = []
            for metric_name in all_metrics:
                val_str = metric_vals.get(metric_name, "--")
                # Bold best value
                val_num = _extract_leading_number(val_str)
                if val_num is not None and metric_name in best_vals:
                    if abs(val_num - best_vals[metric_name]) < 1e-9:
                        val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
            method_display = f"{_escape_latex_text(method)} (Ours)" if is_proposed else _escape_latex_text(method)
            lines.append(f"{method_display} & {' & '.join(cells)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
        ])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # Methods moved to grounding_tables.py: _build_ablation_table_latex,
    # _build_scaffold_main_table, _build_scaffold_ablation_table,
    # _build_real_results_context,
    # _build_experiment_analysis_context, _build_baseline_comparison_context,
    # _build_grounding_status_context, _find_table_span, _verify_and_inject_tables,
    # _table_metrics_match, _build_figure_blocks, _resolve_figure_include,
    # _TOOL_SECTIONS

