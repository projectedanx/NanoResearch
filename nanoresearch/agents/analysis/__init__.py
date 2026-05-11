"""Analysis agent — parses real experiment results and generates figures from actual data."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.analysis._analysis_helpers import _AnalysisHelpersMixin
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)


def _flatten_metric_list(metrics) -> dict[str, float]:
    """Convert [{metric_name: str, value: float}, ...] to {name: value}.

    Filters out NaN/Inf values.
    """
    if isinstance(metrics, dict):
        return {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
    if not isinstance(metrics, list):
        return {}
    flat: dict[str, float] = {}
    for m in metrics:
        if isinstance(m, dict):
            name = m.get("metric_name") or m.get("name")
            val = m.get("value")
            if name and isinstance(val, (int, float)) and math.isfinite(val):
                flat[name] = val
    return flat


class AnalysisAgent(_AnalysisHelpersMixin, BaseResearchAgent):
    """Analyzes real experiment results and generates publication figures from actual data."""

    stage = PipelineStage.ANALYSIS

    @property
    def stage_config(self):
        """Use writing model config for result analysis (needs strong reasoning)."""
        return self.config.for_stage("writing")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        execution_output: dict = inputs.get("execution_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting analysis of real experiment results")

        # Step 1: Analyze results (LLM)
        analysis = await self._analyze_results(execution_output, experiment_blueprint)
        if not isinstance(analysis, dict):
            analysis = {}
        analysis = self._sanitize_analysis_against_raw_metrics(analysis, execution_output)
        normalized_evidence = self._normalize_structured_evidence(execution_output, experiment_blueprint)
        analysis.update(normalized_evidence)
        analysis.setdefault("execution_output", execution_output)
        self.log(f"Analysis complete: {list(analysis.keys())}")

        # Step 1.5: Computational analysis (deterministic, no LLM)
        computational = self._compute_analysis(
            execution_output, experiment_blueprint, analysis
        )
        if computational:
            self.log(f"Computational analysis: {list(computational.keys())}")

        # Step 2: Figure generation is handled entirely by FIGURE_GEN agent.
        # ANALYSIS only provides data; it does NOT generate any figures.
        figures = {}
        self.log("Skipping figure generation (handled by FIGURE_GEN agent)")

        # Step 3: Write an experiment summary markdown for downstream writing/review
        summary_markdown = self._render_experiment_summary_markdown(
            analysis,
            execution_output,
            experiment_blueprint,
            computational,
        )
        summary_path = self.workspace.write_text(
            "drafts/experiment_summary.md",
            summary_markdown,
        )

        result = {
            "analysis": analysis,
            "computational_analysis": computational,
            "figures": figures,
            "execution_output": execution_output,
            "experiment_summary": summary_markdown,
            "experiment_summary_path": str(summary_path),
        }

        self.workspace.write_json("plans/analysis_output.json", result)
        return result

    async def _analyze_results(
        self, execution_output: dict, blueprint: dict
    ) -> dict:
        """Use LLM to interpret and summarize experiment results."""
        metrics = execution_output.get("metrics", {})
        parsed_metrics = execution_output.get("parsed_metrics", {})
        stdout_log = execution_output.get("stdout_log", "")[-5000:]
        training_log_csv = execution_output.get("training_log_csv", "")
        final_status = execution_output.get("final_status", "UNKNOWN")

        system_prompt = (
            "You are an ML researcher analyzing experiment results. "
            "Given the training logs and metrics, provide a comprehensive analysis. "
            "Be honest about results — if the model didn't converge or results are poor, say so. "
            "Return JSON only."
        )

        user_prompt = f"""Job Status: {final_status}

Metrics JSON:
{json.dumps(metrics, indent=2)[:3000]}

Parsed Metrics from Log:
{json.dumps(parsed_metrics, indent=2)[:2000]}

Training Log CSV (last part):
{training_log_csv[-3000:] if training_log_csv else 'N/A'}

Stdout Log (last part):
{stdout_log[-3000:]}

Expected Metrics: {json.dumps(blueprint.get('metrics', []), indent=2)[:500]}
Baselines: {json.dumps(blueprint.get('baselines', []), indent=2)[:1000]}
Ablation Groups: {json.dumps(blueprint.get('ablation_groups', []), indent=2)[:1000]}

Analyze these results. Return JSON:
{{
  "summary": "1-2 paragraph summary of results...",
  "converged": true/false,
  "final_metrics": {{"metric_name": value, ...}},
  "comparison_with_baselines": {{
    "our_method": {{"metric1": value, "metric2": value}},
    "baseline1_name": {{"metric1": value_or_null, "metric2": value_or_null}},
    "baseline2_name": {{"metric1": value_or_null, "metric2": value_or_null}}
  }},
  "ablation_results": [],
  "training_dynamics": "Description of training curve behavior...",
  "key_findings": ["finding1", "finding2", ...],
  "limitations": ["limitation1", ...],
  "figures_to_generate": [
    {{
      "figure_id": "fig_training_curve",
      "title": "Training Curve",
      "type": "line",
      "data_source": "training_log"
    }},
    {{
      "figure_id": "fig_results_comparison",
      "title": "Results Summary",
      "type": "bar",
      "data_source": "metrics"
    }}
  ]
}}

IMPORTANT:
- For comparison_with_baselines, return a DICT mapping method names to their metrics.
  Include our method's actual numbers from Metrics JSON. For baselines, report numeric values only if they appear in Metrics JSON; otherwise use null.
  Do not copy blueprint expected_performance into experimental results.
- For ablation_results, use ONLY ablation variants explicitly present in Metrics JSON under ablation_results.
  If Metrics JSON has no ablation_results, return an empty list. Do not create an ablation table or engineering-status sentence.
  Mention the narrower evaluation scope only in paper-facing limitation language if it affects the claims.
  Never estimate, infer, or invent ablation values from training dynamics, blueprint components, or expected results.
- NEVER use "N/A" as a numeric value. Use null for truly unknown values.
- figures_to_generate MUST contain at most 3 figures.
- NEVER propose diagnostic/failure/error figures. No figure titled "failure diagnosis",
  "error analysis", "debug", etc.
- Propose an ablation figure only when real ablation_results are present in Metrics JSON.
- If the experiment failed or a result category is absent, do not invent fallback data; omit the corresponding result table/figure.
- If the absence materially affects the scientific claim, describe the evaluation scope in natural academic prose rather than using run-log or engineering-status wording."""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}


    @staticmethod
    def _normalize_structured_evidence(execution_output: dict, blueprint: dict) -> dict:
        """Normalize machine-generated artifacts into stable paper-facing fields.

        This function never fabricates missing values. Published baselines from
        the blueprint remain separate from measured baseline runs.
        """
        raw_metrics = execution_output.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}

        def flat_metrics(value: Any) -> dict[str, float]:
            return _flatten_metric_list(value)

        measured_results: list[dict[str, Any]] = []
        measured_baselines: list[dict[str, Any]] = []
        for entry in raw_metrics.get("main_results", []) if isinstance(raw_metrics.get("main_results"), list) else []:
            if not isinstance(entry, dict):
                continue
            item = {
                "run_id": str(entry.get("run_id") or ""),
                "method": str(entry.get("method_name") or entry.get("method") or ""),
                "dataset": str(entry.get("dataset") or ""),
                "metrics": flat_metrics(entry.get("metrics", {})),
                "source": "results/metrics.json",
            }
            if not item["metrics"]:
                continue
            role = str(entry.get("role") or "").lower()
            if entry.get("is_proposed") or role == "proposed":
                measured_results.append(item)
            else:
                measured_baselines.append(item)

        ablation_results: list[dict[str, Any]] = []
        for entry in raw_metrics.get("ablation_results", []) if isinstance(raw_metrics.get("ablation_results"), list) else []:
            if not isinstance(entry, dict):
                continue
            metrics = flat_metrics(entry.get("metrics", {}))
            if not metrics:
                continue
            ablation_results.append({
                "run_id": str(entry.get("run_id") or ""),
                "variant_name": str(entry.get("variant_name") or entry.get("method") or ""),
                "metrics": metrics,
                "source": "results/metrics.json",
            })

        optimization_history: list[dict[str, Any]] = []
        csv_text = str(execution_output.get("optimization_history_csv") or "").strip()
        if csv_text:
            try:
                for row in csv.DictReader(io.StringIO(csv_text)):
                    clean_row: dict[str, Any] = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        raw = str(value or "").strip()
                        if raw == "":
                            clean_row[str(key)] = raw
                            continue
                        try:
                            clean_row[str(key)] = float(raw)
                        except ValueError:
                            clean_row[str(key)] = raw
                    if clean_row:
                        optimization_history.append(clean_row)
            except csv.Error:
                optimization_history = []

        final_metrics = execution_output.get("final_metrics")
        if not isinstance(final_metrics, dict):
            final_metrics = {}
        complexity_metrics = raw_metrics.get("complexity_metrics")
        if not isinstance(complexity_metrics, dict):
            complexity_metrics = final_metrics.get("complexity_metrics") if isinstance(final_metrics.get("complexity_metrics"), dict) else {}

        pareto = execution_output.get("pareto_front")
        pareto_summary = pareto if isinstance(pareto, (dict, list)) else {}

        result_contract = execution_output.get("result_contract")
        if not isinstance(result_contract, dict):
            result_contract = {}

        published_baseline_results: list[dict[str, Any]] = []
        for baseline in blueprint.get("baselines", []) if isinstance(blueprint.get("baselines"), list) else []:
            if not isinstance(baseline, dict):
                continue
            perf = baseline.get("expected_performance") if isinstance(baseline.get("expected_performance"), dict) else {}
            provenance = baseline.get("performance_provenance") if isinstance(baseline.get("performance_provenance"), dict) else {}
            for metric, value in perf.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                published_baseline_results.append({
                    "method": str(baseline.get("name") or ""),
                    "dataset": "",
                    "metric": str(metric),
                    "value": float(value),
                    "source": str(provenance.get(metric) or baseline.get("reference_paper_id") or ""),
                    "provenance": "published_literature_context",
                })

        return {
            "measured_results": measured_results,
            "measured_baselines": measured_baselines,
            "ablation_results": ablation_results,
            "optimization_history": optimization_history,
            "complexity_metrics": complexity_metrics if isinstance(complexity_metrics, dict) else {},
            "pareto_summary": pareto_summary,
            "missing_required_runs": list(result_contract.get("missing_required_runs") or []),
            "missing_artifacts": list(result_contract.get("missing_artifacts") or []),
            "failed_runs": list(result_contract.get("failed_runs") or []),
            "published_baseline_results": published_baseline_results,
        }

    @staticmethod
    def _sanitize_analysis_against_raw_metrics(analysis: dict, execution_output: dict) -> dict:
        """Remove LLM-invented result structures that are absent from raw metrics."""
        raw_metrics = execution_output.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}

        proposed_metrics: dict[str, Any] = {}
        main_results = raw_metrics.get("main_results", [])
        if isinstance(main_results, list):
            for entry in main_results:
                if not isinstance(entry, dict) or not entry.get("is_proposed"):
                    continue
                metrics = entry.get("metrics", [])
                if isinstance(metrics, list):
                    for metric in metrics:
                        if not isinstance(metric, dict):
                            continue
                        name = metric.get("metric_name") or metric.get("name")
                        value = metric.get("value")
                        if isinstance(name, str) and isinstance(value, (int, float)):
                            proposed_metrics[name] = value
                elif isinstance(metrics, dict):
                    proposed_metrics.update({
                        str(k): v for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    })
        if proposed_metrics:
            analysis["final_metrics"] = proposed_metrics

        comparison = analysis.get("comparison_with_baselines")
        if isinstance(comparison, dict):
            sanitized_comparison: dict[str, Any] = {}
            if proposed_metrics:
                sanitized_comparison["our_method"] = proposed_metrics
            for name, values in comparison.items():
                if name == "our_method":
                    continue
                if isinstance(values, dict):
                    sanitized_comparison[str(name)] = {str(k): None for k in values}
            analysis["comparison_with_baselines"] = sanitized_comparison

        raw_ablation = raw_metrics.get("ablation_results", [])
        if isinstance(raw_ablation, list) and raw_ablation:
            analysis["ablation_results"] = raw_ablation
        else:
            analysis["ablation_results"] = []
            limitations = analysis.get("limitations")
            if not isinstance(limitations, list):
                limitations = []
            note = "Ablation studies were not executed in this run; no ablation values are reported."
            if note not in limitations:
                limitations.append(note)
            analysis["limitations"] = limitations

        figures = analysis.get("figures_to_generate")
        if isinstance(figures, list) and not analysis.get("ablation_results"):
            analysis["figures_to_generate"] = [
                fig for fig in figures
                if "ablation" not in str(fig.get("figure_id", "")).lower()
                and "ablation" not in str(fig.get("title", "")).lower()
                and "ablation" not in str(fig.get("data_source", "")).lower()
            ]

        return analysis

    # ── Computational analysis (deterministic, no LLM) ──────────────────

    def _compute_analysis(
        self,
        execution_output: dict,
        blueprint: dict,
        llm_analysis: dict,
    ) -> dict:
        """Run deterministic computational analysis alongside LLM analysis."""
        from nanoresearch.agents.analysis.training_dynamics import (
            analyze_training_dynamics,
        )
        from nanoresearch.agents.analysis.ablation_analysis import (
            quantify_ablation_contributions,
        )
        from nanoresearch.agents.analysis.comparison_matrix import (
            build_comparison_matrix,
            comparison_matrix_to_latex,
        )

        result: dict = {}
        raw_metrics = execution_output.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}

        bp_metrics = blueprint.get("metrics", [])
        if not isinstance(bp_metrics, list):
            bp_metrics = []

        # 1. Training dynamics
        training_log = raw_metrics.get("training_log", [])
        if isinstance(training_log, list) and len(training_log) >= 3:
            dynamics = analyze_training_dynamics(training_log)
            result["training_dynamics"] = dynamics

        # 2. Comparison matrix
        main_results = raw_metrics.get("main_results", [])
        bp_baselines = blueprint.get("baselines", [])
        matrix_inputs = self._build_matrix_inputs(
            main_results, bp_baselines, bp_metrics
        )
        if matrix_inputs:
            baselines_list, proposed, metrics_list = matrix_inputs
            matrix = build_comparison_matrix(baselines_list, proposed, metrics_list)
            result["comparison_matrix"] = matrix
            result["comparison_latex"] = comparison_matrix_to_latex(matrix)

        # 3. Ablation contributions
        ablation_raw = raw_metrics.get("ablation_results", [])
        if isinstance(ablation_raw, list) and len(ablation_raw) >= 2:
            primary_metric = self._find_primary_metric(bp_metrics)
            higher = self._metric_higher_is_better(primary_metric, bp_metrics)
            full_result, ablation_variants = self._split_ablation(
                ablation_raw, primary_metric, higher
            )
            if full_result and ablation_variants:
                contributions = quantify_ablation_contributions(
                    full_result, ablation_variants, primary_metric, higher
                )
                result["ablation_contributions"] = contributions

        return result

    @staticmethod
    def _build_matrix_inputs(
        main_results: list,
        bp_baselines: list,
        bp_metrics: list,
    ):
        """Convert pipeline data into (baselines, proposed, metrics) for comparison_matrix.

        Returns None if insufficient data.
        """
        if not isinstance(main_results, list):
            return None

        # Flatten main_results into {name, metrics: {metric: value}}
        proposed = None
        baselines = []
        for entry in main_results:
            if not isinstance(entry, dict):
                continue
            flat = _flatten_metric_list(entry.get("metrics", []))
            item = {"name": entry.get("method_name", "Unknown"), "metrics": flat}
            if entry.get("is_proposed"):
                proposed = item
            else:
                baselines.append(item)

        if proposed is None or not proposed.get("metrics"):
            return None
        if not baselines:
            return None

        # Build metrics list
        metrics_list = []
        seen_m: set[str] = set()
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("name"):
                metrics_list.append({
                    "name": m["name"],
                    "higher_is_better": m.get("higher_is_better", True),
                })
                seen_m.add(m["name"])
        # Add any metric from proposed that's not in blueprint
        for mname in proposed.get("metrics", {}):
            if mname not in seen_m:
                metrics_list.append({"name": mname, "higher_is_better": True})
                seen_m.add(mname)

        if not metrics_list:
            return None
        return baselines, proposed, metrics_list

    @staticmethod
    def _find_primary_metric(bp_metrics: list) -> str:
        """Return the primary metric name from blueprint, or first available."""
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("primary"):
                return m.get("name", "accuracy")
        if bp_metrics and isinstance(bp_metrics[0], dict):
            return bp_metrics[0].get("name", "accuracy")
        return "accuracy"

    @staticmethod
    def _metric_higher_is_better(metric_name: str, bp_metrics: list) -> bool:
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("name") == metric_name:
                return m.get("higher_is_better", True)
        return True

    _FULL_MODEL_NAMES = frozenset({
        "full", "full model", "full_model", "ours", "proposed", "complete",
    })

    @staticmethod
    def _split_ablation(
        ablation_raw: list, primary_metric: str, higher_is_better: bool = True,
    ) -> tuple:
        """Split ablation entries into (full_result_dict, variants_list).

        Identifies the full model by name first; falls back to best score
        (direction determined by *higher_is_better*).
        Returns ({metric: value}, [{variant_name, metrics: {metric: value}}]).
        """
        entries = []
        for entry in ablation_raw:
            if not isinstance(entry, dict):
                continue
            flat = _flatten_metric_list(entry.get("metrics", []))
            if not flat:
                # metrics might already be a dict
                raw_m = entry.get("metrics", {})
                if isinstance(raw_m, dict):
                    flat = {
                        k: v for k, v in raw_m.items()
                        if isinstance(v, (int, float)) and math.isfinite(v)
                    }
            entries.append({
                "variant_name": entry.get("variant_name", "unknown"),
                "metrics": flat,
                "score": flat.get(primary_metric),
            })

        scored = [e for e in entries if isinstance(e["score"], (int, float))]
        if len(scored) < 2:
            return None, None

        # Find full model: prefer name match, fall back to best score
        full_entry = None
        for e in scored:
            vn = e["variant_name"].lower().strip()
            if vn in AnalysisAgent._FULL_MODEL_NAMES or "full" in vn:
                full_entry = e
                break
        if full_entry is None:
            scored.sort(key=lambda e: e["score"], reverse=higher_is_better)
            full_entry = scored[0]

        full = full_entry["metrics"]
        variants = [
            {"variant_name": e["variant_name"], "metrics": e["metrics"]}
            for e in scored if e is not full_entry
        ]
        return full, variants

    async def close(self) -> None:
        await super().close()
