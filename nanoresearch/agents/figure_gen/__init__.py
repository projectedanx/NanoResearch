"""Figure generation agent — dynamic figure planning + hybrid AI/code charts."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

from ._constants import (  # noqa: F401 — re-exported
    AI_FIGURE_TEMPLATES,
    CHART_EXEC_TIMEOUT,
    FIGURE_PLAN_SYSTEM,
    _run_chart_subprocess,
    _clean_ai_image_caption,
)
from .evidence import _EvidenceMixin
from .ai_figure import _AiFigureMixin
from .code_figure import _CodeFigureMixin
from .trim import _TrimMixin
from .save_figure import _SaveFigureMixin

__all__ = ["FigureAgent"]

logger = logging.getLogger(__name__)


class FigureAgent(
    _EvidenceMixin,
    _AiFigureMixin,
    _CodeFigureMixin,
    _TrimMixin,
    _SaveFigureMixin,
    BaseResearchAgent,
):
    stage = PipelineStage.FIGURE_GEN

    async def run(self, **inputs: Any) -> dict[str, Any]:
        blueprint: dict = inputs.get("experiment_blueprint", {})
        if not blueprint:
            logger.warning("No experiment_blueprint provided; using empty dict")
            blueprint = {}
        ideation_output: dict = inputs.get("ideation_output", {})
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_status: str = inputs.get("experiment_status", "pending")
        survey_blueprint: dict = inputs.get("survey_blueprint", {})
        # ANALYSIS no longer generates figures; FIGURE_GEN owns all 4 figures.
        existing_figures: dict = {}

        # Remove stale generated figure files from previous resume attempts so the
        # exported release folder contains only the active figure plan.
        self._cleanup_stale_generated_figures()

        # Detect survey mode: no experiment blueprint but survey_blueprint exists
        is_survey = bool(survey_blueprint) and not blueprint
        if is_survey:
            self.log("Survey mode detected: using survey-specific figure planning")
        else:
            self.log("Starting figure generation (dynamic planning + hybrid)")

        if experiment_results:
            self.log(f"Using REAL experiment results (status: {experiment_status})")
        else:
            self.log(f"No real experiment results available (status: {experiment_status})")

        # Build context and plan figures based on mode
        if is_survey:
            # Survey context from ideation_output
            theme_clusters = ideation_output.get("theme_clusters", [])
            key_challenges = ideation_output.get("key_challenges", [])
            future_directions = ideation_output.get("future_directions", [])
            survey_size = survey_blueprint.get("survey_size", "standard")

            survey_context = (
                f"Survey topic: {blueprint.get('title', 'Survey Paper')}\n"
                f"Survey size: {survey_size}\n"
                f"Theme clusters ({len(theme_clusters)}): {', '.join(theme_clusters[:10])}\n"
                f"Key challenges ({len(key_challenges)}): {', '.join(key_challenges[:5])}\n"
                f"Future directions ({len(future_directions)}): {', '.join(future_directions[:5])}\n"
            )
            evidence_block = self._build_evidence_block(
                ideation_output, blueprint, experiment_results, experiment_status
            )
            # Plan survey figures
            figure_plan = await self._plan_survey_figures(survey_context, evidence_block, survey_size)
            context = survey_context
            # Survey figure generation uses these with survey-specific defaults
            method_name = "Surveyed Methods"
            baselines = "N/A"
            metrics = "Benchmark metrics"
            ablation_groups = "N/A"
            primary_metric = "Performance"
        else:
            method = blueprint.get("proposed_method") or {}
            method_name = method.get("name", "Proposed Method")
            components = ", ".join(method.get("key_components") or [])
            baselines_list = blueprint.get("baselines") or []
            baselines = ", ".join(b.get("name", "") for b in baselines_list)
            metrics_list = blueprint.get("metrics") or []
            metrics = ", ".join(m.get("name", "") for m in metrics_list)
            ablation_groups = ", ".join(
                a.get("group_name", "") for a in (blueprint.get("ablation_groups") or [])
            )
            primary_metric = next(
                (m.get("name", "") for m in metrics_list if m.get("primary")),
                metrics_list[0].get("name", "Score") if metrics_list else "Score",
            )
            datasets = ", ".join(d.get("name", "") for d in (blueprint.get("datasets") or []))

            context = (
                f"Research title: {blueprint.get('title', '')}\n"
                f"Method: {method_name}\n"
                f"Components: {components}\n"
                f"Datasets: {datasets}\n"
                f"Baselines: {baselines}\n"
                f"Metrics: {metrics}\n"
                f"Ablation groups: {ablation_groups}\n"
                f"Primary metric: {primary_metric}\n"
            )

            # Build evidence block for chart prompts
            evidence_block = self._build_evidence_block(
                ideation_output, blueprint, experiment_results, experiment_status
            )

            # Step 1: LLM plans which figures to generate
            figure_plan = await self._plan_figures(context, evidence_block)
        self.log(f"Figure plan: {len(figure_plan)} figures")

        figure_results = {}

        # Step 2: Generate each planned figure (skip those already from ANALYSIS)
        # Build coroutines for all figures, then run concurrently
        async def _gen_one(fig_spec: dict) -> tuple[str, dict | None]:
            """Generate one figure; returns (fig_key, result_or_None)."""
            if not isinstance(fig_spec, dict) or "fig_key" not in fig_spec:
                logger.warning("Skipping invalid fig_spec: %s", fig_spec)
                return ("", None)
            fig_key = fig_spec["fig_key"]
            fig_type = fig_spec.get("fig_type", "code_chart")
            chart_type = fig_spec.get("chart_type", "grouped_bar")
            description = fig_spec.get("description", "")
            caption = fig_spec.get("caption", description)
            title = fig_spec.get("title", "")

            self.log(f"Generating {fig_key} ({fig_type}/{chart_type})")
            try:
                if fig_type == "ai_image":
                    ai_image_type = fig_spec.get("ai_image_type", "generic")
                    result = await self._generate_ai_figure(
                        context, fig_key, fig_key, description, ai_image_type,
                        caption=caption,
                    )
                    if isinstance(result, dict):
                        result.setdefault("figure_kind", fig_spec.get("figure_kind", "schematic"))
                        result.setdefault("required_backend", "image2")
                        result.setdefault("fig_type", "ai_image")
                        result.setdefault("ai_image_type", ai_image_type)
                else:
                    output_path = str(
                        self.workspace.path / "figures" / f"{fig_key}.png"
                    )
                    chart_prompt = self._build_chart_prompt(
                        chart_type=chart_type,
                        title=title,
                        description=description,
                        method_name=method_name,
                        baselines=baselines,
                        metrics=metrics,
                        ablation_groups=ablation_groups,
                        primary_metric=primary_metric,
                        evidence_block=evidence_block,
                        output_path=output_path,
                        context=context,
                    )
                    result = await self._generate_code_figure(
                        fig_key, output_path, chart_prompt, caption,
                    )
                    if isinstance(result, dict):
                        result.setdefault("figure_kind", fig_spec.get("figure_kind", "data_chart"))
                        result.setdefault("fig_type", "code_chart")
                        result.setdefault("chart_type", chart_type)
                return (fig_key, result)
            except Exception as exc:
                logger.warning(
                    "Figure generation failed for %s: %s",
                    fig_key, exc, exc_info=True,
                )
                self.log(f"Figure failed for {fig_key}, skipping: {exc}")
                return (fig_key, None)

        # Generate all planned figures
        new_specs = [spec for spec in figure_plan if isinstance(spec, dict)]

        results = await asyncio.gather(
            *(_gen_one(spec) for spec in new_specs),
            return_exceptions=False,
        )
        for fig_key, result in results:
            if fig_key and result is not None:
                figure_results[fig_key] = result

        self.log(f"Figure generation complete: {len(figure_results)} new figures")

        # All figures come from FIGURE_GEN; count and type are evidence-driven.
        merged = figure_results
        self.log(f"Total figures: {len(merged)}")

        # Persist output so that resume can reload it
        output = {"figures": merged}
        self.workspace.write_json("drafts/figure_output.json", output)

        return output


    def _cleanup_stale_generated_figures(self) -> None:
        """Delete stale generated figure images before a fresh FIGURE_GEN run."""
        for rel_dir in ("figures", "drafts"):
            directory = self.workspace.path / rel_dir
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                name = path.name.lower()
                if not name.startswith("fig"):
                    continue
                allowed_suffixes = {".png", ".pdf", ".svg"}
                if rel_dir == "figures":
                    allowed_suffixes.update({".py", ".txt", ".json", ".log"})
                if path.suffix.lower() not in allowed_suffixes:
                    continue
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove stale figure %s: %s", path, exc)

    # -----------------------------------------------------------------------
    # Figure planning
    # -----------------------------------------------------------------------

    async def _plan_figures(self, context: str, evidence_block: str) -> list[dict]:
        """Ask LLM to plan which figures to generate."""
        prompt = (
            f"Plan the figures for this research paper.\n\n"
            f"Research context:\n{context}\n\n"
            f"{evidence_block}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. First, identify the research domain (nlp/cv/llm/multimodal/general_ml)\n"
            f"2. Follow the domain-specific figure convention from the system prompt\n"
            f"3. Select 3-6 figures based on available evidence; do not force a fixed split.\n"
            f"   - Include one method/framework ai_image for the Method section.\n"
            f"   - Include result code_chart figures only when the evidence block contains usable numeric data.\n"
            f"   - Prefer distinct result figures when data exists: main results vs measured baselines, ablation, optimization curve, complexity tradeoff, Pareto/frontier, or published baseline context.\n"
            f"4. Use ai_image_type only for conceptual method/architecture/qualitative figures.\n"
            f"5. Every code_chart must use a DIFFERENT chart_type -- NO duplicates.\n"
            f"6. Plan an ablation chart ONLY if the evidence block contains real ablation numbers.\n"
            f"7. Never plan a result chart from missing data or placeholders.\n\n"
            f"Return the figure plan as JSON with 'domain' and 'figures' fields."
        )

        try:
            # Use figure_prompt config (text model), NOT the image generation model.
            figure_prompt_config = self.config.for_stage("figure_prompt")
            result = await self.generate_json(
                FIGURE_PLAN_SYSTEM, prompt, stage_override=figure_prompt_config
            )
            figures = result.get("figures", [])
            if not figures:
                self.log("Figure plan returned empty, using default plan")
                return self._default_figure_plan()
            # Validate each figure spec
            validated = []
            seen_chart_types: set[str] = set()
            for fig in figures:
                if "fig_key" not in fig:
                    continue
                fig.setdefault("fig_type", "code_chart")
                fig.setdefault("chart_type", "grouped_bar")
                fig.setdefault("caption", fig.get("description", ""))
                # Validate ai_image_type for AI figures & clean verbose captions
                if fig["fig_type"] == "ai_image":
                    img_type = fig.get("ai_image_type", "generic")
                    if img_type not in AI_FIGURE_TEMPLATES:
                        logger.warning(
                            "Unknown ai_image_type %r, falling back to 'generic'",
                            img_type,
                        )
                        fig["ai_image_type"] = "generic"
                    # Clean caption: LLM sometimes returns generation-prompt-
                    # length text as the caption.  Keep it short & academic.
                    fig["caption"] = _clean_ai_image_caption(
                        fig["caption"], fig.get("title", ""),
                    )
                # Deduplicate chart_type for code_chart figures
                if fig["fig_type"] == "code_chart":
                    ct = fig["chart_type"]
                    if ct in seen_chart_types:
                        logger.warning(
                            "Duplicate chart_type %r in figure plan, skipping %s",
                            ct, fig.get("fig_key"),
                        )
                        continue
                    seen_chart_types.add(ct)
                validated.append(fig)
            if not validated:
                return self._default_figure_plan()

            validated = self._enforce_method_schematic(validated)
            validated = self._limit_experiment_conceptual_figures(validated)

            has_real_ablation = "--- Ablation Results [source: REAL EXPERIMENT] ---" in evidence_block
            if not has_real_ablation:
                validated = [
                    f for f in validated
                    if "ablation" not in str(f.get("fig_key", "")).lower()
                    and "ablation" not in str(f.get("title", "")).lower()
                    and "ablation" not in str(f.get("description", "")).lower()
                ]

            has_real_results = "=== REAL EXPERIMENT RESULTS [source: REAL EXPERIMENT] ===" in evidence_block
            has_published_context = "PUBLISHED LITERATURE" in evidence_block or "PUBLISHED BASELINE" in evidence_block
            if not has_real_results:
                validated = [
                    f for f in validated
                    if f.get("fig_type") == "ai_image"
                    or (has_published_context and "published" in str(f.get("description", "") + f.get("title", "")).lower())
                ]
            if has_real_results:
                joined = " ".join(
                    str(f.get("fig_key", "")) + " " + str(f.get("title", "")) + " " + str(f.get("description", ""))
                    for f in validated if isinstance(f, dict)
                ).lower()
                if not any(kw in joined for kw in ("complexity", "efficiency", "runtime", "cost")):
                    validated.append({
                        "fig_key": "fig5_complexity_profile",
                        "fig_type": "code_chart",
                        "figure_kind": "data_chart",
                        "chart_type": "horizontal_bar",
                        "title": "Complexity Profile",
                        "description": "Measured selected features, nonzero coefficients, fit time, predict time, and random-forest tree-depth or split-feature metrics from structured experiment artifacts.",
                        "caption": "Complexity profile comparing feature count, coefficient footprint, and runtime proxies using only measured experiment artifacts.",
                    })
            if not validated:
                return self._default_figure_plan(has_real_results=has_real_results)
            return validated[:6]
        except Exception as e:
            logger.warning("Figure planning failed: %s", e, exc_info=True)
            self.log(f"Figure planning failed ({e}), using default plan")
            return self._default_figure_plan()

    def _enforce_method_schematic(self, figures: list[dict]) -> list[dict]:
        """Ensure Figure 1 is an image2-generated method schematic."""
        schematic = {
            "fig_key": "fig_method_schematic",
            "fig_type": "ai_image",
            "figure_kind": "schematic",
            "ai_image_type": "system_overview",
            "chart_type": None,
            "title": "Method Overview",
            "description": (
                "Paper-facing schematic of the proposed method: input data, "
                "preprocessing, core algorithmic components, optimization loop, "
                "and evaluation outputs. Use the actual method and dataset names."
            ),
            "caption": "Overview of the proposed method and evaluation workflow.",
        }
        kept: list[dict] = []
        for fig in figures:
            if not isinstance(fig, dict):
                continue
            key = str(fig.get("fig_key", "")).lower()
            title = str(fig.get("title", "") + " " + fig.get("description", "")).lower()
            is_method_like = any(kw in key or kw in title for kw in (
                "method", "framework", "overview", "architecture", "pipeline", "workflow", "schematic"
            ))
            if is_method_like and fig.get("fig_type") != "code_chart":
                merged = {**schematic, **fig}
                merged["fig_key"] = "fig_method_schematic"
                merged["fig_type"] = "ai_image"
                merged["figure_kind"] = "schematic"
                # Force the paper's Figure 1 through the compact method-schematic
                # prompt path; generic multi-stage prompts tend to generate cropped
                # title-heavy diagrams.
                merged["ai_image_type"] = "system_overview"
                merged["chart_type"] = None
                schematic = merged
                continue
            if key == "fig_method_schematic":
                continue
            kept.append(fig)
        return [schematic] + kept


    def _limit_experiment_conceptual_figures(self, figures: list[dict]) -> list[dict]:
        """Keep one method schematic and reserve remaining slots for result charts."""
        kept: list[dict] = []
        method_seen = False
        for fig in figures:
            if not isinstance(fig, dict):
                continue
            if fig.get("fig_type") == "ai_image":
                key = str(fig.get("fig_key", "")).lower()
                kind = str(fig.get("figure_kind", "")).lower()
                is_method = key == "fig_method_schematic" or kind == "schematic"
                if is_method and not method_seen:
                    kept.append(fig)
                    method_seen = True
                continue
            kept.append(fig)
        if not method_seen:
            return self._enforce_method_schematic(kept)
        return kept

    def _default_figure_plan(self, has_real_results: bool = True) -> list[dict]:
        """Fallback figure plan driven by available evidence."""
        conceptual = [
            {
                "fig_key": "fig_method_schematic",
                "fig_type": "ai_image",
                "figure_kind": "schematic",
                "ai_image_type": "system_overview",
                "chart_type": None,
                "title": "Method Overview",
                "description": "Paper-facing schematic showing the proposed method, key components, optimization loop, and data flow.",
                "caption": "Overview of the proposed method and evaluation workflow.",
            },
        ]
        if not has_real_results:
            return conceptual
        return conceptual + [
            {
                "fig_key": "fig_main_results",
                "fig_type": "code_chart",
                "figure_kind": "data_chart",
                "chart_type": "grouped_bar",
                "title": "Measured Main Results",
                "description": "Measured proposed method and measured baseline results from results/metrics.json.",
                "caption": "Measured performance comparison using only executed experiment outputs.",
            },
            {
                "fig_key": "fig_optimization_complexity",
                "fig_type": "code_chart",
                "figure_kind": "data_chart",
                "chart_type": "line",
                "title": "Optimization and Complexity",
                "description": "Optimization history, complexity metrics, or Pareto tradeoff from structured artifacts.",
                "caption": "Optimization and efficiency evidence from machine-checkable experiment artifacts.",
            },
        ]

    # -----------------------------------------------------------------------
    # Survey figure planning
    # -----------------------------------------------------------------------

    async def _plan_survey_figures(
        self, context: str, evidence_block: str, survey_size: str = "standard"
    ) -> list[dict]:
        """Ask LLM to plan which figures to generate for a survey paper."""
        # Size-based figure count: short=2, standard=3, long=4
        size_fig_counts = {"short": 2, "standard": 3, "long": 4}
        target_figs = size_fig_counts.get(survey_size, 3)

        prompt = (
            f"Plan the figures for this survey paper.\n\n"
            f"Survey context:\n{context}\n\n"
            f"{evidence_block}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Identify the research domain (nlp/cv/llm/multimodal/general_ml)\n"
            f"2. Select EXACTLY {target_figs} figures appropriate for a survey paper:\n"
            f"   - One taxonomy/overview diagram (ai_image, ai_image_type='system_overview')\n"
            f"   - One benchmark comparison chart (code_chart, e.g. grouped_bar or line)\n"
            f"   - Additional figures as appropriate for the survey size\n"
            f"3. For ai_image: use 'system_overview' or 'qualitative_comparison' types\n"
            f"4. For code_chart: use meaningful chart types (grouped_bar, line, heatmap)\n"
            f"5. Every code_chart must use a DIFFERENT chart_type — NO duplicates\n\n"
            f"Return the figure plan as JSON with 'domain' and 'figures' fields."
        )

        try:
            figure_prompt_config = self.config.for_stage("figure_prompt")
            result = await self.generate_json(
                FIGURE_PLAN_SYSTEM, prompt, stage_override=figure_prompt_config
            )
            figures = result.get("figures", [])
            if not figures:
                self.log("Survey figure plan returned empty, using default plan")
                return self._default_survey_figure_plan(survey_size)
            # Validate and normalize
            validated = []
            seen_chart_types: set[str] = set()
            for fig in figures:
                if "fig_key" not in fig:
                    continue
                fig.setdefault("fig_type", "ai_image")
                fig.setdefault("chart_type", None)
                fig.setdefault("caption", fig.get("description", ""))
                if fig["fig_type"] == "ai_image":
                    img_type = fig.get("ai_image_type", "system_overview")
                    if img_type not in AI_FIGURE_TEMPLATES:
                        fig["ai_image_type"] = "system_overview"
                    fig["caption"] = _clean_ai_image_caption(
                        fig["caption"], fig.get("title", ""),
                    )
                elif fig["fig_type"] == "code_chart":
                    ct = fig.get("chart_type", "grouped_bar")
                    if ct in seen_chart_types:
                        logger.warning(
                            "Duplicate chart_type %r in survey figure plan, skipping %s",
                            ct, fig.get("fig_key"),
                        )
                        continue
                    seen_chart_types.add(ct)
                validated.append(fig)
            if not validated:
                return self._default_survey_figure_plan(survey_size)
            return validated[:target_figs]
        except Exception as e:
            logger.warning("Survey figure planning failed: %s", e, exc_info=True)
            self.log(f"Survey figure planning failed ({e}), using default plan")
            return self._default_survey_figure_plan(survey_size)

    def _default_survey_figure_plan(self, survey_size: str = "standard") -> list[dict]:
        """Fallback figure plan for survey papers."""
        # Size-based figures: short=2, standard=3, long=4
        if survey_size == "short":
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
            ]
        elif survey_size == "long":
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Comprehensive taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
                {
                    "fig_key": "fig3_evolution",
                    "fig_type": "ai_image",
                    "ai_image_type": "qualitative_comparison",
                    "chart_type": None,
                    "title": "Field Evolution",
                    "description": "Timeline showing how the field evolved over time.",
                    "caption": "Evolution of the field over time, highlighting key developments.",
                },
                {
                    "fig_key": "fig4_method_comparison",
                    "fig_type": "code_chart",
                    "chart_type": "heatmap",
                    "title": "Method Comparison Matrix",
                    "description": "Comparison matrix showing methods vs characteristics.",
                    "caption": "Method comparison matrix showing capabilities across different dimensions.",
                },
            ]
        else:  # standard (default)
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
                {
                    "fig_key": "fig3_evolution",
                    "fig_type": "ai_image",
                    "ai_image_type": "qualitative_comparison",
                    "chart_type": None,
                    "title": "Field Evolution",
                    "description": "Timeline showing how the field evolved over time.",
                    "caption": "Evolution of the field over time, highlighting key developments.",
                },
            ]

    # -----------------------------------------------------------------------
    # Chart prompt builder
    # -----------------------------------------------------------------------
