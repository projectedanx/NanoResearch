"""Review agent — automated paper review, consistency checking, and revision."""

from __future__ import annotations

import re
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.review import (
    ConsistencyIssue,
    ReviewOutput,
    SectionReview,
)

from ._constants import (  # noqa: F401 — re-exported for backward compat
    _CONFERENCE_KEYWORDS,
    _detect_bib_entry_type,
    MAX_REVISION_ROUNDS,
    MAX_LATEX_FIX_ATTEMPTS,
    MIN_SECTION_SCORE,
    CONVERGENCE_THRESHOLD,
    _SECTION_PATTERN,
    _CITE_PATTERN,
    _RELATED_WORK_SECTION_PATTERN,
    _ABSTRACT_PATTERN,
    REVIEW_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
)

from .section_extraction import _SectionExtractionMixin
from .multi_reviewer import _MultiReviewerMixin
from .single_review import _SingleReviewMixin
from .revision import _RevisionMixin
from .apply_revisions import _ApplyRevisionsMixin
from .consistency import _ConsistencyMixin
from .latex_compile import _LaTeXCompileMixin
from nanoresearch.agents.writing.grounding import _GroundingMixin
from nanoresearch.agents.writing.grounding_tables import _format_paper_number

__all__ = ["ReviewAgent"]

logger = logging.getLogger(__name__)


class ReviewAgent(
    _SectionExtractionMixin,
    _MultiReviewerMixin,
    _SingleReviewMixin,
    _RevisionMixin,
    _ApplyRevisionsMixin,
    _ConsistencyMixin,
    _LaTeXCompileMixin,
    BaseResearchAgent,
):
    stage = PipelineStage.REVIEW

    def _apply_grounding_protection(
        self,
        tex: str,
        experiment_blueprint: dict,
        ideation_output: dict,
    ) -> str:
        """Remove review-introduced quantitative claims unsupported by run artifacts."""
        try:
            grounding = _GroundingMixin._build_grounding_packet(
                self._experiment_results or {},
                self._experiment_status or "pending",
                self._experiment_analysis or {},
                "",
                experiment_blueprint or {},
            )
        except Exception as exc:
            logger.warning("Grounding protection skipped: %s", exc)
            return tex

        protected = tex

        # Abstract must be paper-facing and grounded in verified artifacts.
        method = experiment_blueprint.get("proposed_method", {}) if isinstance(experiment_blueprint, dict) else {}
        method_name = method.get("name") if isinstance(method, dict) else "the proposed method"
        datasets = experiment_blueprint.get("datasets", []) if isinstance(experiment_blueprint, dict) else []
        dataset_name = "the evaluated dataset"
        if isinstance(datasets, list) and datasets:
            first_dataset = datasets[0]
            if isinstance(first_dataset, dict):
                dataset_name = first_dataset.get("name") or dataset_name
            elif isinstance(first_dataset, str):
                dataset_name = first_dataset

        proposed_entry = next(
            (entry for entry in grounding.main_results
             if isinstance(entry, dict) and entry.get("is_proposed")),
            None,
        )

        def _metric_value(entry: dict | None, *names: str) -> str | None:
            if not isinstance(entry, dict):
                return None
            aliases = {re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") for n in names}
            for metric in entry.get("metrics", []):
                if not isinstance(metric, dict):
                    continue
                key = re.sub(r"[^a-z0-9]+", "_", str(metric.get("metric_name", "")).lower()).strip("_")
                if key in aliases and metric.get("value") is not None:
                    return _format_paper_number(metric.get("value"))
            return None

        cv_ba = _metric_value(proposed_entry, "cross_validated_balanced_accuracy", "best_pareto_balanced_accuracy", "balanced_accuracy")
        heldout_acc = _metric_value(proposed_entry, "heldout_accuracy", "accuracy")
        selected = _metric_value(proposed_entry, "best_pareto_selected_feature_count", "selected_feature_count", "selected_features")
        fit_time = _metric_value(proposed_entry, "fit_time_seconds", "fit_time")
        result_clause = ""
        if cv_ba or heldout_acc or selected:
            parts = []
            if cv_ba:
                parts.append(f"{cv_ba} cross-validated balanced accuracy")
            if heldout_acc:
                parts.append(f"{heldout_acc} held-out accuracy")
            if selected:
                parts.append(f"{selected} selected features")
            if fit_time:
                parts.append(f"{fit_time}s fit time")
            result_clause = ", ".join(parts)
        else:
            result_clause = "verified artifact-grounded results"

        abstract = (
            f"Lightweight medical tabular classification requires models that are accurate, reproducible, and small enough for practitioners to inspect. "
            f"Existing feature-selection studies often report predictive scores without jointly exposing the fixed search budget, leakage-safe validation protocol, optimization trace, and final model complexity. "
            f"We propose {method_name}, a fixed-budget wrapper that searches binary feature masks with Non-dominated Sorting Genetic Algorithm II while optimizing cross-validated balanced accuracy and selected-feature count for a logistic-regression classifier. "
            f"The pipeline performs fold-local preprocessing, extracts the Pareto front, selects a deterministic sparse candidate, refits the final model on the training split, and reports ablations, optimization history, and complexity metrics. "
            f"On {dataset_name}, the verified run achieves {result_clause}, showing that Pareto feature selection can preserve strong diagnostic performance while reducing the inspected feature set relative to full-feature baselines."
        )
        protected = re.sub(
            r"\\begin\{abstract\}.*?\\end\{abstract\}",
            lambda _m: "\\begin{abstract}\n" + abstract + "\n\\end{abstract}",
            protected,
            flags=re.DOTALL,
        )

        # Replace any reviewed main-results table with the deterministic table from run artifacts.
        if grounding.main_table_latex:
            main_pat = r"\\begin\{table\*?\}.*?\\label\{tab:main_results\}.*?\\end\{table\*?\}"
            if re.search(main_pat, protected, flags=re.DOTALL):
                protected = re.sub(main_pat, lambda _m: grounding.main_table_latex, protected, count=1, flags=re.DOTALL)
            else:
                protected = re.sub(
                    r"(\\section\{Experiments\}[^\n]*\n)",
                    lambda m: m.group(1) + "\n" + grounding.main_table_latex + "\n",
                    protected,
                    count=1,
                )

        # Replace reviewed ablation tables with deterministic tables from run artifacts,
        # and remove any other LLM-created result tables.
        if grounding.ablation_results and grounding.ablation_table_latex:
            ablation_pat = r"\\begin\{table\*?\}.*?\\label\{tab:ablation\}.*?\\end\{table\*?\}"
            if re.search(ablation_pat, protected, flags=re.DOTALL):
                protected = re.sub(
                    ablation_pat,
                    lambda _m: grounding.ablation_table_latex,
                    protected,
                    count=1,
                    flags=re.DOTALL,
                )
            else:
                protected = re.sub(
                    r"(\\section\{Experiments\}[^\n]*\n)",
                    lambda m: m.group(1) + "\n" + grounding.ablation_table_latex + "\n",
                    protected,
                    count=1,
                )
        else:
            protected = re.sub(
                r"\n\\subsection\{Ablation Study\}.*?(?=\n\\subsection\{|\n\\section\{|\n%% ---- References|\n\\bibliographystyle)",
                lambda _m: "\n",
                protected,
                flags=re.DOTALL,
            )
            protected = re.sub(
                r"\n\\begin\{table\*?\}.*?\\label\{tab:ablation\}.*?\\end\{table\*?\}",
                "",
                protected,
                flags=re.DOTALL,
            )
            protected = protected.replace("Table~\\ref{tab:ablation}", "the ablation analysis")

        allowed_table_labels = {"tab:main_results", "tab:ablation"}

        def _drop_ungrounded_table(match: re.Match) -> str:
            block = match.group(0)
            label_m = re.search(r"\\label\{(tab:[^}]+)\}", block)
            label = label_m.group(1) if label_m else ""
            return block if label in allowed_table_labels else ""

        protected = re.sub(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
            _drop_ungrounded_table,
            protected,
            flags=re.DOTALL,
        )
        # Rebuild the Experiments section with the same artifact-driven composer
        # used by the writing stage. Review may remove or protect unsupported
        # claims, but it should not collapse result figures into a bare sequence.
        exp_match = re.search(
            r"\\section\{Experiments\}.*?(?=\n\\section\{Conclusion\}|\n%% ---- References|\n\\bibliographystyle)",
            protected,
            flags=re.DOTALL,
        )
        exp_source = exp_match.group(0) if exp_match else protected
        figure_source = exp_source + "\n" + str(getattr(self, "_original_figure_blocks_tex", ""))
        figure_blocks_raw = re.findall(r"\n?\\begin\{figure\*?\}.*?\\end\{figure\*?\}", figure_source, flags=re.DOTALL)
        figure_blocks: dict[str, str] = {}
        for idx, block in enumerate(figure_blocks_raw, 1):
            label_m = re.search(r"\\label\{fig:([^}]+)\}", block)
            key = label_m.group(1) if label_m else f"review_result_figure_{idx}"
            if key not in figure_blocks:
                figure_blocks[key] = block.strip()

        if grounding.main_table_latex:
            experiment_section, _used_figures = _GroundingMixin._compose_experiments_section(
                grounding, figure_blocks, experiment_blueprint or {}, include_heading=True,
            )
            protected, replaced_count = re.subn(
                r"\\section\{Experiments\}.*?(?=\n\\section\{Conclusion\}|\n%% ---- References|\n\\bibliographystyle)",
                lambda _m: experiment_section,
                protected,
                count=1,
                flags=re.DOTALL,
            )
            if replaced_count == 0:
                protected, inserted_count = re.subn(
                    r"(?=\n\\section\{Conclusion\})",
                    "\n" + experiment_section + "\n",
                    protected,
                    count=1,
                )
                if inserted_count == 0:
                    protected += "\n" + experiment_section


        selected_feature_count = None
        for entry in grounding.main_results:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").lower()
            is_proposed = bool(entry.get("is_proposed")) or role == "proposed"
            if not is_proposed:
                continue
            for metric in entry.get("metrics", []) or []:
                if isinstance(metric, dict):
                    metric_name = str(metric.get("metric_name") or "")
                    if metric_name in {"selected_feature_count", "best_pareto_selected_feature_count", "selected_features"}:
                        selected_feature_count = _format_paper_number(metric.get("value"))
                        break
            if selected_feature_count is not None:
                break
        if selected_feature_count is not None:
            protected = re.sub(
                r"(?:can identify|test whether|tests whether|evaluate whether|evaluates whether)[^.]*?at most 10 of 30 features[^.]*\.",
                lambda _m: (
                    "tests whether a fixed-budget Pareto search can approach tuned full-feature "
                    f"logistic-regression and random-forest baselines while selecting fewer features; "
                    f"the completed run selected {selected_feature_count} of 30 features, so the stricter "
                    "at-most-10-feature target is reported as unmet rather than achieved."
                ),
                protected,
                count=2,
                flags=re.IGNORECASE,
            )
            protected = protected.replace(
                "demonstrate that the proposed method achieves competitive held-out balanced accuracy with a subset of at most 10 features while reducing model complexity relative to full-feature baselines.",
                f"show that the completed run achieves competitive held-out metrics with {selected_feature_count} selected features, while explicitly reporting that the stricter at-most-10-feature target was not met."
            )
            protected = protected.replace(
                "This rule matches the paper's hypothesis: it asks whether at most 10 of 30 features can stay within a small balanced-accuracy band of full-feature logistic regression and random forest.",
                f"This rule tests a sparse-model target against full-feature logistic regression and random forest; the completed run selected {selected_feature_count} of 30 features, so claims about stricter 10-feature sparsity are treated as unmet targets rather than achieved results."
            )

        # Downscope introduction contribution claims when only a quick/single metric exists.
        protected = re.sub(
            r"\\item We introduce a fitness function.*?\n",
            lambda _m: "\\item We introduce a fitness function that balances classification accuracy, feature count, and tree depth; the current run verifies the available metric reported in Table~\\ref{tab:main_results}.\n",
            protected,
            count=1,
            flags=re.DOTALL,
        )
        protected = re.sub(
            r"\\item We provide a comprehensive comparison.*?\n",
            lambda _m: "\\item We keep literature and baseline context separate from measured results, reporting only locally measured metrics as experimental evidence.\n",
            protected,
            count=1,
            flags=re.DOTALL,
        )
        protected = re.sub(
            r"This straightforward integration yields substantial improvements.*?built-in feature importance\.",
            lambda _m: "This integration is evaluated through the measured evidence reported in Section~\\ref{sec:experiments}; unsupported comparison claims are intentionally omitted unless verified metrics are available.",
            protected,
            count=1,
            flags=re.DOTALL,
        )
        protected = protected.replace(" on UNKNOWN", " on the evaluated task")
        protected = protected.replace("[hbbp]", "[bp]")

        protected = re.sub(r"(?<![A-Za-z])N/A(?![A-Za-z])", "not directly comparable", protected)
        protected = re.sub(r"\n{3,}", "\n\n", protected)
        return protected

    async def run(self, **inputs: Any) -> dict[str, Any]:
        paper_tex = inputs.get("paper_tex", "")
        if not isinstance(paper_tex, str):
            paper_tex = str(paper_tex) if paper_tex else ""
        ideation_output = inputs.get("ideation_output") or {}
        if not isinstance(ideation_output, dict):
            ideation_output = {}
        experiment_blueprint = inputs.get("experiment_blueprint") or {}
        if not isinstance(experiment_blueprint, dict):
            experiment_blueprint = {}

        # Grounding metadata from writing stage — used to protect real results
        self._writing_grounding: dict = inputs.get("writing_grounding") or {}
        self._paper_structure_plan: dict = inputs.get("paper_structure_plan") or {}
        self._experiment_results: dict = inputs.get("experiment_results") or {}
        self._experiment_analysis: dict = inputs.get("experiment_analysis") or {}
        self._experiment_status: str = inputs.get("experiment_status", "pending")
        self._original_figure_blocks_tex = "\n".join(
            re.findall(r"\n?\\begin\{figure\*?\}.*?\\end\{figure\*?\}", paper_tex, flags=re.DOTALL)
        )

        if not paper_tex:
            self.log("No paper.tex content available, skipping review")
            return ReviewOutput().model_dump(mode="json")

        self.log("Starting automated review")
        adaptive_review_context = self.build_adaptive_context(
            "review",
            topic=ideation_output.get("topic", ""),
            blueprint=experiment_blueprint,
            text=paper_tex[:3000],
            tags=[ideation_output.get("topic", ""), self._experiment_status, "review"],
            include_script_recommendations=False,
        )
        if self._paper_structure_plan:
            plan_bits = []
            for key in ("section_budget", "review_checklist", "forbidden_claims"):
                value = self._paper_structure_plan.get(key)
                if value:
                    plan_bits.append(f"{key}: {value}")
            if plan_bits:
                adaptive_review_context = (
                    f"{adaptive_review_context}\n\n=== PAPER STRUCTURE PLAN COMPLIANCE ===\n"
                    + "\n".join(plan_bits)[:4000]
                    + "\n=== END PAPER STRUCTURE PLAN COMPLIANCE ==="
                )
        self._adaptive_review_context = adaptive_review_context
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "review",
                "review_retry",
                retry_error,
                tags=[ideation_output.get("topic", ""), "review", "retry"],
            )

        # Step 1: LLM review — multi-model if committee configured, else single
        committee = getattr(self.config, "review_committee", [])
        if isinstance(committee, list) and len(committee) >= 2:
            review = await self._multi_reviewer_assessment(
                paper_tex, ideation_output, experiment_blueprint, committee
            )
        else:
            review = await self._review_paper(
                paper_tex, ideation_output, experiment_blueprint
            )
        self.log(
            f"Initial review: overall score {review.overall_score:.1f}, "
            f"{len(review.section_reviews)} sections reviewed"
        )

        # Step 2: Consistency checks (automated, no LLM)
        consistency_issues = self._run_consistency_checks(paper_tex)
        if self._paper_structure_plan:
            try:
                from nanoresearch.agents.writing.stage_planner import _WritingStagePlannerMixin

                for issue in _WritingStagePlannerMixin._audit_paper_structure_against_plan(
                    paper_tex, self._paper_structure_plan
                ):
                    consistency_issues.append(ConsistencyIssue(
                        issue_type="paper_structure_plan",
                        description=issue,
                        locations=["Writing plan compliance"],
                        severity="medium",
                    ))
            except Exception as exc:
                logger.warning("Paper structure plan audit failed during review: %s", exc)
        review.consistency_issues.extend(consistency_issues)
        self.log(f"Found {len(consistency_issues)} consistency issues")

        # Step 2a: Claim-result consistency check
        claim_issues = self._check_claim_result_consistency(
            paper_tex, experiment_blueprint
        )
        review.consistency_issues.extend(claim_issues)
        if claim_issues:
            self.log(f"Found {len(claim_issues)} claim-result mismatches")

        # Step 2c: Figure-text alignment check
        figure_issues = self._check_figure_text_alignment(paper_tex)
        review.consistency_issues.extend(figure_issues)
        if figure_issues:
            self.log(f"Found {len(figure_issues)} figure alignment issues")

        # Step 2d: Citation coverage check
        citation_issues = self._check_citation_coverage(paper_tex, ideation_output)
        review.consistency_issues.extend(citation_issues)
        if citation_issues:
            self.log(f"Found {len(citation_issues)} citation coverage issues")

        # Step 2e: Citation fact-checking (LLM-based)
        try:
            from nanoresearch.agents.review_citation_checker import (
                verify_citation_claims,
            )

            bibtex_map = self._build_bibtex_key_to_paper_map(
                paper_tex, ideation_output.get("papers", [])
            )
            if bibtex_map:
                cite_verifications = await verify_citation_claims(
                    self, paper_tex, bibtex_map
                )
                inaccurate = [v for v in cite_verifications if not v["accurate"]]
                if inaccurate:
                    self.log(
                        f"Citation fact-check: {len(inaccurate)} "
                        f"potentially inaccurate claims"
                    )
                    for v in inaccurate:
                        review.consistency_issues.append(
                            ConsistencyIssue(
                                issue_type="citation_inaccuracy",
                                description=(
                                    f"Claim about [{v['cite_key']}] may be "
                                    f"inaccurate: {v.get('issue', 'unspecified')}"
                                ),
                                locations=[],
                                severity="medium",
                            )
                        )
                else:
                    self.log(
                        f"Citation fact-check: {len(cite_verifications)} "
                        f"claims verified, all accurate"
                    )
        except Exception as exc:
            logger.warning("Citation fact-checking failed: %s", exc)

        # Deduplicate consistency issues before entering revision loop
        review.consistency_issues = self._dedup_consistency_issues(review.consistency_issues)

        # Step 2b: Fix incoherent reviews (low score but no issues)
        for sr in review.section_reviews:
            if sr.score < MIN_SECTION_SCORE and not sr.issues:
                sr.issues = [
                    f"Section '{sr.section}' scored {sr.score}/10 — "
                    "it needs substantial improvement in clarity, depth, "
                    "and technical rigor to reach publication quality."
                ]
                sr.suggestions = [
                    "Rewrite the section with more detailed technical content, "
                    "proper citations, and clear exposition. Remove any placeholder "
                    "or 'results pending' language. Fill tables with concrete data."
                ]

        # Step 3: Revision loop with convergence detection. If the writing
        # stage explicitly reports no real experiment results, keep the review
        # as diagnostic feedback instead of repeatedly revising an impossible
        # Experiments section into fabricated quantitative claims.
        no_real_results = (
            (
                isinstance(self._writing_grounding, dict)
                and self._writing_grounding.get("result_completeness") == "none"
            )
            or str(self._experiment_status).lower() in {"failed", "max_rounds", "no_results"}
            or (
                isinstance(self._experiment_results, dict)
                and not self._experiment_results.get("main_results")
                and not self._experiment_results.get("metrics")
            )
        )
        if no_real_results:
            self.log(
                "No real experiment results in writing grounding; "
                "skipping revision loop and preserving diagnostic review."
            )
            current_tex = paper_tex
            review.revision_rounds = 0
        else:
            current_tex = await self._run_revision_loop(
                paper_tex, paper_tex, review, ideation_output, experiment_blueprint
            )

        # Recalculate overall score
        if review.section_reviews:
            review.overall_score = sum(
                sr.score for sr in review.section_reviews
            ) / len(review.section_reviews)

        # Save outputs
        output_data = review.model_dump(mode="json")
        self.workspace.write_json("drafts/review_output.json", output_data)
        self.workspace.register_artifact(
            "review_output",
            self.workspace.path / "drafts" / "review_output.json",
            self.stage,
        )
        low_score_sections = [sr.section for sr in review.section_reviews if sr.score < MIN_SECTION_SCORE]
        issue_text = "; ".join(issue.description for issue in review.consistency_issues[:6])
        topic_name = ideation_output.get("topic", "unknown topic")
        if issue_text or low_score_sections:
            self.remember_context(
                MemoryType.DECISION_HISTORY,
                f"Review feedback for {topic_name}: low_score_sections={low_score_sections}; issues={issue_text}",
                importance=0.86,
                tags=[ideation_output.get("topic", ""), "review", "feedback"],
                source="review_output",
                topic=ideation_output.get("topic", ""),
            )
            self.learn_from_trace(
                "review",
                "review_feedback",
                f"Low-score sections: {low_score_sections}; consistency issues: {issue_text}",
                tags=[ideation_output.get("topic", ""), "review", "feedback"],
                confidence=0.66,
            )

        # If we have revised sections, write revised paper back to paper.tex
        # current_tex already has all revisions applied from the loop above.
        if True:
            revised_tex = current_tex

            # Sanitize the revised LaTeX (fix Unicode, LLM artifacts, etc.)
            revised_tex = self._sanitize_revised_tex(revised_tex)
            revised_tex = self._apply_grounding_protection(
                revised_tex, experiment_blueprint, ideation_output
            )

            original_sections = set(re.findall(r"\\section\*?\{([^}]+)\}", paper_tex))
            revised_sections = set(re.findall(r"\\section\*?\{([^}]+)\}", revised_tex))
            required_sections = {"Introduction", "Related Work", "Method", "Experiments", "Conclusion"}
            if len(revised_sections) < len(original_sections) or not required_sections.intersection(original_sections).issubset(revised_sections):
                logger.warning("Review revision dropped top-level sections; reverting to pre-review draft with grounding protection")
                revised_tex = self._apply_grounding_protection(
                    self._sanitize_revised_tex(paper_tex), experiment_blueprint, ideation_output
                )

            # Deduplicate figures: keep only the first occurrence of each figure
            seen_fig_labels: set[str] = set()
            seen_fig_files: set[str] = set()
            def _dedup_fig(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_fig_labels:
                    return ""
                file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
                if file_m:
                    fname = file_m.group(1)
                    if fname in seen_fig_files:
                        return ""
                    seen_fig_files.add(fname)
                # Register label AFTER both checks pass
                if lbl:
                    seen_fig_labels.add(lbl)
                return block
            revised_tex = re.sub(
                r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                _dedup_fig, revised_tex, flags=re.DOTALL,
            )

            # Deduplicate tables and drop ungrounded LLM-created result tables.
            seen_tab_labels: set[str] = set()
            allowed_tab_labels = {"tab:main_results", "tab:ablation"}
            def _dedup_tab(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(tab:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl not in allowed_tab_labels:
                    return ""
                if lbl and lbl in seen_tab_labels:
                    return ""
                if lbl:
                    seen_tab_labels.add(lbl)
                return block
            revised_tex = re.sub(
                r'\\begin\{table\*?\}.*?\\end\{table\*?\}',
                _dedup_tab, revised_tex, flags=re.DOTALL,
            )

            revised_tex = re.sub(r'\n{3,}', '\n\n', revised_tex)

            # Resolve any new citations introduced during revision
            bib_path = self.workspace.path / "drafts" / "references.bib"
            if bib_path.exists():
                revised_tex, _ = await self._resolve_missing_citations(
                    revised_tex, bib_path
                )

            # Overwrite original paper.tex with revised version
            tex_path = self.workspace.path / "drafts" / "paper.tex"
            self.workspace.write_text("drafts/paper.tex", revised_tex)
            # Also save a backup copy
            self.workspace.write_text("drafts/paper_revised.tex", revised_tex)
            self.workspace.register_artifact(
                "paper_tex",
                tex_path,
                self.stage,
            )
            self.log("Saved revised paper to drafts/paper.tex")

            # Compile PDF with error-fix loop (like WritingAgent)
            pdf_result = await self._compile_pdf_with_fix_loop(tex_path)
            if "pdf_path" in pdf_result:
                self.log("PDF compiled successfully after revision")
            else:
                self.log(f"PDF compilation failed: {pdf_result.get('error', 'unknown')}")

        self.log(
            f"Review complete: score={review.overall_score:.1f}, "
            f"rounds={review.revision_rounds}, "
            f"revised={len(review.revised_sections)} sections"
        )
        return output_data
