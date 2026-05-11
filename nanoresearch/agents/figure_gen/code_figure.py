"""Code-based chart generation (matplotlib) mixin."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

from ._constants import (
    CHART_CODE_SYSTEM,
    CHART_EXEC_TIMEOUT,
    MAX_CODE_CHART_RETRIES,
    MAX_FIG_ASPECT_RATIO,
    MAX_FIG_HEIGHT_PX,
    _FIGURE_CODE_PREAMBLE,
    _run_chart_subprocess,
)

logger = logging.getLogger(__name__)


class _CodeFigureMixin:
    """Mixin — matplotlib code-based chart generation."""

    async def _generate_code_figure(
        self,
        fig_key: str,
        output_path: str,
        user_prompt: str,
        caption: str,
    ) -> dict[str, Any]:
        """Have LLM generate plotting code, then execute it to create the chart.

        Retries up to MAX_CODE_CHART_RETRIES times, feeding error messages
        back to the LLM so it can fix matplotlib API issues, missing imports, etc.
        """
        filename_stem = fig_key
        figure_code_config = self.config.for_stage("figure_code")
        self.log(f"  {fig_key} chart code model={figure_code_config.model}")
        png_path = Path(output_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = ""
        prev_error = ""

        for attempt in range(MAX_CODE_CHART_RETRIES):
            # Early-exit if the same error repeats (LLM can't fix it)
            if attempt >= 2 and last_error and last_error == prev_error:
                self.log(f"  {fig_key} same error repeated — stopping retry loop")
                break
            prev_error = last_error

            # Build prompt — on retry, include the error feedback
            current_prompt = user_prompt
            if last_error:
                current_prompt += (
                    f"\n\n=== PREVIOUS ATTEMPT FAILED (attempt {attempt}) ===\n"
                    f"Error:\n{last_error[:1500]}\n\n"
                    f"Common fixes:\n"
                    f"- 'capthick' does NOT exist in matplotlib — remove it entirely\n"
                    f"- Check that all kwargs are valid for your matplotlib version\n"
                    f"- Ensure the output path is exactly: {output_path}\n"
                    f"- Use fig.tight_layout() before saving\n"
                    f"=== FIX THE ERROR AND REGENERATE THE COMPLETE CODE ==="
                )

            # Step 1: LLM generates the plotting script
            try:
                code = await self._dispatcher.generate(
                    figure_code_config, CHART_CODE_SYSTEM, current_prompt
                )
            except Exception as e:
                last_error = f"LLM generation error: {e}"
                self.log(f"  {fig_key} attempt {attempt + 1}/{MAX_CODE_CHART_RETRIES} LLM failed: {e}")
                continue

            code = code.strip()
            # Strip markdown fences if present
            if code.startswith("```"):
                lines = code.split("\n")
                lines = [l for l in lines[1:] if not l.strip().startswith("```")]
                code = "\n".join(lines)

            # Inject preamble: enforce sane rcParams in the subprocess
            # Strip any imports the LLM wrote that conflict with the preamble
            # (matplotlib, numpy, seaborn, ticker are all provided by preamble)
            code = re.sub(
                r"^import matplotlib(?:\.\w+)? as .*$|"
                r"^import matplotlib$|"
                r"^from matplotlib(?:\.\w+)? import .*$|"
                r"^matplotlib\.use\(.*\)$|"
                r"^mpl\.use\(.*\)$|"
                r"^import matplotlib\.pyplot as plt$|"
                r"^import numpy as np$|"
                r"^import seaborn as sns$",
                "", code, flags=re.MULTILINE,
            )
            code = _FIGURE_CODE_PREAMBLE + code

            # Save the generated code for debugging/reproducibility
            code_path = self.workspace.write_text(
                f"figures/{filename_stem}_plot.py", code
            )
            self.log(f"  {fig_key} attempt {attempt + 1} code generated ({len(code)} chars)")

            # Step 2: Execute the plotting script
            try:
                loop = asyncio.get_running_loop()
                python_exe = self._resolve_experiment_python()
                result = await loop.run_in_executor(
                    None,
                    partial(
                        _run_chart_subprocess,
                        [python_exe, str(code_path)],
                        timeout=CHART_EXEC_TIMEOUT,
                        cwd=str(self.workspace.path),
                    ),
                )
                if result["returncode"] != 0:
                    last_error = result["stderr"][:1500]
                    self.log(f"  {fig_key} attempt {attempt + 1} execution failed: {last_error[:300]}")
                    self.workspace.write_text(
                        f"logs/{filename_stem}_error.log",
                        f"STDOUT:\n{result['stdout']}\n\nSTDERR:\n{result['stderr']}",
                    )
                    continue
            except subprocess.TimeoutExpired:
                last_error = f"Execution timed out after {CHART_EXEC_TIMEOUT}s"
                self.log(f"  {fig_key} attempt {attempt + 1} timed out")
                continue
            except Exception as exc:
                last_error = str(exc)
                self.log(f"  {fig_key} attempt {attempt + 1} error: {exc}")
                continue

            # Step 3: Verify PNG was created
            # LLMs often ignore absolute output_path and save to relative
            # path instead.  Search likely locations before giving up.
            if not png_path.exists():
                _ws = Path(self.workspace.path)
                # LLMs often ignore the absolute output_path and use a
                # bare filename in plt.savefig().  With cwd=workspace the
                # PNG lands in the workspace root instead of figures/.
                _alt_candidates = [
                    _ws / f"{fig_key}.png",                   # cwd-relative (most common)
                    _ws / "experiment" / f"{fig_key}.png",     # saved in experiment dir
                    _ws / "experiment" / "results" / f"{fig_key}.png",
                ]
                _found_alt = None
                for _alt in _alt_candidates:
                    if _alt.exists() and _alt != png_path:
                        _found_alt = _alt
                        break
                if _found_alt:
                    import shutil as _shutil
                    _shutil.move(str(_found_alt), str(png_path))
                    self.log(
                        f"  {fig_key} attempt {attempt + 1}: PNG found at "
                        f"{_found_alt.name}, moved to figures/"
                    )
                    # Also move companion PDF if it exists
                    _alt_pdf = _found_alt.with_suffix(".pdf")
                    if _alt_pdf.exists():
                        _shutil.move(
                            str(_alt_pdf),
                            str(png_path.with_suffix(".pdf")),
                        )
                else:
                    last_error = (
                        f"Code ran successfully but PNG not generated at "
                        f"{output_path}. IMPORTANT: You MUST use this exact "
                        f"output path in plt.savefig()."
                    )
                    self.log(f"  {fig_key} attempt {attempt + 1}: {last_error}")
                    continue

            # Step 3b: Validate image dimensions — reject absurd sizes
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(png_path) as _img:
                    _w, _h = _img.size
                self.log(f"  {fig_key} output size: {_w}x{_h}")
                aspect = _h / max(_w, 1)
                if _h > MAX_FIG_HEIGHT_PX and aspect > MAX_FIG_ASPECT_RATIO:
                    last_error = (
                        f"Figure too tall: {_w}x{_h} pixels "
                        f"(aspect {aspect:.1f} > {MAX_FIG_ASPECT_RATIO}). "
                        f"Use a smaller figsize like (7, 4.3) or (7, 5) "
                        f"and call fig.tight_layout(). "
                        f"Do NOT use figsize with height > 8 inches."
                    )
                    self.log(f"  {fig_key} attempt {attempt + 1} rejected: {last_error}")
                    png_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass  # PIL not available or file invalid — let it through

            self.log(f"  {fig_key} saved (attempt {attempt + 1})")
            return await self._save_figure_files(fig_key, filename_stem, caption,
                                                 png_path.read_bytes(), already_saved=True,
                                                 code_generated=True)

        # All retries exhausted — use fallback placeholder
        self.log(f"  {fig_key} all {MAX_CODE_CHART_RETRIES} attempts failed, using fallback")
        result = await self._generate_fallback_chart(fig_key, filename_stem, caption, user_prompt=user_prompt, last_error=last_error)
        result["is_fallback"] = True
        return result

    async def _generate_fallback_chart(
        self, fig_key: str, filename_stem: str, caption: str,
        user_prompt: str = "", last_error: str = "",
    ) -> dict[str, Any]:
        """Fallback for failed figure generation.

        Data charts are marked failed rather than filled with made-up values.
        Conceptual figures get a deterministic schematic so the paper still has
        a method/architecture diagram when the image API is unavailable.
        """
        key_l = fig_key.lower()
        diagram_keywords = (
            "framework", "overview", "architecture", "pipeline", "model",
            "workflow", "taxonomy", "qualitative", "example", "diagram",
        )
        if any(kw in key_l for kw in diagram_keywords):
            png_path = self.workspace.path / "figures" / f"{filename_stem}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

                prompt_text = user_prompt or caption or fig_key

                def _pick(patterns, default):
                    for pat in patterns:
                        m = re.search(pat, prompt_text, re.IGNORECASE)
                        if m:
                            value = re.sub(r"[_\s]+", " ", m.group(1)).strip(" .,:;`'\"")
                            if value:
                                return value[:32]
                    return default

                dataset = _pick([r"dataset[s]?:\s*([^\n;]+)", r"on\s+([A-Za-z0-9_ -]*(?:cancer|mnist|cifar|uci|sklearn)[A-Za-z0-9_ -]*)"], "Dataset")
                method = _pick([r"proposed method[:\s]+([^\n;]+)", r"method[:\s]+([^\n;]+)", r"propose\s+([^\n.;]+)"], "Proposed method")
                objective = _pick([r"objective[s]?:\s*([^\n;]+)", r"optimi[sz]es?\s+([^\n.;]+)"], "Training objective")
                labels = [dataset, "Preprocess", method, objective, "Evaluation"]
                colors = ["#E8F1F8", "#D9EAD3", "#FCE5CD", "#D9E2F3", "#EADCF8"]
                fig, ax = plt.subplots(figsize=(7.0, 2.8))
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 3)
                ax.axis("off")
                for i, (label, color) in enumerate(zip(labels, colors)):
                    x = 0.35 + i * 1.9
                    box = FancyBboxPatch(
                        (x, 1.05), 1.35, 0.85,
                        boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.2, edgecolor="#333333", facecolor=color,
                    )
                    ax.add_patch(box)
                    ax.text(x + 0.675, 1.475, label, ha="center", va="center", fontsize=9)
                    if i < len(labels) - 1:
                        ax.add_patch(FancyArrowPatch(
                            (x + 1.38, 1.475), (x + 1.85, 1.475),
                            arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#333333",
                        ))
                title = method if method != "Proposed method" else caption or fig_key.replace("_", " ").title()
                title = title.replace("Fig1 Framework", "Method workflow")[:70]
                ax.text(5, 2.45, title, ha="center", va="center", fontsize=11, fontweight="bold")
                fig.tight_layout(pad=0.2)
                fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                result = await self._save_figure_files(
                    fig_key, filename_stem, caption, png_path.read_bytes(),
                    already_saved=True, code_generated=True,
                )
                result["is_fallback"] = True
                result["fallback_type"] = "deterministic_concept_diagram"
                return result
            except Exception as exc:
                self.log(f"  {fig_key} deterministic diagram fallback failed: {exc}")

        image_result = await self._generate_image2_result_fallback(
            fig_key, filename_stem, caption, user_prompt, last_error
        )
        if image_result is not None:
            return image_result

        self.log(f"  {fig_key} chart generation failed — marking as failed (no placeholder)")
        return {
            "fig_key": fig_key,
            "caption": caption,
            "status": "failed",
            "error": "Chart generation failed after all retries",
        }

    async def _generate_image2_result_fallback(
        self,
        fig_key: str,
        filename_stem: str,
        caption: str,
        chart_prompt: str,
        last_error: str,
    ) -> dict[str, Any] | None:
        """Generate a result figure with the image model when chart code fails.

        The prompt writer is explicitly constrained to use only numbers already
        present in the chart prompt/evidence block. If no evidence numbers are
        present, this returns None rather than fabricating a chart.
        """
        if not chart_prompt or not re.search(r"[-+]?\d+(?:\.\d+)?", chart_prompt):
            return None
        prompt_config = self.config.for_stage("figure_prompt")
        image_config = self.config.for_stage("figure_gen")
        system = (
            "You write prompts for scientific result figures. Use only the "
            "numbers explicitly present in the supplied evidence. Never invent, "
            "estimate, smooth, or add values. If evidence is insufficient, return "
            "JSON with can_generate=false."
        )
        user = (
            f"The matplotlib code chart for {fig_key} failed.\n"
            f"Failure reason: {last_error[:800]}\n\n"
            f"Chart/evidence prompt:\n{chart_prompt[:5000]}\n\n"
            "Return JSON: {\"can_generate\": true/false, "
            "\"image_prompt\": \"prompt for a clean 2D academic result figure using only evidence numbers\", "
            "\"data_sources\": [\"source labels copied from evidence\"]}. "
            "If can_generate=true, the image_prompt must explicitly state that all plotted numbers "
            "come from the provided evidence and must include those numbers verbatim."
        )
        try:
            payload = await self.generate_json(system, user, stage_override=prompt_config)
        except Exception as exc:
            self.log(f"  {fig_key} image fallback prompt generation failed: {exc}")
            return None
        if not isinstance(payload, dict) or not payload.get("can_generate"):
            return None
        image_prompt = str(payload.get("image_prompt") or "").strip()
        if not image_prompt:
            return None
        self.workspace.write_text(f"figures/{filename_stem}_image2_fallback_prompt.txt", image_prompt)
        try:
            image_payload = await self._generate_image_with_backend_fallback(
                image_config, image_prompt, prefer_image2=True,
            )
            if not image_payload:
                return None
            b64_image, used_config, backend_meta = image_payload
            result = await self._save_figure_files(
                fig_key, filename_stem, caption, base64.b64decode(b64_image)
            )
            result.update(backend_meta)
            result.update({
                "is_fallback": True,
                "fallback_type": "image2_result_figure",
                "prompt_model": prompt_config.model,
                "image_model": used_config.model,
                "data_sources": payload.get("data_sources") if isinstance(payload.get("data_sources"), list) else [],
                "code_attempts": MAX_CODE_CHART_RETRIES,
                "failure_reason": last_error[:1000],
                "generation_prompt": image_prompt,
            })
            self.workspace.write_text(
                f"figures/{filename_stem}_image2_fallback_meta.json",
                json.dumps({
                    "fallback_type": "image2_result_figure",
                    "prompt_model": prompt_config.model,
                    "image_model": image_config.model,
                    "data_sources": result["data_sources"],
                    "code_attempts": MAX_CODE_CHART_RETRIES,
                    "failure_reason": last_error[:1000],
                }, indent=2),
            )
            return result
        except Exception as exc:
            self.log(f"  {fig_key} image2 result fallback failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Shared: save PNG + PDF + register artifacts
    # -----------------------------------------------------------------------
