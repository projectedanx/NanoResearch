"""Standalone re-run of FIGURE_GEN stage on an existing session.

Used to repair session a525e67ac4ed (figures all failed because of stale
model ids). After config.py was patched with the correct gemini /
claude model ids, this script:
  1. loads the existing session workspace
  2. reads ideation_output + experiment_blueprint from disk
  3. instantiates FigureAgent with the freshly-corrected ResearchConfig
  4. calls agent.run() with the same input shape the orchestrator uses
  5. writes drafts/figure_output.json (overwriting the failed one)

We do NOT touch any other stage. WRITING / REVIEW outputs remain as-is.
A separate manual step then patches paper.tex to add \\includegraphics.
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

# Add the merged-branch repo to the path
REPO = Path(r"E:/4.1/ailab/NanoResearch-merged")
sys.path.insert(0, str(REPO))

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.agents.figure_gen import FigureAgent

SESSION_ID = "a525e67ac4ed"
SESSION_DIR = Path.home() / ".nanoresearch" / "workspace" / "research" / SESSION_ID


async def main() -> int:
    print(f"[rerun] session: {SESSION_ID}")
    print(f"[rerun] dir:     {SESSION_DIR}")

    # 1. Load config (will pick up the patched model ids in config.py)
    cfg = ResearchConfig.load()
    # NOTE: A deepseek-V3.2 runtime override was here from 2026-04-07 to work
    # around a transient claude-sonnet-4-6 proxy 503.  Removed 2026-04-09 —
    # upstream recovered 04-08 evening.  See §6.10.7 of汇报_2026-04-07.md.
    print(f"[rerun] figure_prompt model: {cfg.figure_prompt.model}")
    print(f"[rerun] figure_code   model: {cfg.figure_code.model}")
    print(f"[rerun] figure_gen    model: {cfg.figure_gen.model}  ({cfg.figure_gen.image_backend})")

    # 2. Load workspace pointing at the existing session
    workspace = Workspace.load(SESSION_DIR)
    print(f"[rerun] workspace topic: {workspace.manifest.topic}")

    # 3. Read the inputs from disk in the same shape the orchestrator builds
    ideation_path = SESSION_DIR / "papers" / "ideation_output.json"
    blueprint_path = SESSION_DIR / "plans" / "experiment_blueprint.json"
    if not ideation_path.is_file():
        print(f"[rerun] ERROR: missing {ideation_path}")
        return 2
    if not blueprint_path.is_file():
        print(f"[rerun] ERROR: missing {blueprint_path}")
        return 2

    ideation_output = json.loads(ideation_path.read_text(encoding="utf-8"))
    experiment_blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))

    # 4. Build agent inputs — mirrors deep_orchestrator._prepare_inputs(FIGURE_GEN)
    inputs = {
        "ideation_output": ideation_output,
        "experiment_blueprint": experiment_blueprint,
        "experiment_results": {},          # --dev mode = no real results
        "experiment_analysis": {},
        "experiment_summary": "",
        "experiment_status": "pending",
        "existing_figures": {},
        "survey_blueprint": {},
    }

    # 5. Instantiate agent and run
    agent = FigureAgent(workspace, cfg)
    # Light substep callback so we get progress on stdout
    agent._substep_callback = lambda msg: print(f"[rerun]   {msg}")
    print("[rerun] starting FigureAgent.run() ...")
    try:
        result = await agent.run(**inputs)
    except Exception as exc:
        import traceback
        print(f"[rerun] FigureAgent.run() raised: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3

    # 6. Summarize result
    figs = result.get("figures", {}) if isinstance(result, dict) else {}
    print()
    print(f"[rerun] FigureAgent.run() returned {len(figs)} figure entries")
    for k, v in figs.items():
        status = v.get("status", "?") if isinstance(v, dict) else "?"
        path = v.get("path", "") if isinstance(v, dict) else ""
        err = v.get("error", "") if isinstance(v, dict) else ""
        print(f"  {k:30s} status={status}  path={path or '(none)'}{('  error=' + err[:60]) if err else ''}")

    # 7. List what's actually in figures/
    figures_dir = SESSION_DIR / "figures"
    if figures_dir.is_dir():
        files = sorted(figures_dir.iterdir())
        print()
        print(f"[rerun] figures/ now contains {len(files)} files:")
        for f in files:
            print(f"  {f.stat().st_size:>10}  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
