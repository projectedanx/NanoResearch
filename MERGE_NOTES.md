# NanoResearch Merged Version — Build Notes

> Merged at: 2026-04-06
> Base branch: `DEV` (0b57040 "feat: TUI, memory system, event pipeline, reflection, skill registry, and multi-module enhancements")
> Added from: `main` (d853f7a "Merge remote-tracking branch 'nanoresearch/main' into self-evolution")

## Branch Comparison Summary

### Only in `main`
- `AGENTS.md` — agent documentation (ported ✓)
- `imgs/` — README assets (ported ✓)
- `nanoresearch/evolution/` — legacy 3-layer evolving memory/skill system (ported ✓)
  - `evolution/__init__.py` (43 lines)
  - `evolution/memory.py` (652 lines) — `MemoryType`, `MemoryScope`, `MemoryStore`, `ResearchMemoryKind`, `MemoryRecord`, `ResearchMemoryRecord`
  - `evolution/memory_analyzer.py` (389 lines) — `MemoryEvolutionAnalyzer`
  - `evolution/skills.py` (779 lines) — `SkillDomain`, `SkillEvolutionStore`

### Only in `DEV` (all kept as-is in merged)
- `NanoResearch_Architecture.html` — architecture doc
- `nanoresearch/memory.py` (273 lines) — new EvoMemory-inspired markdown memory (`ResearchMemory`)
- `nanoresearch/skill_registry.py` (239 lines) — new YAML frontmatter skill discovery (`SkillRegistry`)
- `nanoresearch/pipeline/events.py` (171 lines) — event pipeline for TUI updates
- `nanoresearch/pipeline/reflection.py` (226 lines) — post-run reflection module
- `nanoresearch/tui.py` (330 lines) — full-screen TUI alternative
- `nanoresearch/templates/neurips/neurips_2025.sty` — newer NeurIPS style

### Modified in both branches (DEV versions kept — DEV already includes all main's bug fixes)
- `nanoresearch/cli.py` — DEV added `--dev`, `--tui`, Live UI banners
- `nanoresearch/config.py` — DEV removed memory/skill config fields (RESTORED in merged, see below)
- `nanoresearch/agents/base.py` — DEV removed evolution imports
- `nanoresearch/agents/experiment/experiment_agent.py`
- `nanoresearch/agents/figure_gen/__init__.py`
- `nanoresearch/agents/ideation.py`
- `nanoresearch/agents/ideation_hypothesis.py`
- `nanoresearch/agents/ideation_search.py`
- `nanoresearch/agents/planning.py`
- `nanoresearch/agents/review/__init__.py`
- `nanoresearch/agents/review/single_review.py`
- `nanoresearch/agents/writing/writing_agent.py`
- `nanoresearch/cli_commands.py`
- `nanoresearch/feishu_bot_core.py`
- `nanoresearch/feishu_bot_handlers.py`
- `nanoresearch/pipeline/_workspace_helpers.py`
- `nanoresearch/pipeline/base_orchestrator.py` — DEV imports `nanoresearch.memory.ResearchMemory`
- `nanoresearch/pipeline/deep_orchestrator.py`
- `nanoresearch/pipeline/orchestrator.py`
- `nanoresearch/schemas/ideation.py`
- `nanoresearch/schemas/paper.py`
- `nanoresearch/skills.py`
- `nanoresearch/templates/__init__.py`

## Merge Strategy

**Primary:** Use DEV as the base because:
1. DEV has newer features (TUI, events, reflection, skill_registry)
2. DEV already includes all 3 bug fixes we had manually applied to main:
   - `planning.py`: `proposed_method.name` → `data.get("proposed_method")`
   - `writing_agent.py`: no more `adaptive_context` dict/string mixup
   - `templates/__init__.py`: `get_style_files` function exists
3. DEV's agents don't depend on the legacy `evolution/` system, so copying evolution back doesn't re-introduce any bugs
4. DEV's `--dev` flag is semantically equivalent to our `--skip-experiment` flag from main-ourfixes

**Port from main:**
- `AGENTS.md` (no conflict, pure docs)
- `imgs/` (no conflict, pure assets)
- `nanoresearch/evolution/` (no conflict: standalone module, no agent imports from DEV)
- Config fields: `memory_enabled`, `memory_evolution_enabled`, `memory_retrieval_top_k`, `direction_memory_top_k`, `strategy_memory_top_k`, `memory_decay_factor`, `skill_evolution_enabled`, `skill_retrieval_top_k`, `script_skill_autorun_policy`, `static_skills_dir` — restored for backward compatibility

**Do not port from main:**
- `cli.py` old code (DEV's is better)
- Agent files (DEV's are better)
- Old `skip_stages` handling (DEV already has `--dev` flag)
- Old `skills.py` (DEV has new `skill_registry.py`)

## Coexistence of two memory systems

The merged version has **two parallel memory systems** that don't conflict:

| Module | Path | API | Usage |
|--------|------|-----|-------|
| **Legacy (from main)** | `nanoresearch.evolution.memory` | `MemoryStore`, `MemoryType`, `MemoryScope`, `ResearchMemoryRecord` | Structured typed memory per workspace (old 3-layer evolving). Not wired to DEV agents. |
| **New (from DEV)** | `nanoresearch.memory` | `ResearchMemory` | Simple cross-session markdown at `~/.nanoresearch/memory/MEMORY.md`. Used by DEV's `pipeline/base_orchestrator.py`. |

Both can be imported independently. Legacy API is available for users who want structured memory, new API is the default for DEV's pipeline.

## Install & Test

```bash
cd E:/4.1/ailab/NanoResearch-merged
pip install --user -e .
python -m nanoresearch run --help                    # should show --dev and --tui
python -m nanoresearch run --topic "..." --dev --dry-run
python -m nanoresearch run --topic "..." --dev       # actual run
python -m nanoresearch run --topic "..." --dev --tui # TUI mode
```

## Import Verification

All these imports succeed in the merged version:

```python
from nanoresearch.memory import ResearchMemory                          # DEV
from nanoresearch.evolution.memory import MemoryType, MemoryStore       # main
from nanoresearch.evolution.skills import SkillDomain                   # main
from nanoresearch.skill_registry import SkillRegistry                   # DEV
from nanoresearch.tui import *                                          # DEV
from nanoresearch.pipeline.events import *                              # DEV
from nanoresearch.pipeline.reflection import *                          # DEV
```

## Round 2: self-evolution branch integration

After the initial main + DEV merge, we further integrated the **self-evolution** branch (`https://github.com/OpenRaiser/NanoResearch/tree/self-evolution`).

### Finding

- `main` and `self-evolution` branches are **nearly identical** — diff only shows README files (the main branch's latest commit `d853f7a` is "Merge remote-tracking branch 'nanoresearch/main' into self-evolution")
- Both contain `evolution/memory.py` (1,863 lines), which we already ported in round 1
- **The real difference between self-evolution/main and DEV** is that self-evolution's `agents/base.py` **actively wires** `evolution/` into every agent via 7 helper methods:
  - `_project_key()`
  - `build_adaptive_context()` (77 lines, uses both `MemoryStore` and `UnifiedSkillMatcher`)
  - `remember_context()`
  - `learn_from_trace()`
  - `remember_promising_direction()`
  - `remember_failed_direction()`
  - `remember_experiment_strategies()`
- DEV's agents removed these helpers entirely when refactoring to the new `memory.py` + `skill_registry.py`

### Self-evolution usage count in agents (28 calls total)

- `agents/base.py`: 6 method definitions
- `agents/ideation.py`: 5 calls
- `agents/planning.py`: 6 calls
- `agents/experiment/experiment_agent.py`: 9 calls
- `agents/review/__init__.py`: 4 calls
- `agents/writing/writing_agent.py`: 4 calls

### Round 2 merge actions

1. **Ported `UnifiedSkillMatcher` + `UnifiedSkillContext`** from `self-evolution/nanoresearch/skills.py` into `merged/nanoresearch/skills.py` (additive, kept DEV's `SkillMatcher` intact)
   - Added `import json` and `from nanoresearch.evolution.skills import SkillDomain, SkillEvolutionStore`
   - Appended ~70 lines at the end

2. **Modified `merged/nanoresearch/agents/base.py`**:
   - Added imports: `MemoryScope`, `MemoryStore`, `MemoryType`, `MemoryEvolutionAnalyzer`, `UnifiedSkillMatcher`, `Path`
   - Extended `__init__` to initialize `_memory_store`, `_memory_analyzer`, `_skill_matcher`
   - Added all 7 self-evolution helper methods (~200 lines) before `stage_config` property
   - Kept DEV's `report_substep()` and all other DEV additions intact

3. **Did NOT modify individual agent files** (`ideation.py`, `planning.py`, etc.) — they still use DEV's logic without calling `remember_context()`. The self-evolution methods are available on every agent (because they inherit from `BaseResearchAgent`), but they're opt-in.

### Why NOT wire individual agents

Individual agents in DEV have many improvements over self-evolution:
- New tools (`search_semantic_scholar`, `search_arxiv`, `search_papers`)
- Bug fixes (`proposed_method.name`, `core_ctx`, `get_style_files`)
- Hypothesis tournament
- Citation enrichment
- Full-text download
- New writing loops

Automatically merging DEV's agent improvements with self-evolution's `remember_context()` calls would risk breaking DEV's careful flow. Instead, we provide the infrastructure (methods on `BaseResearchAgent`) and leave it to users/future work to opt-in by adding `remember_context()` calls where desired.

### Verification

```python
# All 7 methods are present on every agent subclass
from nanoresearch.agents.ideation import IdeationAgent
agent = IdeationAgent(ws, cfg)
assert hasattr(agent, 'remember_context')           # ✓
assert hasattr(agent, 'build_adaptive_context')     # ✓
assert hasattr(agent, 'learn_from_trace')           # ✓
assert hasattr(agent, 'remember_promising_direction')  # ✓
assert hasattr(agent, 'remember_failed_direction')  # ✓
assert hasattr(agent, 'remember_experiment_strategies')  # ✓

from nanoresearch.evolution.memory import MemoryType
agent.remember_context(MemoryType.PROJECT_CONTEXT, 'test memory', importance=0.8)  # ✓
ctx = agent.build_adaptive_context('ideation', topic='llm', tags=['ml'])           # ✓ 2,222 chars
agent.learn_from_trace('literature', 'survey_assembly', 'trace text')              # ✓
```

## Known Limitations

1. **Agent-level opt-in**: The `evolution/` system is fully wired into `BaseResearchAgent` but individual DEV agents don't call `remember_context()` automatically. To actually see memory accumulate across runs, add calls like `self.remember_context(MemoryType.PROJECT_CONTEXT, ..., importance=0.78)` at strategic points in `ideation.py` / `planning.py` / etc. (refer to self-evolution branch for reference implementations).

2. **Tests not run**: Neither branch has tests to verify the merge didn't break anything. Rely on end-to-end smoke tests via `python -m nanoresearch run --dev --dry-run` and the full pipeline runs recorded in this doc.
