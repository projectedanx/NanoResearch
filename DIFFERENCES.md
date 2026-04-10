# Feature Matrix: main vs DEV vs merged

Summary of what each branch has. Everything in **merged** is present and working.

| Feature | main | DEV | merged |
|---------|:---:|:---:|:---:|
| 9-stage pipeline (IDEATION→...→REVIEW) | ✅ | ✅ | ✅ |
| Skip experiment stages | ❌ (only via config file) | ⚠️ `--dev` flag is a no-op (lowercase bug) | ✅ **`--dev` flag works** (fixed uppercase) |
| `--skip` flag | ❌ | ⚠️ same lowercase bug | ✅ fixed |
| `--tui` full-screen TUI | ❌ | ✅ | ✅ |
| Inline Live UI welcome banner | ❌ | ✅ | ✅ |
| `AGENTS.md` documentation | ✅ | ❌ | ✅ (ported from main) |
| `imgs/` README assets | ✅ | ❌ | ✅ (ported from main) |
| `NanoResearch_Architecture.html` | ❌ | ✅ | ✅ |
| Legacy `evolution/memory` (`MemoryStore`, `MemoryType`) | ✅ | ❌ | ✅ (ported from main) |
| Legacy `evolution/skills` (`SkillEvolutionStore`) | ✅ | ❌ | ✅ (ported from main) |
| Legacy `evolution/memory_analyzer` | ✅ | ❌ | ✅ (ported from main) |
| New `memory.ResearchMemory` (markdown-based) | ❌ | ✅ | ✅ |
| New `skill_registry.SkillRegistry` (YAML frontmatter) | ❌ | ✅ | ✅ |
| `pipeline/events.py` | ❌ | ✅ | ✅ |
| `pipeline/reflection.py` | ❌ | ✅ | ✅ |
| `tui.py` | ❌ | ✅ | ✅ |
| NeurIPS 2025 style file | ❌ | ✅ | ✅ |
| Semantic Scholar integration in IDEATION | ❌ | ✅ (`search_semantic_scholar` tool) | ✅ |
| Hypothesis tournament | ❌ | ✅ | ✅ |
| Quantitative evidence extraction | ❌ | ✅ | ✅ |
| Citation count enrichment | ❌ | ✅ | ✅ |
| Full-text download for top papers | ❌ | ✅ | ✅ |
| `memory_evolution_enabled` config field | ✅ | ❌ (removed) | ✅ (restored for backward-compat) |
| `skill_evolution_enabled` config field | ✅ | ❌ (removed) | ✅ (restored for backward-compat) |
| `proposed_method.name` bug | ⚠️ (manually fixed by us in main-ourfixes) | ✅ fixed | ✅ |
| `get_style_files` import error | ⚠️ (manually fixed) | ✅ fixed | ✅ |
| `core_ctx` dict/string mixup | ⚠️ (manually fixed) | ✅ fixed | ✅ |

## Bugs found and fixed in the merged version

1. **`--dev` lowercase/uppercase mismatch** (DEV-only bug, discovered during our run)
   - `cli.py` added `["setup", "coding", "execution", "analysis"]` (lowercase)
   - `pipeline/base_orchestrator.py:169` checked `stage.value in config.skip_stages` where `stage.value` is UPPERCASE (from `PipelineStage.SETUP = "SETUP"`)
   - Result: `--dev` was a no-op, pipeline ran all 9 stages anyway
   - Fix: changed to uppercase `["SETUP", "CODING", "EXECUTION", "ANALYSIS"]` in both `run` and `resume` commands
   - Also fixed the `--skip` flag which had the same bug
   - Also fixed the "(will skip)" display in resume status table

## Files changed in merged (vs DEV base)

```
AGENTS.md                              (new, from main)
imgs/                                  (new dir, from main)
nanoresearch/evolution/                (new dir, from main — 4 files, 1863 lines)
nanoresearch/config.py                 (added back 10 memory/skill config fields)
nanoresearch/cli.py                    (fixed --dev and --skip lowercase bug in 2 places)
MERGE_NOTES.md                         (new, merge strategy doc)
DIFFERENCES.md                         (new, this file)
```
