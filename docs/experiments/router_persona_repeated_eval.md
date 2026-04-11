# Router Persona Repeated Evaluation

This document defines the local research tooling for the 10-persona repeated NanoResearch experiment protocol.

## Inputs

### Question manifest
Use `tools/build_router_persona_experiment_manifest.py` with a JSON array or JSONL file whose rows contain:

- `question_id`
- `domain`
- `difficulty`
- `background`
- `baselines`
- `datasets`
- `user_requirements`

Optional fields accepted by the deep runner:

- `problem_statement`
- `extra_context`
- `persona_brief`

The builder expands each question over:
- 10 default personas
- 8 main ablation variants
- 1 appendix-only context-informed baseline (optional)

### Result records
Use `tools/aggregate_router_persona_results.py` on JSON or JSONL result files with rows containing:

- `persona_id`
- `variant_name`
- `question_id`
- `novelty_score`
- `alignment_pass_at_1`
- `alignment_token_to_pass`
- `plan_executability`
- `implementation_token_to_runnable`
- `implementation_success`
- `final_performance`
- `baseline_performance`
- `total_tokens_from_method_to_code`

`delta_over_baseline` is optional. If omitted, the aggregator computes it from `final_performance - baseline_performance`.

## Runner

### Deep-pipeline execution
Use `tools/run_router_persona_deep_experiment.py` to execute the real deep pipeline on each manifest row. The runner:

- constructs a persona-aware topic prompt from the manifest row
- runs `IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS`
- skips `FIGURE_GEN`, `WRITING`, and `REVIEW`
- judges alignment and novelty with the configured review model
- writes one `result.json` per assignment plus a merged `results.jsonl`

Example smoke run:

```bash
PYTHONPATH=. /usr/bin/python3 tools/run_router_persona_deep_experiment.py \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output \
  --variant base_router \
  --persona resource_constrained_repro_first \
  --limit 1
```

### Current SDPO limitation
The deep runner currently refuses SDPO-bearing variants (`sdpo_only`, `memory_sdpo`, `skill_sdpo`, `full_system`) because the main deep pipeline does not yet have a real same-router SDPO backend wired into `build_adaptive_context`. This is an explicit guard to avoid generating invalid ablation results.

If you want a non-SDPO smoke run now, either:

- filter to `base_router`, `memory_only`, `skill_only`, or `memory_skill`
- or pass `--skip-sdpo-variants`

## Outputs

The aggregator writes:

- `summary.json`
- `main_table.json`
- `efficiency_table.json`
- `persona_breakdown.json`
- `appendix_baselines.json`
- `ablation_contributions.json`
- `main_table.tex`
- `efficiency_table.tex`

The deep runner writes:

- `selected_assignments.json`
- `skipped_assignments.json`
- `results.jsonl`
- `run_summary.json`
- `<assignment>/result.json`
- `<assignment>/workspaces/attempt-*/...`

## Reporting semantics

- `main_table` reports macro-averaged core metrics across personas.
- `efficiency_table` reports macro-averaged efficiency metrics across personas.
- `persona_breakdown` preserves per-persona averages before macro aggregation.
- `appendix_baselines` keeps `context_informed_generation` out of the main table.
- `ablation_contributions` ranks the drop from `Full System` to each ablation for each core metric.
