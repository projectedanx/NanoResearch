# Literature Skill: Require one ablation per core module

- id: skill-b02a6803ef8d
- version: 0.1.0
- domain: literature
- trigger_pattern: retrieval_pipeline_summary
- last_updated: 2026-04-18T18:21:26.627196+00:00
- tags: evaluation question id: light_nlp_biomed_qa persona profile: ai4science journal persona that prefers conservative claims, careful ablations, and strong scientific grounding. research domain: nlp difficulty: incremental_innovation problem statement: design a practical method for improving pubmedqa under limited compute while keeping the implementation easy to reproduce. background context: biomedical qa systems already perform reasonably well on pubmedqa, but lightweight improvements with clean ablations and reproducible training are still valuable. known baselines: biobert; pubmedbert; instruction-tuned biomedical qa baseline evaluation datasets: pubmedqa user requirements: generate a new idea and an implementation-oriented plan. keep the method lightweight, ablatable, and reproducible. additional context: prefer methods that fit within a modest single-node budget and can be compared fairly against standard biomedical qa baselines. task: propose a new research idea, turn it into a rigorous experimental plan, implement the resulting experiment, and analyze the final outcome., literature, original_research, retrieval_pipeline

## Description
Require one ablation per core module.

## When To Use
Use when 'retrieval_pipeline_summary' or a similar recurring pattern appears in literature work.

## Instructions
1. Require one ablation per core module.
2. Make each claimed component correspond to an isolated removal study.

## Provenance
- source_stage: ideation
- source_trace: Literature retrieval for Evaluation Question ID: light_nlp_biomed_qa Persona Profile: AI4Science journal persona that prefers conservative claims, careful ablations, and strong scientific grounding. Research Domain: NLP Difficulty: incremental_innovation Problem Statement: Design a practical method for improving PubMedQA under limited compute while keeping the implementation easy to reproduce. Background Context: Biomedical QA systems already perform reasonably well on PubMedQA, but lightweight improvements with clean ablations and reproducible training are still valuable. Known Baselines: BioBERT; PubMedBERT; instruction-tuned biomedical QA baseline Evaluation Datasets: PubMedQA User Requirements: Generate a new idea and an implementation-oriented plan. Keep the method lightweight, ablatable, and reproducible. Additional Context: Prefer methods that fit within a modest single-node budget and can be compared fairly against standard biomedical QA baselines. Task: Propose a new research 
