# PubMedQA Full System Real Case Package

这个目录是一个真实跑通的 `NanoResearch full_system` case 审阅包，来自 2026-04-19 的 planner 版本实验。

目标不是做摘要，而是把 co-author 写说明文档时真正需要翻的原始材料集中到一个可直接浏览的目录里。

## 1. 这条 case 是什么

- 任务：`light_nlp_biomed_qa`
- persona：`ai4science_journal_conservative`
- variant：`full_system`
- round：`round01`
- 数据集：`PubMedQA`

原始来源：

- workspace  
  `/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/ai4science_journal_conservative-full_system-light_nlp_biomed_qa-round01/workspaces/attempt-01`
- chain home  
  `/mnt/dhwfile/raise/user/xujinhang/nanoresearch/router_plan_fullsystem_20260419_10proc_r2/run/_chains/ai4science_journal_conservative-full_system-light_nlp_biomed_qa/nanoresearch_home`

## 2. 目录结构

- `case_workspace/`
  - 这次真实运行的 workspace 拷贝
- `chain_home/`
  - 这条链对应的 `profile / memory / skill` 原始文件

## 3. co-author 先看哪些文件

如果要快速恢复整条流程，先按这个顺序看：

1. `case_workspace/logs/deep_assignment_context.json`
   - 用户任务输入
2. `chain_home/profile/profile.json`
   - profile
3. `case_workspace/papers/ideation_output.json`
   - IDEATION 输出
4. `case_workspace/plans/experiment_blueprint.json`
   - PLANNING 输出
5. `case_workspace/plans/setup_output.json`
   - SETUP 输出
6. `case_workspace/plans/code_plan.json`
   - router 后 planner 给 CODING 的显式计划
7. `case_workspace/plans/coding_output.json`
   - CODING 结果
8. `case_workspace/plans/execution_output.json`
   - EXECUTION 结果
9. `case_workspace/plans/analysis_output.json`
   - ANALYSIS 结果
10. `case_workspace/results/metrics.json`
   - 最终性能
11. `chain_home/memory/records.json`
   - memory
12. `chain_home/memory/research_records.json`
   - research memory
13. `chain_home/skills/natural_language.json`
   - evolved skill

## 4. 每个阶段常用原始文件

### IDEATION

- `case_workspace/logs/adaptive_context_ideation_literature.json`
- `case_workspace/papers/ideation_output.json`
- `case_workspace/logs/promising_direction_summary_ideation.json`

### PLANNING

- `case_workspace/logs/adaptive_context_planning_planning.json`
- `case_workspace/plans/experiment_blueprint.json`
- `case_workspace/logs/blueprint_review.json`
- `case_workspace/logs/promising_direction_summary_planning.json`

### SETUP

- `case_workspace/logs/adaptive_context_setup_experiment.json`
- `case_workspace/plans/setup_output.json`
- `case_workspace/repos/`
- `case_workspace/dataset_repos/`

### CODING

- `case_workspace/logs/adaptive_context_coding_coding.json`
- `case_workspace/plans/code_plan.json`
- `case_workspace/plans/coding_output.json`
- `case_workspace/experiment/`

### EXECUTION

- `case_workspace/logs/adaptive_context_experiment_experiment.json`
- `case_workspace/plans/execution_output.json`
- `case_workspace/plans/debug_round_*.json`
- `case_workspace/logs/execution_remediation_ledger.json`
- `case_workspace/logs/cluster_job_state.json`
- `case_workspace/logs/cluster_job_events.jsonl`

### ANALYSIS

- `case_workspace/logs/adaptive_context_analysis_analysis.json`
- `case_workspace/plans/analysis_output.json`
- `case_workspace/results/metrics.json`
- `case_workspace/results/metrics.csv`
- `case_workspace/results/test_predictions.jsonl`
- `case_workspace/drafts/experiment_summary.md`

### FIGURE_GEN / WRITING / REVIEW

这条真实 case 的主落盘范围到 `ANALYSIS`。目录里保留了运行现场，但没有完整写作产物。

## 5. 这个包里保留了什么

保留：

- 原始 `manifest.json` 和 `progress.json`
- `plans/` 全部阶段产物
- `logs/` 全部 adaptive context 和执行日志
- `papers/`、`drafts/`
- `experiment/` 生成代码
- `results/` 下的指标、日志、预测结果、tokenizer 配置
- `datasets/`
- `repos/` 和 `dataset_repos/`
- `profile / memory / skills`

## 6. 这个包里刻意排除了什么

排除：

- `results/best_model/model.safetensors`
- `results/best_model/trainer_state.pt`
- `experiment/__pycache__/`
- `experiment/results` 软链接
- `models/BioBERT`、`models/PubMedBERT-base`、`models/T5-small` 软链接

另外做了 GitHub 审阅友好的处理：

- `repos/` 和 `dataset_repos/` 内部的 `.git` 目录已移除，避免被记录成嵌套仓库
- 原来指向本地缓存和结果目录的软链接，改成说明文件保留来源信息

原因：

- 前两个文件体积过大，只会让仓库膨胀，不影响 co-author 写流程说明
- `__pycache__` 是运行缓存，不是原始研究产物
- 软链接在 GitHub 上无法代表真实内容来源，保留说明文件更适合共同写作

## 7. 使用方式

这个包的用途是：

1. 让 co-author 直接翻真实原始文件写 case study
2. 让论文里每个阶段的输入输出都能落到实际文件
3. 让后续补文档时，可以直接引用这里的原始 JSON 和代码，而不是回原始大实验目录里找
