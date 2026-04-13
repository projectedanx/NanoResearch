# NanoResearch NIPS 实验线代码整理与集成说明

更新时间：2026-04-13

本文档用于整理当前 NanoResearch NIPS 实验线的代码入口、核心流程、实验设置、运行目录、以及与 `OpenRaiser/main` 的集成关系。它的目标不是展示单次实验结果，而是回答下面几个实际问题：

1. 现在这套 NIPS 实验到底是由哪些代码驱动的。
2. Nano 的完整 research 流程在代码里是怎么实现的。
3. 当前反复提到的 8 个变体、3 轮演化、persona-topic 配对、SDPO router、memory/skill 隔离分别落在哪些模块。
4. 当前代码是否适合直接合入 `main`，如果不适合，应该如何集成。

## 1. 当前分支与主线关系

### 1.1 当前整理分支

- 当前工作分支：`codex/integrate-nips-setting-20260413`
- 实验功能提交基线：`88dc317`

### 1.2 `OpenRaiser/main` 的关系

2026-04-13 本地检查结果：

| 项目 | 值 |
| --- | --- |
| 当前实验线已提交头 | `88dc317` |
| `OpenRaiser/main` 头 | `7130f14` |
| 共同 merge-base | `d853f7a` |

这意味着当前实验线不是直接建立在最新 `OpenRaiser/main` 之上，而是从更早的共同祖先分叉后继续开发。与此同时，工作树里还叠加了大量尚未提交的实验修复与运行相关修改。

结论：

- 当前不建议直接把整棵工作树强行 merge 到 `main`
- 更合理的做法是先把实验线整理成一个明确的集成分支，再单独处理和 `OpenRaiser/main` 的同步

## 2. 这条实验线的核心代码入口

下面只列当前 NIPS 实验最核心的路径。

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 变体定义与聚合 | `nanoresearch/experiments/router_persona_eval.py` | 定义 8 个主变体、manifest 展开规则、结果聚合指标 |
| 深度评测执行入口 | `nanoresearch/experiments/deep_persona_runner.py` | 把单个 assignment 跑通到 `result.json`，并计算 novelty/alignment/performance/delta |
| SDPO 在线 router | `nanoresearch/router_policy.py` | 加载本地 checkpoint 或远程 OpenAI-compatible endpoint，产出 memory/skill/prompt plan 决策 |
| Agent 自适应上下文拼装 | `nanoresearch/agents/base.py` | 负责 persona profile、memory、skill、SDPO router 的统一接线 |
| Persona/profile 系统 | `nanoresearch/profile.py` | 10 个 persona 的 seed / override / router hint 生成与保存 |
| 统一 pipeline 编排 | `nanoresearch/pipeline/base_orchestrator.py` | 管理 stage 顺序、progress、cost、重试、resume |
| Planning 阶段 blueprint 生成 | `nanoresearch/agents/planning.py` | 生成实验 blueprint，并做 schema + LLM review |
| Blueprint 静态校验 | `nanoresearch/pipeline/blueprint_validator.py` | 做轻量结构/语义检查，主要防明显 execution-invalidating 问题 |
| Memory store | `nanoresearch/evolution/memory.py` | 长期 memory 和 research memory 的持久化与检索 |
| Skill store | `nanoresearch/evolution/skills.py` | 自演化 skill 的候选、review、持久化与 artifact 输出 |
| 平衡 manifest 构建 | `tools/build_balanced_router_persona_manifest.py` | 构造 10 topic × 10 persona 的 ring pairing |
| 深评测 CLI | `tools/run_router_persona_deep_experiment.py` | 从 manifest 执行实验并写出 `results.jsonl` / `result.json` |
| canonical baseline 查表 | `nanoresearch/experiments/canonical_baselines.py` | 用共享 topic baseline 替换单次 blueprint 中不稳定的 baseline 抽取 |
| canonical delta 重算 | `tools/recompute_router_persona_canonical_deltas.py` | 离线重算 `baseline_performance` 和 `delta_over_baseline` |

## 3. Nano 在当前实验线中的完整流程

当前 NIPS 实验使用的是 deep pipeline 的 research 子链，不跑写论文的后半段。真实执行链路如下：

1. `build_balanced_router_persona_manifest.py`
2. `tools/run_router_persona_deep_experiment.py`
3. `nanoresearch/experiments/deep_persona_runner.py`
4. `UnifiedPipelineOrchestrator`
5. `IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS`

明确跳过的 stage：

- `FIGURE_GEN`
- `WRITING`
- `REVIEW`

### 3.1 单个 assignment 是怎么跑的

`deep_persona_runner.run_assignment()` 会为每个 assignment 做这些事：

1. 根据 `persona_id` 构造 profile，并保存到当前 chain 的隔离 `NANORESEARCH_HOME`
2. 根据变体开关覆盖 `ResearchConfig`
3. 生成 topic 文本，把 question、persona、baseline、dataset、user requirement 拼成统一输入
4. 跑 deep pipeline
5. 在 runner 外部额外做：
   - alignment judgment
   - novelty judgment
   - metric 抽取
   - baseline / delta 计算
6. 把所有结果写到 assignment 自己的 `result.json`

### 3.2 为什么现在 memory/skill 不会串

当前实现里，每个 `persona × variant × topic` 都有自己的 chain：

- `chain_id = "{persona}::{variant}::{question_id}"`
- `NANORESEARCH_HOME = output/_chains/{chain_slug}/nanoresearch_home`

因此：

- 不同变体不共享 memory
- 不同 persona 不共享 memory
- 不同 topic 不共享 memory
- 同一条 chain 的 round1/round2/round3 会沿用同一个 `NANORESEARCH_HOME`

这正是这轮“3 轮演化”实验希望测的东西：只允许同一条 chain 内部积累 memory / skill。

## 4. 8 个变体在代码里的真实定义

定义位置：`nanoresearch/experiments/router_persona_eval.py`

| 变体名 | `memory_self_evolution` | `skill_self_evolution` | `same_router_hindsight_sdpo` | 含义 |
| --- | --- | --- | --- | --- |
| `base_router` | `false` | `false` | `false` | 不用 memory，不用 skill，不用 SDPO |
| `memory_only` | `true` | `false` | `false` | 只开 memory |
| `skill_only` | `false` | `true` | `false` | 只开 skill |
| `sdpo_only` | `false` | `false` | `true` | 只开 SDPO router |
| `memory_skill` | `true` | `true` | `false` | memory + skill |
| `memory_sdpo` | `true` | `false` | `true` | memory + SDPO |
| `skill_sdpo` | `false` | `true` | `true` | skill + SDPO |
| `full_system` | `true` | `true` | `true` | memory + skill + SDPO |

`deep_persona_runner.resolve_variant_runtime_settings()` 会把这个定义映射成真正的 runtime config。

## 5. Persona 系统的真实作用

persona 不是只在文档里写一句描述，而是会进入 profile 系统，影响 router 和 prompt 的上下文。

实现位置：

- archetype seed：`nanoresearch/profile.py`
- 10 persona 的 override：`nanoresearch/experiments/deep_persona_runner.py`

每个 persona 最终会生成：

- `research_profile`
- `resource_profile`
- `writing_profile`
- `publication_profile`
- `interaction_profile`
- `router_hints`

这些 profile 会在 agent 构造 adaptive context 时进入 prompt，也会影响 SDPO router 的输入 payload。

## 6. Memory / Skill / SDPO 在流程中分别怎么生效

### 6.1 Memory

实现位置：`nanoresearch/evolution/memory.py`

当前 memory 分两类：

- 通用长期 memory
  - `USER_PROFILE`
  - `PROJECT_CONTEXT`
  - `DECISION_HISTORY`
- research memory
  - `PROMISING_DIRECTION`
  - `FAILED_DIRECTION`
  - `DATA_STRATEGY`
  - `TRAINING_STRATEGY`

在没有开启 SDPO router 时，`BaseResearchAgent.build_adaptive_context()` 会直接把 memory render 成 prompt context。

### 6.2 Skill

实现位置：`nanoresearch/evolution/skills.py`

当前 skill 分两类：

- 自演化自然语言 skill
- 注册脚本 skill

没有 SDPO router 时，`BaseResearchAgent.build_adaptive_context()` 会直接把静态 skill + evolved skill 拼进 prompt。

### 6.3 SDPO router

实现位置：`nanoresearch/router_policy.py` + `nanoresearch/agents/base.py`

当 `same_router_hindsight_sdpo_enabled = true` 时：

1. agent 不再盲目把所有 memory/skill 直接塞进 prompt
2. 会先构造 router payload
3. 调用 SDPO router 决策：
   - `selected_memory_ids`
   - `selected_skill_ids`
   - `prompt_plan`
   - `update_memory`
   - `update_skill`
4. 只把 router 选中的 memory / skill 注入上下文

支持两种后端：

- 本地 HF checkpoint
- 远程 OpenAI-compatible endpoint

## 7. Blueprint 校验现在是什么逻辑

这里需要和之前的讨论对齐：

- 现在不是单纯靠硬编码框架做最终判定
- 也不是完全把所有检查都交给一个模糊的后验 judge

真实逻辑是两层：

1. `ExperimentBlueprint` 的 schema 校验
2. `PlanningAgent._review_blueprint_with_llm()` 的 LLM review

此外 `nanoresearch/pipeline/blueprint_validator.py` 仍保留了轻量静态检查函数，但 orchestrator 备注已经说明：

- 语义 blueprint review 主要在 `PlanningAgent` 内部完成
- orchestrator 不再额外运行一套重的硬编码 heuristic validator

因此当前设计更接近：

- 结构合法性：schema
- 可执行性/评估有效性：LLM review
- 轻量规则检查：保留辅助函数，但不是主裁决者

## 8. 当前轻量化实验设置

### 8.1 题目集

当前主测试集文件：

- `docs/experiments/lightweight_router_persona_questions_v2.json`

共 10 个轻量 topic：

1. `light_nlp_biomed_qa`
2. `light_nlp_short_text_cls`
3. `light_nlp_sentence_pair_cls`
4. `light_cv_small_image_cls`
5. `light_multimodal_efficiency`
6. `light_tabular_budgeted_cls`
7. `light_tabular_regression`
8. `light_timeseries_sensor_cls`
9. `light_graph_node_cls`
10. `light_audio_keyword_cls`

这些题目的共同特点是：

- 单 GPU 可运行
- baseline 清楚
- 数据集轻量
- 适合快速验证 idea -> code -> metric 这条链

### 8.2 Persona-topic 配对

当前采用 balanced ring pairing：

- 10 个 topic
- 10 个 persona
- 每个 topic 对应 2 个 persona
- 每个 persona 覆盖 2 个 topic

具体实现：

- 构造器：`tools/build_balanced_router_persona_manifest.py`
- pairing 文件样例：`/mnt/dhwfile/raise/user/xujinhang/nanoresearch/balanced_ring10topics_minimax3r_20260413_132957/pairings.json`

### 8.3 轮数与 assignment 数

当前主 batch summary：

- 文件：`/mnt/dhwfile/raise/user/xujinhang/nanoresearch/balanced_ring10topics_minimax3r_20260413_132957/manifest_balanced_3rounds_summary.json`

对应设置：

| 项目 | 值 |
| --- | --- |
| topic 数 | 10 |
| persona 数 | 10 |
| 变体数 | 8 |
| rounds | 3 |
| appendix baseline | 不包含 |
| 总 assignments | 480 |

计算方式：

- `10 topics × 2 personas/topic × 8 variants × 3 rounds = 480`

## 9. 指标计算现在落在哪里

结果记录由 `deep_persona_runner.build_result_record()` 统一产出。

当前核心指标：

| 指标 | 当前来源 |
| --- | --- |
| `novelty_score` | runner 外部 LLM judge |
| `alignment_pass_at_1` | runner 外部 LLM judge |
| `alignment_token_to_pass` | `IDEATION + PLANNING` token 累积 |
| `implementation_success` | execution 输出状态解析 |
| `implementation_token_to_runnable` | `SETUP + CODING + EXECUTION` token |
| `final_performance` | 从 execution / analysis 中抽主指标 |
| `baseline_performance` | 优先 canonical baseline registry，否则回退到 blueprint baseline |
| `delta_over_baseline` | 按 metric 方向做差 |

### 9.1 为什么现在要用 canonical baseline

当前新增：

- baseline registry：`docs/experiments/lightweight_router_persona_canonical_baselines_v1.json`
- 查表模块：`nanoresearch/experiments/canonical_baselines.py`
- 重算脚本：`tools/recompute_router_persona_canonical_deltas.py`

原因很直接：

- 以前每个变体的 blueprint 可能给出不同 baseline 数字
- 导致同一 topic 的 `delta_over_baseline` 不一致，甚至缺失
- 现在改成 topic 级共享 canonical baseline，更符合对比实验的要求

## 10. 代码默认配置 vs 当前运行批次配置

### 10.1 仓库里的默认值

默认配置定义在 `nanoresearch/config.py`。

其中当前和实验最相关的默认值有：

| 配置项 | 默认值 |
| --- | --- |
| `code_gen.model` | `MiniMax-M2.7` |
| `same_router_hindsight_sdpo_enabled` | `false` |
| `slurm_quota_type` | `auto` |
| `slurm_default_time` | `""` |
| `execution_auto_repair_enabled` | `false` |

### 10.2 当前 batch 的实际覆盖配置

当前活跃运行目录里，实验配置是通过 batch 自己的 `config.json` 覆盖的。

#### 旧主 batch

- 目录：`/mnt/dhwfile/raise/user/xujinhang/nanoresearch/balanced_ring10topics_minimax3r_20260413_132957`
- `code_gen.model`：`MiniMax-M2.7`

#### Kimi smoke

- 目录：`/mnt/dhwfile/raise/user/xujinhang/nanoresearch/smoke_kimi_fullsystem_20260413_155729`
- `code_gen.model`：`Pro/moonshotai/Kimi-K2.5`

#### Kimi failed-shard rerun

- 目录：`/mnt/dhwfile/raise/user/xujinhang/nanoresearch/rerun_kimi_failedshards_20260413_155729`
- `code_gen.model`：`Pro/moonshotai/Kimi-K2.5`
- 待补跑 assignments：`283`
- 其中 `full_system` 待补：`36`

### 10.3 这两层配置不能混为一谈

因此需要明确区分：

- 仓库代码默认值目前仍是 `MiniMax-M2.7`
- 但实际正在补跑的 batch 已经通过批次配置切到 `Pro/moonshotai/Kimi-K2.5`

## 11. 当前已知问题与修复状态

### 11.1 已经修到代码里的

1. 不同变体之间的 memory / skill 串扰
2. 3 轮实验链没有按 `persona × variant × topic` 隔离
3. `slurm_default_time` 允许为空，不再强制写时间上限
4. `slurm_quota_type` 支持 `auto`
5. blueprint review 改成以 LLM review 为主
6. canonical baseline 引入，delta 不再依赖每个 blueprint 自己瞎写 baseline

### 11.2 还没彻底解决的

1. `final_performance` 抽取仍偏脆弱
   - 某些任务实际上跑出了指标
   - 但 `result.json.final_performance` 仍是 `null`
2. `MiniMax-M2.7` 在部分 coding / execution 阶段存在空回复问题
3. `full_system` 的 Kimi batch 还在补跑中，尚需继续验证稳定性

## 12. 建议的主线集成策略

当前最稳妥的集成顺序是：

1. 保留当前实验线为独立集成分支
2. 先把文档、baseline registry、balanced manifest、deep runner 改动整理清楚
3. 单独把 `OpenRaiser/main` 的最新提交合进来
4. 再处理冲突，尤其关注：
   - `profile.py`
   - `agents/base.py`
   - `config.py`
   - `deep_persona_runner.py`
   - `router_persona_eval.py`
5. 等 Kimi smoke 和 `full_system` 再确认一轮后，再考虑真正合入 `main`

当前不建议直接做的事：

- 直接把整个脏工作树 merge 到 `OpenRaiser/main`
- 不经筛选地提交所有实验目录生成物
- 在未验证 `final_performance` 抽取前就宣称主线已经完全稳定

## 13. 当前整理结论

截至 2026-04-13，这条 NIPS 实验线已经具备完整的代码骨架：

- 10 topic 轻量测试集
- 10 persona
- 8 个主变体
- 3 轮 chain 内演化
- per-chain memory / skill 隔离
- SDPO router 在线接入
- canonical baseline 统一 delta 计算
- SLURM 批量运行入口

但它还不适合直接无差别并入 `main`。原因不是“功能不存在”，而是：

- 分支基线落后于 `OpenRaiser/main`
- 工作树仍在实验推进中
- `final_performance` 抽取与 `full_system` 稳定性还需要继续补验证

因此当前最合理的状态定义是：

- 这是一条可运行、可继续出结果的实验集成分支
- 还不是可以无条件 fast-forward 到 `main` 的最终稳定主线
