# NanoResearch NIPS 实验 Setting、SDPO 训练说明与 Case Study

更新时间：2026-04-12

本文档用于系统记录当前 NanoResearch 重复评测实验的设置、SDPO router 的训练与在线接入方式、指标定义、以及 `pilot_nlp_biomed_qa` 这一道题目的 case study。文档同时回答一个当前结果表里的关键问题：为什么同一个 topic 下，有些结果有 `delta_over_baseline`，有些没有，甚至有些数值彼此不一致。

## 1. 这轮实验在做什么

本轮实验是一个基于固定 research question test set 的 NanoResearch 重复评测协议。每个测试问题都提供：

- 所属领域
- 难度设定
- 背景描述
- 常用 baseline
- 标准数据集
- 用户要求

测试时，不是让系统“回答问题”，而是让系统：

1. 针对给定 research question 生成一个新的 idea
2. 生成与该 idea 对应的 experiment / implementation plan
3. 让下游代码实现子系统尝试把 plan 落成可执行实验
4. 统计 ideation、对齐、实现、性能等多个指标

本轮协议的核心目的是比较不同 router / memory / skill / SDPO 组合，对研究想法生成与下游可执行性的影响。

## 2. 实验协议总览

### 2.1 实验展开方式

协议由以下三个维度做笛卡尔展开：

| 维度 | 当前设置 |
| --- | --- |
| Persona 数量 | 10 |
| Question 数量 | 2 |
| Main 变体数量 | 8 |
| Appendix baseline 数量 | 1 |

因此：

- 主实验总数 = `10 x 2 x 8 = 160`
- 如果把 appendix-only 的 `context_informed_generation` 也算上，总 assignment 数 = `10 x 2 x 9 = 180`

### 2.2 10 个 persona

| Persona ID | 设计意图 |
| --- | --- |
| `ai4science_journal_conservative` | 偏保守、重控制变量、重科学严谨性 |
| `ai4science_reproducibility_first` | 优先可复现、可重跑、设置可审计 |
| `benchmark_maximalist_conference` | 追求更广 benchmark 覆盖和更强比较 |
| `cv_fast_iteration_builder` | 偏快速迭代、实现实用性强 |
| `cv_visual_benchmark_heavy` | 偏 CV 风格、重 benchmark 和展示 |
| `journal_evidence_first_writer` | 证据优先、写作风格克制 |
| `multimodal_systems_engineer` | 偏系统设计、模块边界清晰 |
| `nlp_conference_exploratory` | 更愿意探索新想法，但仍要求可证伪 |
| `nlp_conference_pragmatic` | 偏务实、干净 ablation、实现友好 |
| `resource_constrained_repro_first` | 资源受限、强调轻量和复现 |

### 2.3 8 个主变体

| 变体名 | Memory | Skill | SDPO Router | 说明 |
| --- | --- | --- | --- | --- |
| `base_router` | 否 | 否 | 否 | 纯基础 router |
| `memory_only` | 是 | 否 | 否 | 仅开启 memory/self-evolution |
| `skill_only` | 否 | 是 | 否 | 仅开启 skill/self-evolution |
| `sdpo_only` | 否 | 否 | 是 | 仅切到训练后的 SDPO router |
| `memory_skill` | 是 | 是 | 否 | memory + skill |
| `memory_sdpo` | 是 | 否 | 是 | memory + SDPO |
| `skill_sdpo` | 否 | 是 | 是 | skill + SDPO |
| `full_system` | 是 | 是 | 是 | 全系统 |

### 2.4 Appendix-only 变体

| 变体名 | 是否进主表 | 作用 |
| --- | --- | --- |
| `context_informed_generation` | 否 | 作为附录 baseline，对照“给更多上下文但不引入自演化/SDPO”的情形 |

## 3. 当前测试题集合

### 3.1 Question 清单

| Question ID | Domain | Difficulty | Datasets | 用户要求摘要 |
| --- | --- | --- | --- | --- |
| `pilot_nlp_biomed_qa` | NLP | `incremental_innovation` | `PubMedQA` | 生成轻量、可 ablation、可复现的方法与 plan |
| `pilot_multimodal_efficiency` | Multimodal | `nontrivial_recomposition` | `MMMU`, `ScienceQA` | 提出系统感强、可 benchmark、不过度复杂的方法 |

### 3.2 本文采用的 case study

本文的详细 case study 使用：

- Persona: `ai4science_journal_conservative`
- Topic: `pilot_nlp_biomed_qa`

选择它的原因是：

1. 这是当前结果覆盖最好的组之一
2. 任务本身轻量，适合看 idea -> code -> metric 的整条链路
3. 它恰好暴露了当前 `delta_over_baseline` 定义不一致的问题，适合作为方法学案例

## 4. 当前这轮 OpenAlex batch 的实际运行 setting

### 4.1 Batch 基本信息

| 项目 | 当前值 |
| --- | --- |
| Batch 根目录 | `/mnt/dhwfile/raise/user/xujinhang/nanoresearch/fullgrid_allvariants_openalex_20260412_174400` |
| 提交日期 | 2026-04-12 |
| SLURM 并发 shard 数 | 16 |
| 实际 selected assignments | 180 |
| 当前已写出 `result.json` 数量 | 14 |
| 当前 job 名 | `nr_oa01` 到 `nr_oa16` |

这里的 180 个 assignment，正好对应：

- 10 persona
- 2 question
- 8 个 main 变体
- 1 个 appendix 变体

### 4.2 运行配置

| 配置项 | 当前值 |
| --- | --- |
| `execution_profile` | `cluster_full` |
| `slurm_partition` | `belt_road` |
| `slurm_quota_type` | `auto` |
| `slurm_default_time` | `12:00:00` |
| `slurm_max_gpus` | `1` |
| `ideation_disable_retrieval` | `true` |
| `execution_auto_repair_enabled` | `false` |
| `router_sdpo_model_path` | `/mnt/dhwfile/raise/user/xujinhang/nanoresearch/tmp/router_sdpo_offpolicy_runs/router_sdpo_offpolicy_exact_20260410_0130/train/epoch-02` |
| `router_sdpo_temperature` | `0.0` |
| `router_sdpo_max_new_tokens` | `256` |
| `router_sdpo_timeout` | `180.0` |
| `skip_stages` | `FIGURE_GEN`, `WRITING`, `REVIEW` |

### 4.3 关于 `slurm_quota_type=auto`

配置代码中对 `slurm_quota_type` 的注释写的是：

- `auto / reserved / spot`
- 其中 `auto` 的语义是“优先 reserved，不行则回落到 spot”

因此，当前 batch 的资源配额策略不是强制 reserve，而是自动选择。

### 4.4 当前这轮 retrieval setting 的真实含义

这里有一个容易误解的点：

- 配置文件里仍然是 `ideation_disable_retrieval=true`
- 这意味着“完整 literature retrieval 被关闭”
- 但当前 `ideation.py` 已经修改为：在 eval-fast 模式下，仍执行一个“轻量 OpenAlex baseline retrieval”

因此当前 ideation 阶段的真实行为不是：

- 完全不检索

而是：

- 不做完整 literature survey
- 但会根据 topic 中声明的 baselines / datasets / problem statement，向 OpenAlex 发起少量 baseline-oriented query
- 把查到的 baseline paper / evidence 注入 ideation 输出，作为 baseline context

这一步的定位是：

- 给 planner 一个“最低限度的 baseline grounding”
- 不是恢复完整文献综述
- 主要用于减少 baseline 数值缺失、减少 hallucinated baseline 的概率

## 5. Deep pipeline 的评测链路

当前深度评测 runner 的主链路为：

1. `IDEATION`
2. `PLANNING`
3. `SETUP`
4. `CODING`
5. `EXECUTION`
6. `ANALYSIS`

显式跳过的阶段为：

- `FIGURE_GEN`
- `WRITING`
- `REVIEW`

注意这里的 `REVIEW` 指论文写作后的审稿/修文阶段，不影响当前用于打分的 alignment / novelty judge。后两者是在 runner 中额外调用 review model 做 JSON judgment。

## 6. 每个指标到底是什么意思

这一节尽量按“代码真实计算方式”解释，而不是按论文口径泛化解释。

### 6.1 指标总表

| 指标名 | 类型 | 当前实现中的真实来源 | 越大越好吗 |
| --- | --- | --- | --- |
| `novelty_score` | 主观 | review model 对 idea 相对 baseline 的主观打分 | 是 |
| `alignment_pass_at_1` | 主客观混合 | review model 判断 idea/plan 是否符合用户要求 | 是 |
| `alignment_token_to_pass` | 客观 | IDEATION + PLANNING 阶段累计 token，直到对齐通过 | 否 |
| `plan_executability` | 客观 | 是否同时拿到了 blueprint 和 experiment_output | 是 |
| `implementation_success` | 客观 | execution result contract / status 是否满足成功条件 | 是 |
| `implementation_token_to_runnable` | 客观 | `SETUP + CODING + EXECUTION` token，总前提是实现成功 | 否 |
| `final_performance` | 客观 | 从 primary metric 提取出的最终实验性能 | 是，取决于 metric 方向 |
| `baseline_performance` | 半客观 | 从 blueprint 中 baseline 列表提取的“最佳基线数值” | 是，取决于 metric 方向 |
| `delta_over_baseline` | 派生 | `final_performance - baseline_performance` 或反向 | 是 |
| `total_tokens_from_method_to_code` | 客观 | `IDEATION + PLANNING + SETUP + CODING + EXECUTION` 总 token | 否 |

### 6.2 `novelty_score`

定义：

- 由 review model 比较“生成的方法”与“题目里给定的 baselines”后给出
- 同时会输出 `closest_baseline`

重要限制：

- 当前 novelty prompt 只要求“打分并给 closest_baseline”，没有在 prompt 里明确定义一个严格的 numeric rubric
- 因此 `2.0`、`3.0` 这类分值可以做同一评测 setup 内的相对比较
- 但不能把它当成一个绝对、可跨 setup 标定的量纲

结论：

- 这是一个“相对可比、绝对不可过分解读”的 LLM 主观指标

### 6.3 `alignment_pass_at_1`

定义：

- review model 读取题目要求、selected hypothesis、blueprint method、ablation groups 后，判断当前输出是否满足用户要求

注意：

- runner 支持 alignment retry
- 同时会累计 `alignment_token_to_pass`
- 因此从语义上讲，真正更稳的客观指标其实是 `alignment_token_to_pass`

额外风险：

- 如果 review 调用失败，alignment judge 的 fallback 默认是 `pass_at_1=true`
- 也就是说，这个指标在极端情况下可能偏乐观

因此：

- 论文主文若要强调客观性，建议更倚重 `alignment_token_to_pass`
- `alignment_pass_at_1` 可以保留，但要说明它依赖 LLM judge

### 6.4 `alignment_token_to_pass`

定义：

- 在 runner 的实现里，这个值是 `IDEATION + PLANNING` 两个阶段 token 的累计和
- 如果第一次不通过、第二次才通过，则会累计多轮尝试的 token

意义：

- 它是“达到与用户要求一致所消耗的代价”
- 越小越好

它比 `alignment_pass_at_1` 更客观，因为它不只是一个布尔值，而是记录为通过对齐所支付的真实 token 成本。

### 6.5 `plan_executability`

定义：

- 只要存在 blueprint 且存在 experiment_output，就记为 `true`

这个指标比较宽松：

- 它更接近“系统成功产出了一份可执行计划并进入执行阶段”
- 不等于实验真的完全跑通

### 6.6 `implementation_success`

定义：

- runner 会检查 execution output / result contract / final status
- 如果命中成功状态，或者 `partial + success_path + satisfied_signals` 满足一定条件，也会记为成功

这意味着：

- `implementation_success=true` 并不要求整个 execution status 必须是一个非常干净的 `COMPLETED`
- 当前不少结果里你会看到 `experiment_status="partial"`，但 `implementation_success=true`

这是代码当前的明确定义，不是数据错误。

### 6.7 `implementation_token_to_runnable`

定义：

- 只有当 `implementation_success=true` 时才填值
- 数值等于 `SETUP + CODING + EXECUTION` 三个阶段的 token 总和

意义：

- 这是把方法真正落成 runnable experiment 所需的 token 成本

### 6.8 `final_performance`

定义：

- runner 先从 blueprint 中选 primary metric
- 再从 `analysis_output.final_metrics` 或 `execution_output.metrics` 中取对应数值

在当前 `pilot_nlp_biomed_qa` case 里：

- primary metric 是 `accuracy`
- 因此 `final_performance` 就是最终实验准确率

这也解释了你之前问的那个问题：

- `final_performance` 不是 delta
- 它是当前任务 primary metric 的“原始最终结果”

### 6.9 `baseline_performance`

定义：

- runner 不会去查一个 task-level 的全局 baseline 表
- 它会直接读取“该 variant 自己的 blueprint 里列出来的 baseline 数值”
- 然后取与 primary metric 同名的最佳 baseline 值

因此这个字段的性质是：

- 不是全局 canonical baseline
- 而是 variant-local blueprint baseline

这正是后面 delta 不一致的根源。

### 6.10 `delta_over_baseline`

定义：

- 若 metric 是 higher-is-better，则 `delta = final - baseline`
- 若 metric 是 lower-is-better，则 `delta = baseline - final`

另外还有一个小修正：

- 若一边看起来像百分比，另一边看起来像小数，会先做一次 scale normalization

理论上它应该代表：

- 相对 baseline 的性能增益

但在当前实现里，它只能代表：

- “相对该 variant 自己 blueprint 中 baseline 的增益”

这跟“相对同一题统一 baseline 的增益”不是一回事。

### 6.11 `total_tokens_from_method_to_code`

定义：

- 等于 `IDEATION + PLANNING + SETUP + CODING + EXECUTION`

意义：

- 从提出方法到代码落地的总 token 成本
- 是你想衡量“整个 research-to-code 链路开销”时最直接的客观量

## 7. 为什么同一个 topic，有些有 Delta，有些没有，甚至还不一致

这是当前结果解释里最关键的问题。

### 7.1 概念上应该怎样

如果是“同一个 topic、同一个 primary metric、同一个 baseline 集合”，那么直觉上是对的：

- 它们应该共享一个 baseline 参照
- 因而每个变体都应该能计算 delta
- 而且 delta 应该相对同一个 baseline 来算

### 7.2 当前代码实际上怎样

当前 pipeline 不是这么实现的。

它的做法是：

1. 每个 variant 先各自生成自己的 `experiment_blueprint.json`
2. `baseline_performance` 从这个 blueprint 的 `baselines[*].expected_performance` 里提取
3. 如果该 blueprint 没给出可解析 numeric baseline，则 `baseline_performance=null`
4. 于是 `delta_over_baseline=null`

结果就是：

- 同题不同变体可能拿到不同 baseline
- 同题不同变体可能有的有 delta，有的没有
- 同题不同变体甚至可能 baseline 数值互相冲突

### 7.3 这个问题在 `pilot_nlp_biomed_qa` 上的直接证据

| 变体 | Blueprint 中 baseline 情况 | 原始 `baseline_performance` | 问题 |
| --- | --- | --- | --- |
| `base_router` | BioBERT / PubMedBERT / instruction-tuned baseline 都是 `N/A` | `null` | 没法算 delta |
| `memory_only` | BioBERT `0.68`，PubMedBERT `0.724` | `0.724` | 这一行相对合理 |
| `skill_only` | baseline 基本都是 `N/A` | `null` | 没法算 delta |
| `sdpo_only` | BioBERT `0.78`，PubMedBERT `0.81`，另有 distilled baseline `0.76` | `0.81` | 与其他变体明显不一致，疑似偏离 canonical baseline |
| `memory_skill` | baseline 基本都是 `N/A` | 尚无同 batch result | 即使跑完，也很可能仍拿不到 raw delta |
| `memory_sdpo` | BioBERT `0.68`，PubMedBERT `0.72`，instruction-tuned teacher `0.78` | 尚无同 batch result | 有数值，但和 `0.724` 也不完全一致 |
| `skill_sdpo` | baseline 基本都是 `N/A` | `null` | 没法算 delta |
| `full_system` | baseline 基本都是 `N/A` | `null` | 没法算 delta |

### 7.4 结论

所以答案是：

- 你说的“同一个 topic 应该共享一个 baseline”在研究设计上是对的
- 当前 raw pipeline 的 `delta_over_baseline` 没有真正做到这一点

它现在更像一个：

- “blueprint-local delta”

而不是一个：

- “task-level aligned delta”

## 8. 本文档对 Delta 的处理原则

为了既保留原始结果，又能让 case study 可比，本文档采用双轨记录：

### 8.1 Raw Delta

直接保留 `result.json` 里的：

- `baseline_performance`
- `delta_over_baseline`

优点：

- 忠实反映 pipeline 原始输出

缺点：

- 不可直接做同题横向比较

### 8.2 Aligned Delta

本文另加一个人工对齐列：

- `Aligned Delta (shared baseline = 0.724)`

其中共享 baseline 选用：

- `PubMedBERT accuracy = 0.724`

原因：

1. 它出现在同一 OpenAlex batch 的 `memory_only` blueprint 中
2. 该 blueprint 的 evidence summary 对这组数值描述最一致
3. 对 `pilot_nlp_biomed_qa` 来说，它是最合理的“task-level canonical baseline 候选”
4. 它对应的是该题中最强、也最自然的文本基线之一：PubMedBERT on PubMedQA

因此本文档中：

- `Raw Delta` 用于审计原始 pipeline 行为
- `Aligned Delta` 用于做同 topic、同 metric 的横向比较

注意：

- 本文不会回写任何 `result.json`
- 对齐只发生在文档展示层

## 9. SDPO 是怎么训练的

这一节只写当前已经实际存在、并已被在线接入的那套 SDPO router，不写假设方案。

### 9.1 SDPO 的角色

当前 SDPO 训练的不是“实验模型本体”，而是 router：

- 输入：任务上下文 + candidate memory + candidate skills
- 输出：一个结构化路由决策 JSON

输出字段固定为：

| 字段 | 含义 |
| --- | --- |
| `selected_memory_ids` | 选择哪些 memory |
| `selected_skill_ids` | 选择哪些 skill |
| `prompt_plan` | 用一句很短的话概括后续提示/执行计划 |
| `update_memory` | 反馈后是否更新 memory |
| `update_skill` | 反馈后是否更新 skill |

也就是说，SDPO 学的是：

- 在多轮 research workflow 里，router 怎么更好地做 memory/skill 选择与更新决策

### 9.2 SDPO 数据来源

当前 off-policy SDPO 训练脚本使用了 9 个输入目录，都是 2026-04-09 左右导出的 live router multiturn traces。

这些 trace 会先经过导出、清洗、去重，再形成训练 manifest。

### 9.3 导出后的数据统计

| 统计项 | 数值 |
| --- | --- |
| `raw_rows` | 716 |
| `deduped_rows` | 600 |
| `duplicate_rows` | 116 |
| `final_rows` | 579 |

#### 按 subsystem 统计

| Subsystem | 条数 |
| --- | --- |
| `code_implementation` | 288 |
| `method_generation` | 150 |
| `paper_writing` | 141 |

#### 按 persona 统计

| Persona | 条数 |
| --- | --- |
| `ai4science_journal_conservative` | 60 |
| `ai4science_reproducibility_first` | 60 |
| `benchmark_maximalist_conference` | 60 |
| `cv_fast_iteration_builder` | 60 |
| `cv_visual_benchmark_heavy` | 60 |
| `journal_evidence_first_writer` | 60 |
| `multimodal_systems_engineer` | 59 |
| `nlp_conference_exploratory` | 60 |
| `nlp_conference_pragmatic` | 60 |
| `resource_constrained_repro_first` | 40 |

#### Prompt / completion 长度统计

| 项目 | count | min | max | mean |
| --- | --- | --- | --- | --- |
| `base_prompt_tokens` | 579 | 1486 | 2221 | 2039.21 |
| `hindsight_prompt_tokens` | 579 | 3389 | 4823 | 4244.77 |
| `completion_tokens` | 579 | 92 | 170 | 123.45 |

#### 截断统计

| 项目 | 数值 |
| --- | --- |
| `max_prompt_length` | 2048 |
| `max_completion_length` | 2048 |
| `base_prompt_over_length` | 371 |
| `hindsight_prompt_over_length` | 579 |

这说明：

- hindsight prompt 全部超过 2048，需要截断
- base prompt 也有相当比例被截断

因此当前 SDPO 的学习目标本质上是在“截断后的上下文”上学习 route policy，而不是在完整上下文上学习。

### 9.4 数据清洗时丢弃的样本

drop report 里记录的是 candidate id 合法性问题：

| 项目 | 数值 |
| --- | --- |
| `y0_selected_memory_ids_outside_candidates` | 9 |
| `y0_selected_skill_ids_outside_candidates` | 3 |
| `y1_selected_memory_ids_outside_candidates` | 11 |
| `y1_selected_skill_ids_outside_candidates` | 13 |

这意味着：

- 一部分 router 轨迹中的选择结果，不在当时提供的 candidate 集合内
- 这类样本被排除，以保证训练标签和候选集合是一致的

### 9.5 模型初始化与训练入口

| 项目 | 当前值 |
| --- | --- |
| Base model | `Qwen/Qwen3-8B` |
| Base model path | `/mnt/dhwfile/raise/user/xujinhang/data/modelscope/models/Qwen/Qwen3-8B` |
| 训练脚本 | `scripts/run_router_sdpo_offpolicy.sh` |
| 核心训练代码 | `tools/train_router_sdpo_offpolicy.py` |
| 导出脚本 | `tools/export_router_sdpo_offpolicy.py` |
| 最终 checkpoint | `/mnt/dhwfile/raise/user/xujinhang/nanoresearch/tmp/router_sdpo_offpolicy_runs/router_sdpo_offpolicy_exact_20260410_0130/train/epoch-02` |

### 9.6 训练超参数

| 超参数 | 数值 |
| --- | --- |
| `num_epochs_requested` | 2 |
| `optimizer_steps_completed` | 38 |
| `total_elapsed_seconds` | 283.24 |
| `learning_rate` | `2e-6` |
| `warmup_ratio` | `0.05` |
| `per_device_batch_size` | `1` |
| `gradient_accumulation_steps` | `4` |
| `world_size` | 8 GPUs |
| `global_batch_size` | 32 |
| `weight_decay` | 0.0 |
| `max_grad_norm` | 1.0 |
| `optimizer` | `AdamW8bit` |
| `attn_implementation` | `sdpa` |
| `dtype` | `bfloat16` |
| 分布式方式 | FSDP `FULL_SHARD` |

### 9.7 训练时的工程策略

当前 SDPO 训练还启用了：

- FSDP full shard
- activation checkpointing
- model gradient checkpointing
- AdamW8bit
- cosine warmup scheduler
- bf16 mixed precision

目标很明确：

- 用 8 卡把一个 8B router 以较低显存成本跑起来

### 9.8 Gate run

正式两轮训练之前，还有一个 8-GPU gate run：

- `num_epochs=1`
- `max_optimizer_steps=1`
- `gate_only`

作用：

- 检查 SDPO loss 的 advantage 是否全为 0
- 如果全为 0，说明 base / hindsight 对比没有信息增益，训练配置可能有问题

### 9.9 当前 SDPO 的训练目标

每条训练样本都包含两份 prompt：

- `base_messages`
- `hindsight_messages`

以及同一个 target completion。

训练时会分别计算：

- completion 在 base context 下的 token log-prob
- completion 在 hindsight context 下的 token log-prob

然后定义 token 级别 advantage：

`advantage = hindsight_logprob - base_logprob`

在代码里这个 advantage 是 detach 的，也就是：

- 不对 hindsight 分支回传梯度
- 只用它作为 base logprob 的权重

损失可以写成一个直观形式：

`L_SDPO = - sum_t advantage_t * log p_base(y_t)`

再对样本求平均。

直观理解是：

- 如果 hindsight context 让某个 token 更容易被解释出来，那么对应 advantage 为正
- 训练就鼓励 base router 所对应的输入上下文，也朝着能更好支持这个 completion 的方向更新

它不是标准 DPO 的 preference pair 形式，而是一个：

- 基于 base / hindsight completion likelihood 差值的 exact off-policy SDPO

### 9.10 在线推理时怎么接入

当前评测 batch 中，`sdpo_only / memory_sdpo / skill_sdpo / full_system` 这些变体，都会把：

- `same_router_hindsight_sdpo_enabled = true`

然后在 adaptive context 构建时启用同一个 SDPO router。

在线推理时：

- 若配置了 `router_sdpo_model_path`，则直接本地加载 HF checkpoint
- 本地可用 CUDA 时优先上 GPU
- `temperature=0`
- `max_new_tokens=256`

因此当前 batch 不是“prompt 假装 SDPO”，而是真正在调用训练后的 router checkpoint。

## 10. 当前 case study：`pilot_nlp_biomed_qa`

### 10.1 原始题面

| 字段 | 内容 |
| --- | --- |
| `question_id` | `pilot_nlp_biomed_qa` |
| `domain` | `NLP` |
| `difficulty` | `incremental_innovation` |
| `background` | Biomedical QA systems already perform well on PubMedQA, but lightweight improvements that preserve reproducibility and clean ablations remain valuable. |
| `problem_statement` | Design a new but practical method for improving PubMedQA under limited compute while keeping the implementation easy to reproduce. |
| `baselines` | `BioBERT`, `PubMedBERT`, `instruction-tuned biomedical QA baseline` |
| `datasets` | `PubMedQA` |
| `user_requirements` | Generate a new idea and an implementation-oriented plan. Keep the method lightweight, ablatable, and reproducible. |
| `extra_context` | Prefer methods that fit within a modest single-node budget and can be compared fairly against standard biomedical QA baselines. |

### 10.2 为什么这个题适合做 case

这个题的关键词是：

- lightweight
- ablatable
- reproducible
- modest single-node budget

因此非常适合看 NanoResearch 在“研究想法生成”和“下游真正实现”之间有没有对齐。

它不是一个追求极高 novelty 的题，而是一个非常典型的 NeurIPS/ACL/MLSys 风格：

- 要求新意适中
- 但必须能落地、能比较、能复现

### 10.3 当前同 batch 覆盖情况

以下表格只看：

- Persona: `ai4science_journal_conservative`
- Topic: `pilot_nlp_biomed_qa`
- Batch: `fullgrid_allvariants_openalex_20260412_174400`

| 变体 | 同 batch 状态 | 是否已有同 batch `result.json` | 备注 |
| --- | --- | --- | --- |
| `base_router` | completed | 是 | 可直接使用 |
| `memory_only` | completed | 是 | 可直接使用 |
| `skill_only` | completed | 是 | 可直接使用 |
| `sdpo_only` | completed | 是 | 可直接使用 |
| `memory_skill` | running / unresolved | 否 | 当前 OpenAlex batch 尚未出结果；另有旧 batch 历史结果可作补充参考 |
| `memory_sdpo` | running / unresolved | 否 | 当前 OpenAlex batch 尚未出结果；未找到可比历史结果 |
| `skill_sdpo` | completed | 是 | 可直接使用 |
| `full_system` | completed | 是 | 可直接使用 |

因此：

- 严格意义上的“同 batch 完整 8 变体横向比较”目前还不能做
- 当前只能做 6 个变体的同 batch 比较
- `memory_skill` 可以给一个“历史补充值”
- `memory_sdpo` 目前只能留空

## 11. Case study 主结果表

### 11.1 主结果表：保留 raw 字段，同时加入 aligned delta

共享 baseline 设为：

- `0.724`，对应 `PubMedBERT accuracy on PubMedQA`

| 变体 | 状态 | Novelty | Align Pass | Align Token | Impl Success | Impl Token | Final Perf | Raw Baseline | Raw Delta | Aligned Delta (0.724) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_router` | completed | 2.0 | true | 8643 | true | 38314 | 0.560 |  |  | -0.164 |
| `memory_only` | completed | 2.0 | true | 14056 | true | 47065 | 0.600 | 0.724 | -0.124 | -0.124 |
| `skill_only` | completed | 2.0 | true | 8868 | true | 39073 | 0.552 |  |  | -0.172 |
| `sdpo_only` | completed | 2.0 | true | 7157 | true | 37790 | 0.470 | 0.810 | -0.340 | -0.254 |
| `memory_skill` | running |  |  |  |  |  |  |  |  |  |
| `memory_sdpo` | running |  |  |  |  |  |  |  |  |  |
| `skill_sdpo` | completed | 2.0 | true | 9032 | true | 39655 | 0.552 |  |  | -0.172 |
| `full_system` | completed | 3.0 | true | 10770 | true | 39942 | 0.660 |  |  | -0.064 |

### 11.2 Token 补充表

| 变体 | Total Tokens From Method To Code | 备注 |
| --- | --- | --- |
| `base_router` | 46957 | 同 batch 完整 |
| `memory_only` | 61121 | 同 batch 完整 |
| `skill_only` | 47941 | 同 batch 完整 |
| `sdpo_only` | 44947 | 同 batch 完整 |
| `memory_skill` |  | 同 batch 未完成 |
| `memory_sdpo` |  | 同 batch 未完成 |
| `skill_sdpo` | 48687 | 同 batch 完整 |
| `full_system` | 50712 | 同 batch 完整 |

### 11.3 当前能读出的初步现象

仅从现有 6 个同 batch 完成结果看：

1. `full_system` 目前是这一组里 `final_performance` 最好的，达到 `0.660`
2. 但即使是 `full_system`，相对共享 baseline `0.724` 的 aligned delta 仍是 `-0.064`
3. `sdpo_only` 单独使用时在这个 case 上表现最弱，aligned delta 为 `-0.254`
4. `memory_only` 的 raw delta 和 aligned delta 完全一致，因为它恰好拿到了合理的 raw baseline `0.724`
5. `base_router / skill_only / skill_sdpo / full_system` 虽然都能产出 performance，但 raw delta 为空，因为各自 blueprint 没给出可解析 baseline 数字

这说明：

- 当前系统已经能稳定产出“可运行且有 metric 的实验”
- 但 baseline grounding 还没有完全对齐
- 因而不能直接拿 raw delta 做严肃的横向比较

### 11.4 当前展示案例的运行顺序与 memory 可见性

这一小节专门回答一个更关键的问题：

- 这 8 个变体到底是按什么顺序跑的
- 谁看了谁的 memory

先说结论：

- 这组 case 里，不存在一个可以严肃声明的“同 batch 串行传染链”，即不能简单写成 “`memory_skill` 看了 `memory_only` 的 memory”
- 但存在明确的全局历史 memory 污染，即 `memory_only / memory_skill / memory_sdpo / full_system` 在当前 batch 中读到了更早时间就已经写入 `~/.nanoresearch/memory` 的历史记录
- 因此，这一组 case 不能作为严格隔离条件下的 memory 对比证据

运行顺序这里用 `workspaces/attempt-01` 目录的创建时间作为近似代理：

| 顺序 | 变体 | `attempt-01` 创建时间 | 是否在 ideation 阶段注入 memory |
| --- | --- | --- | --- |
| 1 | `base_router` | `2026-04-12 18:10:58` | 否 |
| 2 | `full_system` | `2026-04-12 18:12:52` | 是 |
| 3 | `skill_only` | `2026-04-12 18:15:11` | 否 |
| 4 | `memory_only` | `2026-04-12 18:16:31` | 是 |
| 5 | `skill_sdpo` | `2026-04-12 18:22:43` | 否 |
| 6 | `sdpo_only` | `2026-04-12 18:37:16` | 否 |
| 7 | `memory_sdpo` | `2026-04-12 21:36:20` | 是 |
| 8 | `memory_skill` | `2026-04-12 21:37:36` | 是 |

但“运行得更早”不等于“后者就一定读了前者刚写出的 memory”。当前日志能支持的说法更保守一些：

| 变体 | 是否读 memory | 看到的 memory 类型 | 当前能确认的来源 |
| --- | --- | --- | --- |
| `base_router` | 否 | 无 | 无 |
| `skill_only` | 否 | 无 | 无 |
| `sdpo_only` | 否 | 无 | 无 |
| `skill_sdpo` | 否 | 无 | 无 |
| `full_system` | 是 | `project_context` + `promising_direction` | 读到了当前 batch 开始前已经存在的同题同 persona 历史 memory |
| `memory_only` | 是 | `promising_direction` + `project_context` + `decision_history` | 读到了当前 batch 开始前已经存在的同题同 persona 历史 memory；还读到了至少一条跨 persona memory |
| `memory_skill` | 是 | `promising_direction` + `project_context` + `decision_history` | 与 `memory_only` 同类，读到了历史全局 memory；还包含至少一条跨 persona memory |
| `memory_sdpo` | 是 | `project_context` + `promising_direction` | 读到了当前 batch 开始前已经存在的同题同 persona 历史 memory |

更具体地说，当前 case 中能明确定位到的历史 memory 包括：

| Memory ID | 时间戳 | 类型 | 说明 |
| --- | --- | --- | --- |
| `mem-0dfee50fc6a7` | `2026-04-12T09:47:38Z` | `project_context` | 同题 `pilot_nlp_biomed_qa`、同 conservative persona 的 ideation 输出 |
| `mem-c0d6e08657e9` | `2026-04-12T09:48:49Z` | `project_context` | 同题、同 conservative persona 的 planning blueprint |
| `mem-863308d5fa80` | `2026-04-12T09:47:38Z` | `decision_history` | 同题、同 conservative persona 的 ideation 决策摘要 |
| `mem-e4dc0fe93307` | `2026-04-12T09:48:49Z` | `decision_history` | 同题、同 conservative persona 的 planning 约束摘要 |
| `rmem-16d6dafedbcbaa` | `2026-04-12T09:47:38Z` | `promising_direction` | 同题、同 conservative persona 的 ideation promising direction |
| `rmem-143a3c5778e843` | `2026-04-11T16:42:51Z` | `promising_direction` | 同题、同 conservative persona 的更早一轮 planning promising direction |
| `rmem-d0a2001129e5c1` | `2026-04-12T08:52:24Z` | `promising_direction` | 同题、同 conservative persona 的历史 planning promising direction |
| `rmem-de455e862d2a60` | `2026-04-12T08:53:37Z` | `promising_direction` | 同题、同 conservative persona 的历史 planning promising direction |
| `mem-68db5fdefe6d` | `2026-04-12T10:41:30Z` | `project_context` | 同题但不同 persona，说明发生了跨 persona memory 泄漏 |

因此，“谁看了谁的 memory”在当前展示案例里应当这样表述：

- `base_router / skill_only / sdpo_only / skill_sdpo` 没看 memory
- `full_system` 看了历史全局 memory，但当前证据不足以证明它直接看了这 8 个变体里某一个前序运行刚写出的 memory
- `memory_only` 看了历史全局 memory，而且还看到了跨 persona memory
- `memory_skill` 看了历史全局 memory，而且还看到了跨 persona memory
- `memory_sdpo` 看了历史全局 memory，但当前证据同样不足以证明它直接看了这 8 个变体里某一个前序运行刚写出的 memory

换句话说，这一组不是一个干净的：

- `A -> 写 memory -> B 读取 A 的 memory -> C 再读取 B 的 memory`

而是一个被全局历史缓存污染过的：

- `历史运行 -> 全局 ~/.nanoresearch/memory`
- `当前 batch 中的若干 memory 变体 -> 从这个全局池子里读历史内容`

这个问题已经在代码侧修复。修复后，新的 rerun 会把 adaptive `memory` 和 `skills` 默认写到每个 assignment 自己的 `NANORESEARCH_HOME` 下，而不是共享 `~/.nanoresearch/...`。因此，后续重跑时才可以严肃回答“哪个变体看了哪个变体的 memory”。

## 12. `memory_skill` 与 `memory_sdpo` 能不能补

### 12.1 `memory_skill`

可以部分补。

已经找到一个历史结果：

- Batch: `fullgrid_allvariants_auto_20260412_1646`
- Persona: `ai4science_journal_conservative`
- Topic: `pilot_nlp_biomed_qa`
- Variant: `memory_skill`

对应结果如下：

| 来源 | Novelty | Align Token | Impl Token | Final Perf | Raw Baseline | Raw Delta | Aligned Delta (0.724) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 历史 no-retrieval batch | 3.0 | 12405 | 44847 | 0.560 | 0.720 | -0.160 | -0.164 |

但这条历史结果只能作为：

- 补充参考

不能作为：

- 与当前 OpenAlex 同 batch 完全等价的主结果

原因是：

1. 它不在当前 OpenAlex batch 中
2. retrieval setting 不同
3. raw baseline 也不是完全相同的 task-level canonical baseline

### 12.2 `memory_sdpo`

目前补不了。

原因：

1. 当前 OpenAlex batch 还没出 `result.json`
2. 在已有历史目录中，没有找到同 persona + 同 topic + 同 variant 的可比较历史结果

因此，`memory_sdpo` 在本文档中必须留空，并明确标注“暂无足够数据”。

## 13. 为什么这两个没出结果

对当前 OpenAlex batch 的 `memory_skill` 和 `memory_sdpo`，从 `debug_round_*.json` 可以看到，它们目前主要卡在下游任务侧的 agent-debug 过程中，而不是 case study 汇总逻辑本身。

### 13.1 `memory_skill` 当前暴露过的问题

已经出现过的故障包括：

- `packaging` 依赖/metadata 异常，导致 `transformers` import 失败
- PubMedQA 数据解析逻辑过严，导致 test examples 读空
- student model 本地路径存在但没有实际 checkpoint 文件
- teacher model 本地目录是一个不完整的分片 checkpoint，缺 shard

### 13.2 `memory_sdpo` 当前暴露过的问题

也出现过类似问题：

- `packaging` 依赖检测异常
- PubMedQA JSON 解析不匹配
- student model 本地目录无权重文件
- teacher model 本地路径缺失 shard，导致 `FileNotFoundError`

### 13.3 这说明什么

这说明当前缺失结果的原因不是：

- case study 表格没整理好

而是：

- 这两个 assignment 的下游 agent-debug 链路还没有收敛到一个最终可记录的 `result.json`

## 14. 对“是否每个变体都有足够数据”的结论

### 14.1 严格按同 batch、同 topic、同 persona

答案是：没有。

当前只有以下 6 个变体具备足够的同 batch 数据：

- `base_router`
- `memory_only`
- `skill_only`
- `sdpo_only`
- `skill_sdpo`
- `full_system`

### 14.2 若允许历史补充

可以补上：

- `memory_skill`

但仍然补不上：

- `memory_sdpo`

### 14.3 因此当前 case 的完整性结论

| 范围 | 覆盖度 |
| --- | --- |
| 严格同 batch 8 变体 | 6 / 8 |
| 允许历史补充后 | 7 / 8 |

所以现在还不能说：

- 这个 topic 的 8 个主变体都已经齐了

最多只能说：

- 6 个已齐
- 1 个可历史补
- 1 个仍缺失

## 15. 写论文时建议怎么用这些结果

### 15.1 主文建议

对于同 topic 的横向比较，建议主文里使用：

- `Aligned Delta`

而不是直接使用 raw `delta_over_baseline`。

原因很简单：

- raw delta 当前不是共享 baseline 下的可比量

### 15.2 附录建议

附录中保留：

- raw `baseline_performance`
- raw `delta_over_baseline`
- 对应 blueprint baseline 摘要

这样评审能看清楚：

- 原始 pipeline 实际输出了什么
- 我们又是如何把它对齐成可比较表的

### 15.3 指标解释建议

在论文指标段建议明确写：

1. `Novelty` 为 LLM-as-a-judge 的主观创新性评分
2. `Alignment Token-to-Pass` 衡量达到用户需求一致所需的 token 成本
3. `Implementation Token-to-Runnable` 衡量生成可执行代码所需 token 成本
4. `Final Performance` 为任务 primary metric
5. `Delta over Baseline` 在最终论文表格中采用 task-level aligned delta，而非 blueprint-local raw delta

## 16. 当前文档版结论

截至 2026-04-12，这份文档可以支持以下结论：

1. 当前实验 setting 已经明确：10 persona、2 题、8 主变体、1 appendix 变体，总共 180 个 assignment
2. 当前这轮 batch 使用的是 `cluster_full + SLURM + quota_type=auto + 轻量 OpenAlex baseline retrieval + 关闭 execution auto-repair`
3. SDPO 不是占位逻辑，而是一个真实训练好的 Qwen3-8B router checkpoint，已经在线接入
4. 当前 `delta_over_baseline` 的不一致不是 topic 本身的问题，而是 pipeline 当前把 baseline 定义成了 variant-local blueprint 字段
5. 对 `pilot_nlp_biomed_qa` 这个 case，应该同时保留 raw delta 和 aligned delta，其中 aligned delta 才适合做同题横向比较
6. 当前同 batch 已完成 6 个主变体，`memory_skill` 只能历史补，`memory_sdpo` 暂时补不了

## 17. 后续待补项

以下内容等结果出来后继续补到本文档即可：

| 待补项 | 当前状态 |
| --- | --- |
| `memory_skill` 同 OpenAlex batch 正式结果 | 待补 |
| `memory_sdpo` 同 OpenAlex batch 正式结果 | 待补 |
| case study 完整 8-way same-batch 表 | 待补 |
| 更大范围的 macro-average 汇总表 | 待补 |
| 是否将 raw delta 逻辑正式替换为 task-level aligned delta | 待决策 |

---

## 附：本文件里对结果列的使用约定

为避免后续继续混淆，这里固定约定：

- `Final Perf`：任务 primary metric 原始值
- `Raw Baseline`：`result.json` 中原样读取的 `baseline_performance`
- `Raw Delta`：`result.json` 中原样读取的 `delta_over_baseline`
- `Aligned Delta`：文档层人工统一 baseline 后重新计算的 delta

其中：

- 空白不代表“这个变体理论上没有该指标”
- 只代表“截至当前版本，缺少足够可靠的数据来填写”
