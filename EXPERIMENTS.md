# 实验记录

## Group4 结果目录（老师入口）

| 路径 | 内容 |
| --- | --- |
| [`results/`](results/) | **正式表**：DeepSeek + Qwen 各 6×20q |
| [`results/deepseek/full_20q_six_methods/`](results/deepseek/full_20q_six_methods/) | DeepSeek thinking-off 正式轮 |
| [`results/qwen27b/full_20q_six_methods/`](results/qwen27b/full_20q_six_methods/) | Qwen3.5-27B 正式轮 |
| [`archive/interim/`](archive/interim/) | Boolean interim 诊断归档 |
| [`archive/probes/qwen27b-5q/`](archive/probes/qwen27b-5q/) | Qwen 5q 开发探针（非正式） |
| [`boolean/results/memory|stratified`](boolean/results/) | DeepSeek-v4 基线 JSON（未搬迁） |

---

模型 `deepseek-v4.1-flash-1017`，种子 `33550336`，`query_multiplier` 1.0。提问次数等于该题合法假设个数。命令仍是 `--decoding memory` 或 `--decoding stratified`。

- memory：历史提示 baseline。每问一次，把已经生成的假设写回 prompt，不使用真实答案的复杂度分布。
- stratified：真实答案分布指导的复杂度分层解码（论文风格复现）。总次数不变，按该题真实答案的复杂度直方图分配名额。Boolean 数算子，3D 数占格，因果图数边。
- adaptive：尚未实现。将来只根据已经生成的假设和验证反馈学习预算分配，不读取真实答案分布。

已完成的 stratified 结果文件还没有下面这组字段，但用的就是这套分配。新结果会写入：

```json
{
  "decoding": "stratified",
  "method_label": "Oracle-informed complexity-stratified decoding",
  "allocation_source": "ground_truth_complexity_histogram",
  "uses_ground_truth_distribution": true
}
```

## 已完成

### Boolean — DeepSeek thinking-off 正式六方法（2026-11-26）

输出：[`results/deepseek/full_20q_six_methods/`](results/deepseek/full_20q_six_methods/)。Macro Recovery 来自 `metrics_thinking_off.json`。

| 方法 | macro Recovery |
|---|---:|
| adaptive_sde | 0.4800 |
| adaptive | 0.4560 |
| round_robin | 0.4307 |
| uniform | 0.3879 |
| memory_sde | 0.3582 |
| memory | 0.3293 |

### Boolean — Qwen3.5-27B 正式六方法（2026-11-26）

协议 `qwen27b-boolean-main-six-methods-thinking-off-v1`：与 DeepSeek thinking-off **同一 20 道 held-out 题**，`enable_thinking=false`，temp 0.7，max_tokens 4096，seed 42，16 slots/题。输出：[`results/qwen27b/full_20q_six_methods/`](results/qwen27b/full_20q_six_methods/)。Truncation 全方法为 0。

| 方法 | macro Recovery | micro Recovery | Recovery@16 | truncation |
|---|---:|---:|---:|---:|
| round_robin | 0.5254 | 0.4452 | 0.5254 | 0 |
| memory | 0.4948 | 0.4041 | 0.4948 | 0 |
| memory_sde | 0.4656 | 0.3630 | 0.4656 | 0 |
| uniform | 0.4592 | 0.3699 | 0.4592 | 0 |
| adaptive | 0.4419 | 0.3562 | 0.4419 | 0 |
| adaptive_sde | 0.4276 | 0.3288 | 0.4276 | 0 |

### Boolean

| 数据 | 解码 | 题数 | 题库 | Validity | Uniqueness | Recovery | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| 简单档，`basic`，深度 2 | memory | 10 | 35 | 0.980 | 0.980 | 0.980 | [boolean/results/memory/boolean_2var_deepseek-v4.json](boolean/results/memory/boolean_2var_deepseek-v4.json) |
| 简单档，`basic`，深度 2 | stratified | 10 | 35 | 1.0 | 1.0 | 1.0 | [boolean/results/stratified/boolean_2var_deepseek-v4.json](boolean/results/stratified/boolean_2var_deepseek-v4.json) |
| 扩展档，`extended`，深度 2 | memory | 10 | 78 | 0.767 | 0.754 | 0.688 | [boolean/results/memory/boolean_extended_d2_deepseek-v4.json](boolean/results/memory/boolean_extended_d2_deepseek-v4.json) |
| 扩展档，`extended`，深度 2 | stratified | 10 | 78 | 0.824 | 0.819 | 0.819 | [boolean/results/stratified/boolean_extended_d2_deepseek-v4.json](boolean/results/stratified/boolean_extended_d2_deepseek-v4.json) |
| 完整档，`full`，深度 2 | memory | 10 | 80 | 0.446 | 0.387 | 0.387 | [boolean/results/memory/boolean_full_d2_deepseek-v4.json](boolean/results/memory/boolean_full_d2_deepseek-v4.json) |
| 完整档，`full`，深度 2 | stratified | 10 | 80 | 0.456 | 0.470 | 0.436 | `boolean/results/stratified/boolean_full_d2_deepseek-v4.json`（**文件缺失，待补**） |

### 3D

| 数据 | 解码 | 题数 | 题库 | Validity | Uniqueness | Recovery | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| 3×3，高度 3，恰好 1 块 | memory | 9 | 9 | 1.0 | 1.0 | 1.0 | `3d/results/...`（**目录未入库，待补**） |
| 3×3，高度 3，恰好 1 块 | stratified | 9 | 9 | 0.963 | 1.0 | 0.963 | `3d/results/...`（**目录未入库，待补**） |

### 因果图

| 数据 | 解码 | 题数 | 题库 | Validity | Uniqueness | Recovery | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| 3 个节点 | memory | 10 | 61 | 1.0 | 1.0 | 1.0 | `causal/results/...`（**目录未入库，待补**） |
| 3 个节点 | stratified | 10 | 61 | 1.0 | 1.0 | 1.0 | `causal/results/...`（**目录未入库，待补**） |
| 4 个节点 | memory | 10 | 847 | 0.956 | 0.946 | 0.946 | `causal/results/...`（**目录未入库，待补**） |
| 4 个节点 | stratified | 10 | 847 | 0.977 | 0.967 | 0.967 | `causal/results/...`（**目录未入库，待补**） |


## Token 与费用


| 数据          | 解码         | 输入      | 输出      | 合计      | 费用（元） |
| ----------- | ---------- | ------- | ------- | ------- | ----- |
| Boolean 简单档 | memory     | 34,317  | 74,273  | 108,590 |       |
| Boolean 简单档 | stratified | 28,332  | 40,596  | 68,928  |       |
| Boolean 扩展档 | memory     | 77,644  | 209,974 | 287,618 |       |
| Boolean 扩展档 | stratified | 92,981  | 243,364 | 336,345 |       |
| Boolean 完整档 | memory     | 225,206 | 642,833 | 868,039 |       |
| Boolean 完整档 | stratified | 247,783 | 693,546 | 941,329 |       |
| 3D，恰好 1 块   | memory     | 21,238  | 15,476  | 36,714  |       |
| 3D，恰好 1 块   | stratified | 21,168  | 9,840   | 31,008  |       |
| 因果图，3 个节点   | memory     | 6,562   | 17,551  | 24,113  |       |
| 因果图，3 个节点   | stratified | 7,087   | 12,946  | 20,033  |       |
| 因果图，4 个节点   | memory     | 79,623  | 241,542 | 321,165 |       |
| 因果图，4 个节点   | stratified | 74,433  | 190,837 | 265,270 |       |
| 合计          |            | 916,374 | 2,392,778 | 3,309,152 |       |


## 已生成、尚未评测

下面两份数据可以用。若按种子 `33550336` 抽 10 题，每种解码的提问次数如下。


| 数据                  | 题库  | 10 题的提问次数 | 文件                                                                                                 |
| ------------------- | --- | --------- | -------------------------------------------------------------------------------------------------- |
| 3D，恰好 2 块           | 36  | 90        | [3d/datasets/3d_grid3_h3_b2.json](3d/datasets/3d_grid3_h3_b2.json)                                 |
| 3D，恰好 3 块           | 84  | 270       | [3d/datasets/3d_grid3_h3_b3.json](3d/datasets/3d_grid3_h3_b3.json)                                 |


