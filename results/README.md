# Formal results / 正式结果（数字索引）

老师入口（组员分工 + headline）见仓库根目录 [`../README.md`](../README.md)。本页只放**正式数字与路径**。

协议：两模型均为 **20 held-out questions × 16 slots × 6 methods**（同一 `question_ids`），thinking **off**。

找数字：打开对应目录的 `metrics_thinking_off.json`（DeepSeek）或 `summary.json` / `README.md`（Qwen）。

| Path | Model | Artifact |
| --- | --- | --- |
| [`deepseek/full_20q_six_methods/`](deepseek/full_20q_six_methods/) | DeepSeek thinking-off | `metrics_thinking_off.json` |
| [`qwen27b/full_20q_six_methods/`](qwen27b/full_20q_six_methods/) | Qwen3.5-27B | `summary.json` |

**读指标：** Recovery = 合法假设集合覆盖率；truncation = 长度截断率（Qwen 正式轮全为 0%）。方法释义见根 [`README.md`](../README.md)。

非正式诊断：[`../archive/interim/`](../archive/interim/)（含 DeepSeek thinking-on 中途快照；非正式对照）。基线 memory / stratified JSON：[`../boolean/results/`](../boolean/results/)。

---

## DeepSeek · thinking-off · 20q × 6 methods

Macro Recovery（`metrics_thinking_off.json` → `methods.<name>.recovery`）

| method | macro Recovery |
| --- | ---: |
| adaptive_sde | 48.00% |
| adaptive | 45.60% |
| round_robin | 43.07% |
| uniform | 38.79% |
| memory_sde | 35.82% |
| memory | 32.93% |

---

## Qwen3.5-27B · thinking-off · 20q × 6 methods

| method | macro Recovery | micro Recovery | Recovery@16 | truncation |
| --- | ---: | ---: | ---: | ---: |
| round_robin | 52.54% | 44.52% | 52.54% | 0% |
| memory | 49.48% | 40.41% | 49.48% | 0% |
| memory_sde | 46.56% | 36.30% | 46.56% | 0% |
| uniform | 45.92% | 36.99% | 45.92% | 0% |
| adaptive | 44.19% | 35.62% | 44.19% | 0% |
| adaptive_sde | 42.76% | 32.88% | 42.76% | 0% |
