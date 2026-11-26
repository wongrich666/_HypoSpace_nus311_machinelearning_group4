# HypoSpace · EEC 4300 / NUS311 · Group 4

**组员 / Group：** 丛嘉蔓、黄瑞祺、王韫祺、夏思

本仓库是课程期末项目：在 HypoSpace 的 **Boolean** 域上做假设空间探索，比较六种解码 / 探索方法对合法假设集合的覆盖能力（**Recovery**）。上游论文：[HypoSpace (arXiv:2510.15614)](https://arxiv.org/abs/2510.15614)。

---

## 老师请先打开 / Start here

正式证据在 **[`results/`](results/)**（数字索引）。两轮协议相同：**20 held-out 题 × 16 slots × 6 methods**（同一 `question_ids`），thinking **off**。

| 模型 | 正式目录 | 看数字 |
| --- | --- | --- |
| DeepSeek | [`results/deepseek/full_20q_six_methods/`](results/deepseek/full_20q_six_methods/) | `metrics_thinking_off.json` → `methods.<name>.recovery` |
| Qwen3.5-27B | [`results/qwen27b/full_20q_six_methods/`](results/qwen27b/full_20q_six_methods/) | `summary.json` 或目录内 `README.md` |

汇总表：[`results/README.md`](results/README.md)。更长实验笔记：[`EXPERIMENTS.md`](EXPERIMENTS.md)。

---

## 六种方法 / Six methods

| method | 一句话 |
| --- | --- |
| `memory` | 历史提示 baseline：把已生成假设写回 prompt，不读真实答案分布 |
| `adaptive` | 仅用已生成假设与验证反馈分配算子个数预算（无 oracle） |
| `adaptive_sde` | adaptive + structure-directed exploration（结构引导） |
| `memory_sde` | memory + structure-directed exploration |
| `uniform` | 在算子个数动作 `[0,1,2,3]` 上独立均匀抽样，不学习 |
| `round_robin` | 每题从 0 起循环 `0→1→2→3`，不学习 |

---

## 如何读指标 / Metrics

- **Recovery**：对可枚举合法假设集合的覆盖率（越高越好）。DeepSeek 表给 macro Recovery；Qwen 另给 micro / Recovery@16。
- **truncation**：输出被长度截断的比例。正式 Qwen 轮六方法均为 **0%**；详见 [`results/README.md`](results/README.md)。
- Validity / Uniqueness 等论文三指标说明见上游 HypoSpace；本课程正式对照以 **Recovery**（及 truncation）为主。

---

## 代码在哪 / Code (minimal)

| 路径 | 内容 |
| --- | --- |
| [`boolean/`](boolean/) | Boolean 域主代码 |
| [`boolean/modules/exploration_controller.py`](boolean/modules/exploration_controller.py) | adaptive / uniform / round_robin 控制器 |
| [`boolean/boolean_benchmark.py`](boolean/boolean_benchmark.py) | 基线 memory（及 stratified）runner |
| [`boolean/member_a_*sde*`](boolean/) | `adaptive_sde` / `memory_sde` |
| [`boolean/run_qwen_27b_full_20q.py`](boolean/run_qwen_27b_full_20q.py) | Qwen 正式 20q × 6 methods runner |
| [`boolean/experiments/`](boolean/experiments/) | 冻结协议 JSON |

`3d/`、`causal/` 为上游域代码，本课程正式结果以 Boolean + [`results/`](results/) 为准。

---

## Citation

```bibtex
@article{chen2025hypospace,
  title={HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination},
  author={Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Goyal, Anirudh and Ong, Yew-Soon and Liu, Dianbo},
  journal={arXiv preprint arXiv:2510.15614},
  year={2025}
}
```
