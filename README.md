# HypoSpace · EEC 4300 / NUS311 · Group 4

本仓库是课程期末项目：在 HypoSpace 的 **Boolean** 域上做假设空间探索，比较六种解码 / 探索方法对合法假设集合的覆盖能力（**Recovery**）。上游论文：[HypoSpace (arXiv:2510.15614)](https://arxiv.org/abs/2510.15614)。

---

## 老师请先打开 / Start here

正式证据在 **[`results/`](results/)**（数字索引）。两轮正式协议相同：**20 held-out 题 × 16 slots × 6 methods**（同一 `question_ids`），thinking **off**。完整表见 [`results/README.md`](results/README.md)。

---

## 组员与分工 / Team & roles

| 成员 | 学号 | 角色 |
| --- | --- | --- |
| **A** HuangRuiqi 黄瑞祺 | CEG26019 | SDE / official mechanism identity / structure guidance |
| **B** CongJiaman 丛嘉蔓 | CEG26006 | adaptive exploration controller / interpretation |
| **C** WangYunqi 王韫祺 | CEG26058 | generation & API layer / protocol |
| **D** Xiasi 夏思 | CEG26064 | validation, Recovery scoring, results |

闭环贡献（与展示一致）：**A** 结构引导（SDE）与机制身份；**B** 探索控制器与结果解读；**C** 生成与 API / 协议；**D** 校验、Recovery 计分与结果归档。

---

## 实验划分 / Experiment division

- **共享正式协议：** 六方法 × 20 held-out 题 × 16 slots（同一 `question_ids`）。  
- **正式结果落点（thinking off）：**  
  - DeepSeek → [`results/deepseek/full_20q_six_methods/`](results/deepseek/full_20q_six_methods/)  
  - Qwen3.5-27B → [`results/qwen27b/full_20q_six_methods/`](results/qwen27b/full_20q_six_methods/)  
- 仓库未单独标注「谁在哪一天跑了哪次 API」；以角色贡献 + `results/` 归档为准，不臆造 runner 归属。  
- DeepSeek **thinking-on** 中途快照（非正式）：[`archive/interim/boolean-20261126/`](archive/interim/boolean-20261126/)（`reasoning_effort=low`；部分共同题）。

---

## 主要结果（headline）/ Main results

Macro Recovery。正式表以 [`results/README.md`](results/README.md) 为准。

| 设置 | 领先 | 要点 |
| --- | --- | --- |
| DeepSeek thinking-on（interim，n=11 共同题） | memory **89.47%** | 六方法均 **>80%**；见 archive interim，非 `results/` 正式轮 |
| DeepSeek thinking-off（正式 20q） | **adaptive_sde 48.00%** | adaptive 45.60% · round_robin 43.07% · uniform 38.79% · memory_sde 35.82% · memory 32.93% |
| Qwen3.5-27B thinking-off（正式 20q） | **round_robin 52.54%** | memory 49.48% · memory_sde 46.56% · uniform 45.92% · adaptive 44.19% · adaptive_sde 42.76%（truncation 全 0%） |

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
- **truncation**：输出被长度截断的比例。正式 Qwen 轮六方法均为 **0%**。
- Validity / Uniqueness 等见上游 HypoSpace；本课程正式对照以 **Recovery**（及 truncation）为主。

---

## 代码在哪 / Code (minimal)

| 路径 | 内容 |
| --- | --- |
| [`boolean/`](boolean/) | Boolean 域主代码 |
| [`boolean/modules/exploration_controller.py`](boolean/modules/exploration_controller.py) | adaptive / uniform / round_robin 控制器（B） |
| [`boolean/boolean_benchmark.py`](boolean/boolean_benchmark.py) | 基线 memory（及 stratified）runner |
| [`boolean/member_a_*sde*`](boolean/) | `adaptive_sde` / `memory_sde`（A） |
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
