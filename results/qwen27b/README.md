# Qwen3.5-27B Boolean 结果

与 `results/deepseek/` 并列。

| 路径 | 内容 |
| --- | --- |
| [`full_20q_six_methods/`](full_20q_six_methods/) | **正式** held-out 20 题 × 16 slots × 六方法（配对 DeepSeek thinking-off 题集） |
| [`three_method_probe_v1/`](three_method_probe_v1/) | 开发探针：memory / adaptive / adaptive_sde（5 题） |
| [`remaining_three_v1/`](remaining_three_v1/) | 开发探针：uniform / round_robin / memory_sde（5 题） |
| [`six_method_combined_summary.json`](six_method_combined_summary.json) | 5 题探针六方法合并摘要 |

正式轮 runner：`boolean/run_qwen_27b_full_20q.py`，协议：`boolean/experiments/qwen27b_full_20q_six_methods_v1.json`。  
重新跑请用新的 `--out` 目录，不要覆盖历史 manifest。
