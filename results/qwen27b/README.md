# Qwen3.5-27B 开发探针

与 `results/deepseek/` 并列。均为 frozen 开发协议下的 **5 题 × 16 slots** 探针（非 DeepSeek 主实验 formal）。

| 路径 | 方法 |
| --- | --- |
| [`three_method_probe_v1/`](three_method_probe_v1/) | memory / adaptive / adaptive_sde（原 `boolean/results/qwen27b_three_method_probe_v1`） |
| [`remaining_three_v1/`](remaining_three_v1/) | uniform / round_robin / memory_sde（原 `boolean/results/qwen27b_remaining_three_v1`） |
| [`six_method_combined_summary.json`](six_method_combined_summary.json) | 六方法合并摘要 |

重新跑探针时请使用新的 `--out` 目录（例如 `results/qwen27b/<new_run_id>`），不要覆盖历史 manifest。
