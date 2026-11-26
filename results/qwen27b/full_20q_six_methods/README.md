# Qwen3.5-27B formal Boolean six-method (20q)

**Status: COMPLETE** — 20 held-out questions × 16 slots × 6 methods = 1920 slots. Truncation = **0** everywhere.

## Config

| Setting | Value |
| --- | --- |
| protocol | `qwen27b-boolean-main-six-methods-thinking-off-v1` |
| paired DeepSeek protocol | `boolean-main-six-methods-thinking-off-v1` |
| provider | DashScope compatible API |
| base_url | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| model | `qwen3.5-27b` |
| enable_thinking | **false** |
| temperature | 0.7 |
| max_tokens | 4096 |
| seed | 42 |
| dataset | `boolean_extended_d2` |
| budget | 20 × 16 = 320 slots / method |
| runner | `boolean/run_qwen_27b_full_20q.py` |

Same 20 `question_ids` as [`../../deepseek/full_20q_six_methods/`](../../deepseek/full_20q_six_methods/).

## Recovery (all six)

| method | macro Recovery | micro Recovery | Recovery@16 | truncation rate |
| --- | ---: | ---: | ---: | ---: |
| **round_robin** | **52.54%** | 44.52% | 52.54% | **0.00%** |
| memory | 49.48% | 40.41% | 49.48% | **0.00%** |
| memory_sde | 46.56% | 36.30% | 46.56% | **0.00%** |
| uniform | 45.92% | 36.99% | 45.92% | **0.00%** |
| adaptive | 44.19% | 35.62% | 44.19% | **0.00%** |
| adaptive_sde | 42.76% | 32.88% | 42.76% | **0.00%** |

Ranking: **round_robin > memory > memory_sde ≳ uniform > adaptive > adaptive_sde**.

## Artifacts

- `summary.json` — metrics for all methods
- `protocol.json` / `manifest.json` / `question_manifest.json`
- `qwen-27b-full-20q-16-<method>.json` — per-method bundles
- per-question logs under `<method>/`
