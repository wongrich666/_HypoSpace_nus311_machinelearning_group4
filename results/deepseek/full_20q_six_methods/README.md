# DeepSeek Boolean · thinking-off · 20q × 6 methods

**Formal.** 20 held-out questions × 16 slots × 6 methods = 1920 slots. `enable_thinking=false`.

Numbers: `metrics_thinking_off.json` → `methods.<name>.recovery`（macro Recovery）.  
Index tables: [`../../README.md`](../../README.md).

| method | macro Recovery |
| --- | ---: |
| adaptive_sde | 48.00% |
| adaptive | 45.60% |
| round_robin | 43.07% |
| uniform | 38.79% |
| memory_sde | 35.82% |
| memory | 32.93% |

Also: `protocol.json`, `manifest.json`, `candidates.jsonl.gz`, per-method folders.
Paired Qwen run: [`../../qwen27b/full_20q_six_methods/`](../../qwen27b/full_20q_six_methods/).
