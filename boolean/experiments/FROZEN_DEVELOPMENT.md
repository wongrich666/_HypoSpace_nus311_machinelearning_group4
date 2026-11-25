# Boolean four-method development freeze: low / 4096

The authoritative configuration is `development_budget.json`; controller semantics are in
`adaptive_strategy_manifest.json`. This freezes the existing five development questions,
not a held-out final evaluation. The agreed approach is to version configuration and the
actual implementation together; historical output manifests are never rewritten.

From the repository root (DASHSCOPE_API_KEY must already be in the environment):

```powershell
python boolean/run_frozen_development.py --out D:/MLfinalproject/boolean_frozen_four_methods_v3 --n-queries 16 --methods memory uniform round_robin adaptive --dry-run
python boolean/run_frozen_development.py --out D:/MLfinalproject/boolean_frozen_four_methods_v3 --n-queries 16 --methods memory uniform round_robin adaptive
python boolean/run_frozen_development.py --out D:/MLfinalproject/boolean_frozen_four_methods_v3 --n-queries 16 --methods adaptive
```

Use `--questions 0 2 3 4` to select a subset; indices follow question_ids in the config.
Dry run validates data, freezes a manifest and source snapshot, and sends no API requests.
Resume only against the same manifest and source version. The new entry point has its
own source fingerprint: do not copy old logs into it or overwrite historical manifests.
An existing 4096 question may be included in a later offline comparison after confirming
identical prompts, core-source snapshot, settings, validator and slot rules; record its
original provenance explicitly. Reusing it is not automatic in this launcher.

Each method has 80 candidate slots, total 320; at most 960 API attempts (not tokens).
Truncation, parse failures and invalid outputs consume a slot; only transient transport
failures retry within that slot, at most three attempts total. No extra replacement slot.
Only final content is parsed, never reasoning_content. Actual requests, token usage,
feedback, failures, target/actual operator counts and controller decisions are logged.

Model: deepseek-v4.1-flash-1017, DashScope compatible endpoint, temperature 0.7,
reasoning_effort low, max_tokens 4096, timeout 60 seconds per request.
enable_thinking is omitted, not explicitly disabled. Seed 42 controls local question
selection/controller randomness; the client does not send a provider seed, so API
generation is not guaranteed bitwise reproducible. A fresh controller is created per
question with a seed derived from [42, question ID]. Actions are 0,1,2,3 from public
depth-two unary/binary grammar. NOT counts as an operator; variables and parentheses do not.
Adaptive: forget_factor .9, controller_temperature 1.0, explore_bonus .15; reward is one
for a new valid answer and zero otherwise, with no additional truncation penalty.
The optional forget_factor=1.0 ablation is not enabled in this protocol.

Validation uses public syntax/depth constraints and observations; identity uses the
official mechanistic key, including the NOR key for NOT(x OR y). Ground truth is not
passed to adaptive. stratified uses true answer complexity distributions and remains
oracle-informed. Recovery is scored offline against ground truth.

Report mean and per-question Recovery, Recovery@4/8/16, truncation, invalidity,
duplicates, attempts, tokens and duration. Prices in client cost estimates are not
verified provider bills. Keep high-effort/2048/8192 runs separate. Use unseen frozen
questions for final evaluation after this development phase.

## Four-method revision

This revision implements the requested memory / uniform / round_robin / adaptive
comparison in one runner. Uniform independently samples the four public actions with
equal probability; round_robin cycles 0,1,2,3 and resets at each question. Neither learns
from rewards. All four use the same generator, validator, history and slot accounting.
Adaptive parameters are unchanged. stratified remains available in boolean_benchmark.py
as an optional oracle-informed comparison, outside this 320-slot budget.
The prior three-method config is archived as development_budget_three_methods_v1.json.

Start with --questions 0 for all four methods. Inspect each q0_METHOD.jsonl for
finish_reason=length, status=truncated, and attempts lengths (must be <=3); verify
16 completed rows per method. Then use --questions 1 2 3 4 with the same output
directory/config/source. Do not silently change settings between batches. API errors
after three attempts stop the launcher. Inspect them before resuming.

## Validator correction (v3)

For extended_d2, candidates must also have an official mechanistic key in the
34-expression space enumerated from public grammar metadata. This corrects the
acceptance of NOT x OR y OR (x AND y); no per-question ground truths are used.
Both _in_space and validate_expression enforce this check. Other grammars retain
their existing checks. Actions 0..3 and adaptive rewards are unchanged.
Use a new output directory and manifest; old adaptive trajectories cannot be
corrected by offline rescoring when incorrect rewards affected later actions.
