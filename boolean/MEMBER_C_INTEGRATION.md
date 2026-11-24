# Boolean generation adapter (Member C)

`modules/generation_executor.py` accepts one operator-count action and returns
one Boolean expression candidate. It reuses the existing Boolean prompt and
response parser. It does not read the ground-truth answer set, decide whether
an expression is valid, or define expression identity.

```python
from modules.generation_executor import generate_boolean_one

target = controller.select_action(state)  # None or an allowed operator count
generation = generate_boolean_one(
    benchmark, llm, observations, all_hypotheses, target,
    max_attempts=1,
)
candidate = generation.candidate
# D: validate candidate with benchmark.validate_expression(...), then check
# novelty with benchmark.get_expression_mechanistic_key(...). Only afterwards
# call controller.update(target, feedback).
```

`generation.status` is `ok`, `api_error`, `parse_error`, `truncated`, or
`stopped`. It also contains `prompt`, `response`, `candidate`,
`target_operator_count`, `actual_operator_count`, `obeyed_operator_count`,
`usage`, `cost`, and `attempts`. `attempts` includes duration, request ID,
finish reason, token usage, and error for each provider call.

One invocation consumes one candidate slot, including failures. Retry attempts
are counted separately within that slot. The caller can stop before a new
request via `should_stop`. Configure output-token limit and request timeout on
the LLM client. The example API config now exposes `request_timeout`.

The existing `boolean_operator_complexity` convention counts each textual
AND, OR, NOT, XOR, and NOR occurrence once. The adapter uses the same count
only to report whether the requested constraint was obeyed. The official
mechanistic key remains the sole identity rule. Controller action limits
should come from the public Boolean grammar, never the per-question answer
set. D owns question selection, fixed candidate budget, reward feedback,
manifest, and final evaluation.

Run the offline check from `boolean/`:

```bash
python3 -m unittest -v test_generation_executor.py
```

These checks make no paid API calls and do not establish a Recovery gain.

For a single-question prompt preview with no API call:

```bash
python3 run_generation_pilot.py \
  --dataset datasets/boolean_extended_d2.json \
  --config config/config_api.example.yaml \
  --target-operator-count 1 --seed 17 --dry-run
```

Remove `--dry-run` to make one provider call, after setting
`DASHSCOPE_API_KEY` in the shell. The pilot prints a compact development
record; `--output` optionally saves it locally. Do not add pilot logs to the
formal results on `main`.
