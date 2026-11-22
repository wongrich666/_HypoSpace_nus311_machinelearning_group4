# Causal generation layer

The generation layer converts a target edge count into the existing causal
benchmark prompt and requests one candidate. The benchmark keeps parsing,
validation, graph identity and scoring. Each candidate slot records every API
attempt separately in `request_log`, including request ID, elapsed time, token
usage, errors and finish reason. A retry does not create another candidate slot.

`generate_causal_one(benchmark, llm, observations, prior_hypotheses,
target_edge_count, max_attempts=3)` returns a `GenerationResult`. Pass
`target_edge_count=None` for the history prompt baseline. The returned
`candidate` is a `CausalGraph` or `None`; the caller supplies validation
feedback. `should_stop` and `on_attempt` are optional callbacks.

DashScope uses `DASHSCOPE_API_KEY` from the environment. Set `max_tokens` and
`request_timeout` in the causal YAML config. Unknown model prices produce no
cost estimate. Add a verified `pricing` block to calculate one.

The paired prompt diagnostic runs one unconstrained and one constrained slot
per sampled observation set. The arms use the same questions and alternate
request order.

```bash
python3 causal/run_generation_pilot.py \
  --dataset causal/datasets/node04/n4_all_observations.json \
  --config causal/config/config_api.example.yaml \
  --target-edge-count 2 \
  --n-samples 1 \
  --output causal/results/generation-pilot.json
```

The output contains each candidate, the official graph hash, actual edge count,
validation result, raw response, token usage and all API attempts. The default
`--max-attempts 1` makes at most two provider requests for this one-question
example. A target count can be infeasible for a particular observation set, so
interpret this as a prompt diagnostic rather than a formal method comparison.

After generation, inspect feasibility without exposing the answer set to the
generator:

```bash
python3 causal/analyze_generation_pilot.py \
  --result causal/results/generation-pilot.json \
  --dataset causal/datasets/node04/n4_all_observations.json
```

The analyzer checks whether each target edge count occurs among the official
answers. It reports validity and target matching separately for feasible and
infeasible targets. This lookup runs after all model calls have finished.
