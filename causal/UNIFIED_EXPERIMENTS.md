# Unified causal experiments (D + B + C)

Run from the `causal` directory. The example budget below is illustrative, not
an approved paid experiment size. All five methods must reuse the same manifest.

```powershell
python run_causal_benchmark.py --dataset datasets/node03/n3_all_observations.json --config config/config_api.example.yaml --n-samples 10 --n-queries 32 --seed 42 --experiment-manifest experiments/paired.json --decoding memory --output results/paired/memory.json
```

Repeat with `--decoding uniform`, `round_robin`, `adaptive`, or `stratified`, and
use a distinct `--output` path for each. Keep all other settings identical.
Configure your actual model in the config and use environment variables for keys.
No API calls were made to choose the example budget or model.

- `--n-queries`, `--seed`, and `--experiment-manifest` are mandatory. The Python
  multi-question runner also rejects missing/nonpositive fixed budgets.
- The manifest is created on the first run and reused thereafter. It freezes
  question IDs/order, dataset SHA256, model/provider, temperature, output token
  limit, retries, timeout, seed, sample count and observation filter. Mismatches
  fail before model setup. Do not change/delete the manifest between methods.
- `--gt-filter` is rejected by this formal CLI. `--query-multiplier` is deprecated
  and has no effect. Candidate budgets never depend on the correct-answer count.
- Each question gets a fresh B controller. Its seed is derived reproducibly from
  the global seed and question ID. With no max_edges metadata, the action-space
  bound is the public DAG maximum n*(n-1)/2, not the answer-set distribution.
- The seed fixes sampling/controller randomness; it does not guarantee identical
  outputs from a remote LLM API. Model and temperature are captured in the manifest.
- Each completed candidate gets a flushed JSONL record with B's select/update
  events (rationale, scores, counts), generation attempts, usage and D feedback.
  The full result stores per-question decision_log and controller_seed as well.
  JSON object score/count keys become strings on serialization.
- Every run uses unique checkpoint and candidate-log names. This is not a resume
  implementation; never silently reuse old logs/checkpoints for a new run.
  Unexpected processing/logging failures now abort rather than silently omit a
  question from the comparison. API/parse failures remain recorded candidate slots.
- `stratified` remains oracle informed. Ground-truth answers/counts in offline
  scoring results must not be passed to B. Existing cost estimates are not actual bills.

Tests from repository root:

```powershell
python -m unittest discover -s causal -p 'test_*.py' -v
```

21 tests pass, including a CLI run of all five methods on two questions using a
fake model, per-question reset, fixed budget enforcement and manifest mismatch
rejection. B is integrated; A's future interfaces are not implicitly included.
