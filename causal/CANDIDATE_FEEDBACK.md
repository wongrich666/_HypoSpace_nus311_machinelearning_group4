# D: online candidate feedback and B integration

The integration is in `CausalBenchmarkEnhanced.evaluate_single_observation_set`.
It uses C's generator and the official parser, observation validator and graph
hash. The existing baseline metrics are preserved. B's real policy is not included
yet; tests exercise the interface with a fake controller and fake model.

## B's interface

Create a fresh controller per question (D resets duplicate tracking per question):

```python
class Controller:
    def select_action(self, state):
        # Return None (unconstrained) or an integer target edge count.
        return 1

    def update(self, action, feedback):
        # Learn from feedback['reward']; do not use the offline result object.
        pass
```

`state` contains `query_index` (zero-based), `n_queries`, `observations`, `nodes`,
`max_edges`, and `n_unique_valid`. The action cannot exceed `max_edges` when it is
set. Pass B a fresh policy per question, or explicitly document cross-question
training. Controller calls are synchronous. Exceptions from the controller or
logger abort the run instead of silently dropping updates or records.

`feedback` contains:

- `status`: C's generation status. `ok` means parsed, not necessarily valid.
- `is_valid`: consistency with the public observations.
- `failure_reason`: `parse_error`, `truncated_output`, `api_error`,
  `observation_mismatch`, or null for a valid candidate.
- `failure_detail`: parser exception or API error when available. A parser that
  returns None without an exception provides only the generic `parse_error`.
- `canonical_hash`: official `CausalGraph.get_hash()`, null when parsing failed.
- `is_new_valid`, `is_duplicate_valid`, `reward`: reward is 1 only for a new valid
  hash in this question, otherwise 0. Duplicate valid candidates remain valid;
  they do not have a validation failure reason.
- `target_edge_count`, `actual_edge_count`, `obeyed_edge_count`: target compliance
  is separate from validity and reward.
- `usage`, `api_attempt_count`: all attempts for this candidate, including retries.

No ground-truth set, ground-truth counts, or Recovery enters state or feedback.
The full result has offline Recovery metrics and must not be passed to B.
`stratified` remains **oracle informed** and cannot be combined with a controller.

## Completed-candidate logging

Run from `causal`, using your configured `benchmark`, `llm`, and `question`:

```python
from modules.candidate_feedback import CandidateJSONLLogger

result = benchmark.evaluate_single_observation_set(
    llm, question,
    n_queries=32,  # Example only: freeze the same count for all formal methods.
    max_retries=3,
    controller=Controller(),
    on_candidate_complete=CandidateJSONLLogger('run-candidates.jsonl'),
)
```

Omit `controller` for the baseline; the completion callback works there too.
This is a Python integration point; the CLI does not yet instantiate B's policy
or enable this JSONL sink. Existing CLI results still include the enriched
`request_log` saved with the result/checkpoint.

Order: select action → C generates → validate/hash/deduplicate → B updates →
write completed-candidate log. Each log includes question ID, query index,
candidate (serialized or null), status, prompt, response, target counts, usage,
full attempts, and feedback. Records passed to callbacks are defensive copies.
Use a distinct JSONL path per run; its parent directory must exist. This sink
appends and flushes each completed record; it is not a checkpoint/resume system.

C's `on_attempt` still reports the API attempt before parsing. Do not treat that
event as a final candidate record. The completion record contains later parser
exceptions in `attempts[*].parse_error` and `feedback.failure_detail`. A process
crash during generation cannot produce a completed-candidate record.

Every loop iteration is one fixed candidate slot, including parse failures,
invalid candidates and exhausted API retries. Retries add attempts, not slots.
The D runner has no cancellation/resume hook yet. Existing cost values are
estimates/provider values, not verified Alibaba Cloud billing.

## Verification and remaining integration

From the repository root:

```powershell
python -m unittest discover -s causal -p test_generation_pilot.py -v
python -m unittest discover -s causal -p test_candidate_feedback.py -v
```

The tests use no paid API calls. They cover retries, fixed slots, parse exceptions,
truncation, invalid observations, duplicate validity, target noncompliance,
per-question reset, feedback independence from ground truths, JSONL persistence,
and the mock select/update loop. After B submits its implementation, adapt its
action representation to this interface and repeat the fake-model integration
before a paid pilot. Freeze question IDs, model, temperature, seed and explicit
`--n-queries` across formal methods; 32 above is not a finalized experiment budget.
