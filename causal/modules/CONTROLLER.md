# Module B → D: exploration controller API

Implementation: `exploration_controller.py` (`ExplorationController`, alias `Controller`).

Actions are **edge counts** `0 .. max_edges`. There is no K-means clustering.
The controller never reads the ground-truth answer set, missing answers, or
Recovery; it only uses D's `state` and validation `feedback`.

## Construct and reset (per question)

D resets duplicate tracking per question. Do the same for B:

```python
from modules.exploration_controller import ExplorationController

controller = ExplorationController(
    max_edges=benchmark.max_edges,  # required non-negative int
    strategy="adaptive",            # "uniform" | "round_robin" | "adaptive"
    seed=33550336,                  # optional; freeze RNG for formal runs
    forget_factor=0.9,              # adaptive only: exponential score decay
    temperature=1.0,                # adaptive softmax temperature
    explore_bonus=0.15,             # adaptive under-sampled edge-count bonus
)

# Or reuse one object across questions:
controller.reset(max_edges=benchmark.max_edges, seed=33550336)
```

**RNG seed:** pass `seed` at construction or into `reset`. `seed=None` leaves a
nondeterministic stream (fine for demos; freeze an int for comparable runs).
Learning scores and the round-robin cursor are cleared on every `reset`.

## `select_action(state) -> int`

Returns a **non-negative int** target edge count for C's `generate_causal_one`.

`state` fields used (same as `CANDIDATE_FEEDBACK.md`):

| Field | Role |
| --- | --- |
| `max_edges` | If present, resizes the discrete action set |
| `query_index`, `n_queries` | Logged only |
| `n_unique_valid` | Logged only |
| `observations`, `nodes` | Ignored for action choice (no answer-set lookup) |

Oracle keys such as `ground_truth_graphs`, `missing_answers`, `recovery_rate`
raise `ValueError`.

## `update(action, feedback)`

Call after C generates and D validates. Required signal fields:

| Field | Required | Notes |
| --- | --- | --- |
| `is_valid` | yes | Observation consistency |
| `is_new_valid` **or** `is_new_valid_answer` | yes* | Novelty; `reward` accepted as fallback |
| `failure_reason` | optional | Logged into the update rationale |

\*Novelty drives adaptive learning. Duplicate-valid candidates should score 0.

`action` is the value returned by the matching `select_action` (or `None` if D
ever passes an unconstrained slot — learning is skipped).

## Strategies (shared action set)

1. **uniform** — independent uniform draws over edge counts.
2. **round_robin** — cycle `0, 1, …, max_edges, 0, …`.
3. **adaptive** — each update multiplies all edge-count scores by `forget_factor`,
   then adds `1` when the feedback marks a new valid answer. Selection is a
   temperature softmax over score + exploration bonus (no K-means).

## Decision log

Every `select_action` / `update` appends to `controller.decision_log` with
`target_edge_count` / `action` and a human-readable `rationale`.

## Wire into D's loop

```python
controller = ExplorationController(max_edges=benchmark.max_edges, strategy="adaptive", seed=42)

result = benchmark.evaluate_single_observation_set(
    llm, question,
    n_queries=32,
    max_retries=3,
    controller=controller,  # decoding stays "memory"; not stratified
)
```

Order already implemented by D: `select_action` → C generate → validate →
`update`. Create a fresh controller (or `reset`) before the next question.

## Local checks (no API key)

```powershell
cd causal
python demo_exploration_controller.py
python -m unittest discover -s . -p test_exploration_controller.py -v
```
