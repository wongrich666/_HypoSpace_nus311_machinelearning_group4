# Member C development checks, 22–24 November 2026

These are development checks using `deepseek-v4.1-flash-1017`, not the frozen
formal comparison. API request IDs were removed from the archived copies.
No API key is stored here. Original files remain in the local `causal/results`
directory.

## Paired prompt checks

`generation-pilot.json` contains the initial one-question check.
`generation-pilot-5.json` and `generation-pilot-5-edges2.json` contain the
same five sampled three-node questions with targets of one and two edges.
Each question has one unconstrained and one targeted candidate.

| Target | Target-feasible questions | Targeted valid and compliant | Unconstrained valid | Targeted tokens | Unconstrained tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 edge | 2/5 | 2/2 | 5/5 | 8,351 | 3,591 |
| 2 edges | 4/5 | 4/4 | 5/5 | 6,801 | 4,069 |

All four infeasible targeted calls across the two five-question runs reached
the configured 2,048 output-token limit. On question `n2_022`, the two-edge
target produced a valid graph different from the unconstrained candidate.
Feasibility was computed offline from the official answer set after generation.
See `causal/GENERATION_PILOT_RESULTS.md` for interpretation.

## Adaptive integration check

`smoke-adaptive-17.json`, the matching `.jsonl` candidate log and
`smoke-17.json` manifest record a one-question, three-candidate run using
`adaptive`, seed 17, one API attempt per slot, and a fixed three-slot budget.
The selected question is `n2_022`.

| Slot | Target edges | Actual edges | Valid | New valid reward | Tokens |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1 | 1 | yes | 1 | 677 |
| 2 | 0 | 0 | no | 0 | 746 |
| 3 | 0 | 0 | no | 0 | 810 |

All three responses parsed and obeyed the requested edge count. The two empty
graphs conflict with the public observations. The result has three candidate
slots and three API attempts, with paired select/update events and feedback
in each JSONL record. The reward only marks new valid graphs. The run does not
measure adaptive performance or establish the final question list or budget.
