"""Minimal fake-feedback demo for Module B (no LLM / no API keys)."""

from modules.exploration_controller import ExplorationController


def fake_feedback(is_valid: bool, is_new: bool, failure_reason=None):
    return {
        "is_valid": is_valid,
        "is_new_valid_answer": is_new,
        "is_new_valid": is_new,
        "reward": int(is_new),
        "failure_reason": failure_reason,
        "status": "ok" if is_valid or failure_reason == "observation_mismatch" else "parse_error",
    }


def run_rounds(strategy: str, max_edges: int = 3, seed: int = 7):
    print(f"\n=== strategy={strategy!r} max_edges={max_edges} seed={seed} ===")
    controller = ExplorationController(max_edges=max_edges, strategy=strategy, seed=seed)
    # Scripted outcomes: mix of invalid, duplicate-valid, and new-valid.
    script = [
        fake_feedback(False, False, "observation_mismatch"),
        fake_feedback(True, True),
        fake_feedback(True, False),  # duplicate valid
        fake_feedback(False, False, "parse_error"),
        fake_feedback(True, True),
        fake_feedback(True, True),
    ]
    for i, fb in enumerate(script):
        state = {
            "query_index": i,
            "n_queries": len(script),
            "max_edges": max_edges,
            "n_unique_valid": sum(1 for e in controller.decision_log
                                  if e.get("event") == "update" and e.get("is_new_valid_answer")),
            "observations": [{"string": "Perturb A: B=1"}],
            "nodes": ["A", "B", "C", "D"],
        }
        action = controller.select_action(state)
        controller.update(action, fb)
        select = controller.decision_log[-2]
        update = controller.decision_log[-1]
        print(
            f"q={i} select->{action} ({select['rationale']}) | "
            f"update ({update['rationale']})"
        )
    print(f"actions chosen: {[e['target_edge_count'] for e in controller.decision_log if e['event'] == 'select_action']}")
    print(f"final scores: {controller._scores}")


def main():
    for strategy in ("uniform", "round_robin", "adaptive"):
        run_rounds(strategy)
    # Per-question reset demo
    ctrl = ExplorationController(max_edges=2, strategy="adaptive", seed=1)
    a1 = [ctrl.select_action({"query_index": i, "max_edges": 2}) for i in range(3)]
    for a in a1:
        ctrl.update(a, fake_feedback(True, True))
    ctrl.reset(max_edges=2, seed=1)
    a2 = [ctrl.select_action({"query_index": i, "max_edges": 2}) for i in range(3)]
    print(f"\nreset reproducibility: first={a1} after_reset={a2} match={a1 == a2}")


if __name__ == "__main__":
    main()
