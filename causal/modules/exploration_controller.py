"""Module B: exploration strategy and online learning controller.

Action space is discrete edge counts ``0 .. max_edges`` (no K-means).
The controller sees only D's online state and validation feedback — never the
ground-truth answer set, missing answers, or Recovery metrics.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence


STRATEGIES = ("uniform", "round_robin", "adaptive")


def _as_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _novelty_signal(feedback: Mapping[str, Any]) -> int:
    """Prefer D's ``is_new_valid``; accept ``is_new_valid_answer`` / ``reward``."""
    if "is_new_valid_answer" in feedback:
        return int(bool(feedback["is_new_valid_answer"]))
    if "is_new_valid" in feedback:
        return int(bool(feedback["is_new_valid"]))
    if "reward" in feedback:
        return int(bool(feedback["reward"]))
    return 0


class ExplorationController:
    """Per-question edge-count budget controller for D's candidate loop.

    Construct a fresh instance per observation set, or call :meth:`reset`
    before each question. Pass ``seed`` for reproducible RNG; ``None`` uses
    nondeterministic entropy (documented for D experiment freezes).

    Strategies share the same action set ``{0, 1, ..., max_edges}``:

    - ``uniform``: sample uniformly
    - ``round_robin``: cycle through edge counts in order
    - ``adaptive``: softmax over exponentially forgotten novelty scores
    """

    def __init__(
        self,
        max_edges: int,
        strategy: str = "adaptive",
        *,
        seed: Optional[int] = None,
        forget_factor: float = 0.9,
        temperature: float = 1.0,
        explore_bonus: float = 0.15,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not 0.0 < forget_factor <= 1.0:
            raise ValueError("forget_factor must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if explore_bonus < 0:
            raise ValueError("explore_bonus must be non-negative")
        self.strategy = strategy
        self.forget_factor = float(forget_factor)
        self.temperature = float(temperature)
        self.explore_bonus = float(explore_bonus)
        self.seed: Optional[int] = None
        self._rng = random.Random()
        self.max_edges = 0
        self._actions: List[int] = []
        self._scores: Dict[int, float] = {}
        self._counts: Dict[int, int] = {}
        self._rr_index = 0
        self.decision_log: List[Dict[str, Any]] = []
        self.reset(max_edges=max_edges, seed=seed)

    @property
    def actions(self) -> Sequence[int]:
        return tuple(self._actions)

    def reset(
        self,
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Clear per-question learning state. Call once before each observation set.

        ``seed`` reseeds the controller RNG. Omit to keep the current RNG stream
        after the first construction. Pass an explicit int to freeze a question.
        """
        if max_edges is not None:
            self.max_edges = _as_nonneg_int(max_edges, "max_edges")
        if self.max_edges < 0:
            raise ValueError("max_edges must be a non-negative int")
        if seed is not None:
            self.seed = int(seed)
            self._rng = random.Random(self.seed)
        self._actions = list(range(self.max_edges + 1))
        self._scores = {k: 0.0 for k in self._actions}
        self._counts = {k: 0 for k in self._actions}
        self._rr_index = 0
        self.decision_log = []

    def select_action(self, state: Optional[Mapping[str, Any]] = None) -> int:
        """Return a non-negative target edge count for C's generator."""
        state = state or {}
        self._reject_oracle_leak(state)
        if state.get("max_edges") is not None:
            declared = _as_nonneg_int(state["max_edges"], "state['max_edges']")
            if declared != self.max_edges:
                # Resize action set without inventing answer-set knowledge.
                self.max_edges = declared
                self._actions = list(range(self.max_edges + 1))
                for k in self._actions:
                    self._scores.setdefault(k, 0.0)
                    self._counts.setdefault(k, 0)
                self._scores = {k: self._scores[k] for k in self._actions}
                self._counts = {k: self._counts[k] for k in self._actions}
                self._rr_index %= len(self._actions)

        if self.strategy == "uniform":
            action = self._rng.choice(self._actions)
            rationale = "uniform sample over edge counts"
        elif self.strategy == "round_robin":
            action = self._actions[self._rr_index % len(self._actions)]
            self._rr_index += 1
            rationale = f"round-robin index -> edge count {action}"
        else:
            action, rationale = self._select_adaptive()

        record = {
            "event": "select_action",
            "strategy": self.strategy,
            "target_edge_count": action,
            "rationale": rationale,
            "query_index": state.get("query_index"),
            "n_unique_valid": state.get("n_unique_valid"),
            "scores": dict(self._scores),
            "counts": dict(self._counts),
        }
        self.decision_log.append(record)
        return action

    def update(self, action: Optional[int], feedback: Mapping[str, Any]) -> None:
        """Incorporate D validation feedback. Never reads ground-truth graphs."""
        self._reject_oracle_leak(feedback)
        if action is None:
            rationale = "skip learning: unconstrained / null action"
            self.decision_log.append({
                "event": "update",
                "strategy": self.strategy,
                "action": None,
                "rationale": rationale,
                "is_valid": bool(feedback.get("is_valid", False)),
                "is_new_valid_answer": bool(_novelty_signal(feedback)),
                "failure_reason": feedback.get("failure_reason"),
            })
            return

        action = _as_nonneg_int(action, "action")
        if action not in self._scores:
            # Allow mild expansion if C/D used a larger target than configured.
            self.max_edges = max(self.max_edges, action)
            self._actions = list(range(self.max_edges + 1))
            for k in self._actions:
                self._scores.setdefault(k, 0.0)
                self._counts.setdefault(k, 0)

        is_valid = bool(feedback.get("is_valid", False))
        is_new = bool(_novelty_signal(feedback))
        failure_reason = feedback.get("failure_reason")
        reward = float(is_new)

        if self.strategy == "adaptive":
            for k in self._actions:
                self._scores[k] *= self.forget_factor
            self._scores[action] += reward
            self._counts[action] += 1
            rationale = (
                f"adaptive forget={self.forget_factor}: "
                f"action={action} is_valid={is_valid} new={is_new} "
                f"score[{action}]={self._scores[action]:.4f}"
            )
            if failure_reason:
                rationale += f" failure_reason={failure_reason}"
        else:
            self._counts[action] += 1
            rationale = (
                f"{self.strategy}: recorded feedback only "
                f"(is_valid={is_valid}, new={is_new})"
            )
            if failure_reason:
                rationale += f" failure_reason={failure_reason}"

        self.decision_log.append({
            "event": "update",
            "strategy": self.strategy,
            "action": action,
            "rationale": rationale,
            "is_valid": is_valid,
            "is_new_valid_answer": is_new,
            "failure_reason": failure_reason,
            "reward": reward,
            "scores": dict(self._scores),
            "counts": dict(self._counts),
        })

    def _select_adaptive(self) -> tuple[int, str]:
        total = max(1, sum(self._counts.values()))
        logits: List[float] = []
        for k in self._actions:
            bonus = self.explore_bonus * math.sqrt(
                math.log(total + 1.0) / (self._counts[k] + 1.0)
            )
            logits.append((self._scores[k] + bonus) / self.temperature)
        # Softmax with RNG — stable even when scores are equal.
        peak = max(logits)
        weights = [math.exp(x - peak) for x in logits]
        mass = sum(weights)
        probs = [w / mass for w in weights]
        draw = self._rng.random()
        cumulative = 0.0
        action = self._actions[-1]
        for k, p in zip(self._actions, probs):
            cumulative += p
            if draw <= cumulative:
                action = k
                break
        top = sorted(
            ((k, self._scores[k], self._counts[k]) for k in self._actions),
            key=lambda row: (-row[1], row[0]),
        )[:3]
        rationale = (
            "adaptive softmax over forgotten novelty scores; "
            f"top={top}; chose {action}"
        )
        return action, rationale

    @staticmethod
    def _reject_oracle_leak(payload: Mapping[str, Any]) -> None:
        forbidden = (
            "ground_truth_graphs",
            "ground_truths",
            "gt_hashes",
            "missing_answers",
            "missing_graphs",
            "recovery_rate",
            "recovery",
            "answer_set",
            "all_answers",
        )
        for key in forbidden:
            if key in payload:
                raise ValueError(
                    f"controller must not receive oracle field {key!r}"
                )


# Backwards-friendly alias matching D's interface sketch.
Controller = ExplorationController
