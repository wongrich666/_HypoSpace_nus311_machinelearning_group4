"""Boolean adaptive controller. Actions are operator counts, not answer-set lookups."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional


_ORACLE_KEYS = {
    "ground_truth", "ground_truth_expressions", "ground_truths",
    "n_compatible_expressions", "n_ground_truths", "recovery", "recovery_rate",
    "missing_answers",
}


def _as_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _reject_oracle(payload: Mapping[str, Any]) -> None:
    for key in payload:
        if key in _ORACLE_KEYS or key.startswith("ground_truth"):
            raise ValueError(f"controller must not receive oracle field {key!r}")


class BooleanAdaptiveController:
    """Fresh instance per question. Defaults match B's adaptive parameters."""

    def __init__(
        self,
        max_operator_count: int,
        *,
        seed: Optional[int] = None,
        strategy: str = "adaptive",
        forget_factor: float = 0.9,
        temperature: float = 1.0,
        explore_bonus: float = 0.15,
    ) -> None:
        if not 0.0 < forget_factor <= 1.0:
            raise ValueError("forget_factor must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if explore_bonus < 0:
            raise ValueError("explore_bonus must be non-negative")
        if strategy not in ("adaptive", "uniform", "round_robin"):
            raise ValueError("Unknown controller strategy")
        self.strategy = strategy
        self.max_operator_count = _as_nonneg_int(max_operator_count, "max_operator_count")
        self.forget_factor = float(forget_factor)
        self.temperature = float(temperature)
        self.explore_bonus = float(explore_bonus)
        self.seed = None if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self._actions = list(range(self.max_operator_count + 1))
        self._scores = {k: 0.0 for k in self._actions}
        self._counts = {k: 0 for k in self._actions}
        self.decision_log: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._scores = {k: 0.0 for k in self._actions}
        self._counts = {k: 0 for k in self._actions}
        self.decision_log = []

    def select_action(self, state: Optional[Mapping[str, Any]] = None) -> int:
        state = state or {}
        _reject_oracle(state)
        if self.strategy == "uniform":
            chosen = self._rng.choice(self._actions)
        elif self.strategy == "round_robin":
            chosen = self._actions[sum(self._counts.values()) % len(self._actions)]
        else:
            total = max(1, sum(self._counts.values()))
            logits = []
            for action in self._actions:
                bonus = self.explore_bonus * math.sqrt(
                    math.log(total + 1.0) / (self._counts[action] + 1.0)
                )
                logits.append((self._scores[action] + bonus) / self.temperature)
            shift = max(logits)
            weights = [math.exp(item - shift) for item in logits]
            draw = self._rng.random() * sum(weights)
            cursor = 0.0
            chosen = self._actions[-1]
            for action, weight in zip(self._actions, weights):
                cursor += weight
                if draw <= cursor:
                    chosen = action
                    break
        self.decision_log.append({
            "event": "select_action",
            "strategy": self.strategy,
            "target_operator_count": chosen,
            "rationale": (f"adaptive softmax forget={self.forget_factor}" if self.strategy == "adaptive" else self.strategy),
            "query_index": state.get("query_index"),
            "n_unique_valid": state.get("n_unique_valid"),
            "scores": {str(k): v for k, v in self._scores.items()},
            "counts": {str(k): v for k, v in self._counts.items()},
            "forget_factor": self.forget_factor,
            "controller_temperature": self.temperature,
            "explore_bonus": self.explore_bonus,
        })
        return chosen

    def update(self, action: Optional[int], feedback: Mapping[str, Any]) -> None:
        _reject_oracle(feedback)
        if action is None:
            self.decision_log.append({
                "event": "update", "strategy": self.strategy, "action": None,
                "rationale": "skip learning: null action",
            })
            return
        action = _as_nonneg_int(action, "action")
        if action not in self._scores:
            raise ValueError("action is outside the public operator-count bound")
        reward = 1.0 if feedback.get("is_new_valid") else 0.0
        if self.strategy == "adaptive":
            for key in self._actions:
                self._scores[key] *= self.forget_factor
            self._scores[action] += reward
        self._counts[action] += 1
        self.decision_log.append({
            "event": "update",
            "strategy": self.strategy,
            "action": action,
            "rationale": (
                f"{self.strategy} forget={self.forget_factor}: action={action} "
                f"new={bool(feedback.get('is_new_valid'))} score[{action}]={self._scores[action]:.4f}"
            ),
            "is_valid": bool(feedback.get("is_valid", False)),
            "is_new_valid": bool(feedback.get("is_new_valid", False)),
            "failure_reason": feedback.get("failure_reason"),
            "reward": reward,
            "scores": {str(k): v for k, v in self._scores.items()},
            "counts": {str(k): v for k, v in self._counts.items()},
        })
