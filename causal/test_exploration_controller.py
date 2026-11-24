"""Unit tests for Module B exploration controller (no paid API calls)."""

import unittest
from unittest.mock import patch

from modules.exploration_controller import ExplorationController, STRATEGIES
from run_causal_benchmark import CausalBenchmarkEnhanced


class ExplorationControllerTest(unittest.TestCase):
    def test_strategies_share_edge_count_action_set(self):
        for strategy in STRATEGIES:
            ctrl = ExplorationController(max_edges=2, strategy=strategy, seed=0)
            self.assertEqual(list(ctrl.actions), [0, 1, 2])
            for i in range(6):
                action = ctrl.select_action({"query_index": i, "max_edges": 2})
                self.assertIsInstance(action, int)
                self.assertFalse(isinstance(action, bool))
                self.assertGreaterEqual(action, 0)
                self.assertLessEqual(action, 2)
                ctrl.update(action, {"is_valid": True, "is_new_valid_answer": i % 2 == 0})

    def test_round_robin_cycles(self):
        ctrl = ExplorationController(max_edges=2, strategy="round_robin", seed=0)
        seen = [ctrl.select_action({"query_index": i}) for i in range(5)]
        self.assertEqual(seen, [0, 1, 2, 0, 1])

    def test_uniform_respects_seed(self):
        a = ExplorationController(max_edges=4, strategy="uniform", seed=99)
        b = ExplorationController(max_edges=4, strategy="uniform", seed=99)
        seq_a = [a.select_action({}) for _ in range(10)]
        seq_b = [b.select_action({}) for _ in range(10)]
        self.assertEqual(seq_a, seq_b)

    def test_adaptive_prefers_rewarding_edge_counts(self):
        ctrl = ExplorationController(
            max_edges=2, strategy="adaptive", seed=3,
            forget_factor=0.5, temperature=0.2, explore_bonus=0.0,
        )
        # Train: only edge count 1 yields new valid answers.
        for _ in range(12):
            action = ctrl.select_action({"max_edges": 2})
            new = action == 1
            ctrl.update(action, {
                "is_valid": new,
                "is_new_valid": new,
                "failure_reason": None if new else "observation_mismatch",
            })
        picks = [ctrl.select_action({"max_edges": 2}) for _ in range(20)]
        self.assertGreaterEqual(picks.count(1), 10)

    def test_forgetting_decays_idle_scores(self):
        ctrl = ExplorationController(
            max_edges=1, strategy="adaptive", seed=0, forget_factor=0.5,
        )
        ctrl.update(0, {"is_valid": True, "is_new_valid_answer": True})
        self.assertEqual(ctrl._scores[0], 1.0)
        ctrl.update(1, {"is_valid": False, "is_new_valid_answer": False})
        self.assertAlmostEqual(ctrl._scores[0], 0.5)
        self.assertAlmostEqual(ctrl._scores[1], 0.0)

    def test_reset_clears_learning_and_reseeds(self):
        ctrl = ExplorationController(max_edges=2, strategy="adaptive", seed=5)
        first = [ctrl.select_action({"query_index": i}) for i in range(4)]
        for a in first:
            ctrl.update(a, {"is_valid": True, "is_new_valid_answer": True})
        self.assertTrue(any(v > 0 for v in ctrl._scores.values()))
        ctrl.reset(max_edges=2, seed=5)
        self.assertTrue(all(v == 0 for v in ctrl._scores.values()))
        second = [ctrl.select_action({"query_index": i}) for i in range(4)]
        self.assertEqual(first, second)

    def test_rejects_oracle_fields_in_state_and_feedback(self):
        ctrl = ExplorationController(max_edges=1, strategy="uniform", seed=0)
        with self.assertRaises(ValueError):
            ctrl.select_action({"ground_truth_graphs": []})
        with self.assertRaises(ValueError):
            ctrl.update(0, {"is_valid": True, "is_new_valid": True, "recovery_rate": 1.0})
        with self.assertRaises(ValueError):
            ctrl.update(0, {"is_valid": True, "missing_answers": []})

    def test_decision_log_records_choice_and_rationale(self):
        ctrl = ExplorationController(max_edges=1, strategy="round_robin", seed=0)
        action = ctrl.select_action({"query_index": 0})
        ctrl.update(action, {
            "is_valid": False,
            "is_new_valid_answer": False,
            "failure_reason": "parse_error",
        })
        self.assertEqual(ctrl.decision_log[0]["event"], "select_action")
        self.assertEqual(ctrl.decision_log[0]["target_edge_count"], action)
        self.assertIn("rationale", ctrl.decision_log[0])
        self.assertEqual(ctrl.decision_log[1]["event"], "update")
        self.assertEqual(ctrl.decision_log[1]["failure_reason"], "parse_error")
        self.assertIn("rationale", ctrl.decision_log[1])

    def test_accepts_d_is_new_valid_alias(self):
        ctrl = ExplorationController(max_edges=1, strategy="adaptive", seed=0)
        ctrl.update(1, {"is_valid": True, "is_new_valid": True})
        self.assertEqual(ctrl.decision_log[-1]["is_new_valid_answer"], True)
        self.assertGreater(ctrl._scores[1], 0)

    def test_benchmark_integration_with_fake_model(self):
        benchmark = CausalBenchmarkEnhanced()
        benchmark.nodes = ["A", "B"]
        benchmark.max_edges = 1
        question = {
            "observation_set_id": "q-b",
            "observations": [{
                "string": "Perturb A: B=1",
                "perturbed_node": "A",
                "effects": {"A": 0, "B": 1},
            }],
            "ground_truth_graphs": [{"nodes": ["A", "B"], "edges": [["A", "B"]]}],
        }

        class FakeModel:
            def __init__(self):
                self.n = 0

            def query_with_usage(self, prompt):
                self.n += 1
                # Alternate empty / one-edge so both actions can succeed or fail.
                text = "Graph: No edges" if self.n % 2 else "Graph: A->B"
                return {"response": text, "usage": {"total_tokens": 2}}

        controller = ExplorationController(
            max_edges=1, strategy="adaptive", seed=11,
        )
        with patch("modules.generation_executor.sleep"):
            result = benchmark.evaluate_single_observation_set(
                FakeModel(), question, n_queries=4, max_retries=1,
                verbose=False, controller=controller,
            )
        self.assertEqual(result["decoding"], "controller")
        self.assertEqual(result["n_queries"], 4)
        selects = [e for e in controller.decision_log if e["event"] == "select_action"]
        updates = [e for e in controller.decision_log if e["event"] == "update"]
        self.assertEqual(len(selects), 4)
        self.assertEqual(len(updates), 4)
        for row in result["request_log"]:
            self.assertNotIn("ground_truth_graphs", row["feedback"])


if __name__ == "__main__":
    unittest.main()
