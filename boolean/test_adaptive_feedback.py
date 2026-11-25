import json
import tempfile
import unittest
from pathlib import Path

from boolean_benchmark import BooleanBenchmarkRefined
from modules.experiment_manifest import freeze_questions
from modules.exploration_controller import BooleanAdaptiveController


def _question(question_id):
    return {
        "observation_set_id": question_id,
        "n_observations": 1,
        "n_compatible_expressions": 1,
        "observations": [{"inputs": {"x": 1, "y": 1}, "output": 1, "string": "(x=1, y=1) -> 1"}],
        "ground_truth_expressions": [{
            "formula": "x AND y",
            "mechanistic_key": "('AND', (('VAR', 'x'), ('VAR', 'y')))",
            "truth_table": {"(1, 1)": 1},
        }],
    }


def _dataset(directory):
    path = Path(directory) / "questions.json"
    path.write_text(json.dumps({
        "metadata": {
            "variables": ["x", "y"],
            "operators": ["NOT", "AND", "OR"],
            "max_depth": 1,
            "mechanistic_opts": {
                "apply_commutativity": True,
                "apply_idempotence_and_or": True,
                "flatten_associativity": True,
            },
        },
        "datasets_by_n_observations": {"1": [_question("q1"), _question("q2")]},
    }), encoding="utf-8")
    return path


class FakeModel:
    def __init__(self):
        self.prompts = []

    def get_name(self):
        return "fake"

    def query_with_usage(self, prompt):
        self.prompts.append(prompt)
        return {
            "response": "x AND y",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "finish_reason": "stop",
        }


class AdaptiveFeedbackTest(unittest.TestCase):
    def test_two_questions_reset_and_hide_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark = BooleanBenchmarkRefined(str(_dataset(directory)))
            self.assertEqual(benchmark.max_operator_count, 1)
            questions = benchmark.all_observation_sets
            model = FakeModel()
            result = benchmark.run_benchmark(
                model, n_samples=2, n_queries_per_sample=2, seed=7, verbose=False,
                checkpoint_dir=str(Path(directory) / "checkpoints"), decoding="adaptive",
                selected_questions=questions, max_retries=1,
            )
        self.assertEqual(result["uses_ground_truth_distribution"], False)
        self.assertEqual(result["method_label"], "Online operator-count adaptive decoding")
        self.assertEqual(len(result["per_sample_results"]), 2)
        for sample in result["per_sample_results"]:
            self.assertEqual(sample["n_queries"], 2)
            self.assertEqual(len(sample["request_log"]), 2)
            self.assertEqual(sample["decision_log"][0]["counts"], {"0": 0, "1": 0})
            self.assertEqual(sample["method_parameters"]["forget_factor"], 0.9)
            for row in sample["request_log"]:
                blob = json.dumps(row["feedback"])
                self.assertNotIn("ground_truth", blob)
                self.assertNotIn("recovery", blob)
                self.assertNotIn("n_compatible", blob)
        self.assertGreaterEqual(model.prompts[1].count("Expression:"), 1)

    def test_controller_rejects_oracle_fields(self):
        controller = BooleanAdaptiveController(1, seed=1)
        with self.assertRaises(ValueError):
            controller.select_action({"ground_truth_expressions": []})
        with self.assertRaises(ValueError):
            controller.update(0, {"recovery_rate": 1, "is_new_valid": True})

    def test_manifest_rejects_a_changed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark = BooleanBenchmarkRefined(str(_dataset(directory)))
            path = Path(directory) / "manifest.json"
            settings = {"n_samples": 2, "n_queries": 16, "seed": 42}
            _questions, manifest = freeze_questions(benchmark, path, settings)
            self.assertEqual(len(manifest["question_ids"]), 2)
            with self.assertRaises(ValueError):
                freeze_questions(benchmark, path, {**settings, "n_queries": 8})


if __name__ == "__main__":
    unittest.main()
