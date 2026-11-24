"""Check candidate-slot and provider-attempt budgets through the runner."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from boolean_benchmark import BooleanBenchmarkRefined


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def get_name(self):
        return "FailingFake"

    def query_with_usage(self, prompt):
        self.calls += 1
        return {
            "response": "Error querying OpenRouter: HTTP 503",
            "retryable": True,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
            "cost": 0.0,
        }


class AttemptAccountingTests(unittest.TestCase):
    def test_max_attempts_reaches_generator_and_error_rate_uses_attempts(self):
        dataset = Path(__file__).parent / "datasets/boolean_extended_d2.json"
        benchmark = BooleanBenchmarkRefined(str(dataset))
        llm = FailingLLM()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result = benchmark.run_benchmark(
                llm, n_samples=1, n_queries_per_sample=1,
                seed=1, verbose=False, checkpoint_dir=directory,
                run_id="attempt-accounting", max_attempts_per_slot=3,
            )
        self.assertEqual(llm.calls, 3)
        self.assertEqual(result["max_attempts_per_slot"], 3)
        self.assertEqual(result["per_sample_results"][0]["n_queries"], 1)
        self.assertEqual(result["per_sample_results"][0]["api_attempt_count"], 3)
        self.assertEqual(result["error_summary"]["api_attempt_count"], 3)
        self.assertEqual(result["error_summary"]["error_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
