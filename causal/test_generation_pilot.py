import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from run_generation_pilot import main
from run_causal_benchmark import CausalBenchmarkEnhanced


def provider_response(status_code, text="", finish_reason="stop"):
    response = requests.Response()
    response.status_code = status_code
    response.headers["x-request-id"] = "mock-id"
    response._content = json.dumps({
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode()
    return response


class GenerationPilotTest(unittest.TestCase):
    def test_normal_benchmark_uses_candidate_slots(self):
        benchmark = CausalBenchmarkEnhanced()
        benchmark.nodes = ["A", "B"]
        benchmark.max_edges = 1
        question = {
            "observation_set_id": "q1",
            "ground_truth_graphs": [{"nodes": ["A", "B"], "edges": [["A", "B"]]}],
            "observations": [{"string": "Perturb A: B=1", "perturbed_node": "A",
                              "effects": {"A": 0, "B": 1}}],
        }
        class MockLLM:
            calls = 0

            def query_with_usage(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("timed out")
                return {"response": "Graph: A->B", "usage": {
                    "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
                }}

        with patch("modules.generation_executor.sleep"):
            result = benchmark.evaluate_single_observation_set(
                MockLLM(), question, n_queries=1, max_retries=2, verbose=False
            )
        self.assertEqual(result["n_queries"], 1)
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["parse_success_count"], 1)
        self.assertEqual(len(result["request_log"][0]["attempts"]), 2)
        self.assertEqual(result["token_usage"]["total_tokens"], 15)

    def test_paired_cli_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset, config, output = [directory / filename for filename in
                                       ("questions.json", "config.yaml", "result.json")]
            dataset.write_text(json.dumps({
                "metadata": {"nodes": ["A", "B"], "max_edges": 1},
                "datasets": [{
                    "observation_set_id": "q1",
                    "observations": [{"string": "Perturb A: B=1", "perturbed_node": "A",
                                      "effects": {"A": 0, "B": 1}}],
                }],
            }), encoding="utf-8")
            config.write_text(
                "llm: {type: dashscope, models: {dashscope: mock}, "
                "max_tokens: 64, request_timeout: 2}\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "mock"}), patch(
                "modules.llm_interface.requests.post",
                side_effect=[
                    provider_response(429),
                    provider_response(200, "Graph: A->B"),
                    provider_response(200, "Graph: A->B", finish_reason="length"),
                ],
            ) as post, patch("modules.generation_executor.sleep"):
                main(["--dataset", str(dataset), "--config", str(config),
                      "--output", str(output), "--target-edge-count", "1",
                      "--max-attempts", "2"])
            report = json.loads(output.read_text(encoding="utf-8"))["arms"]

        baseline, constrained = report["unconstrained"], report["constrained"]
        self.assertEqual((baseline["candidate_slots"], constrained["candidate_slots"]), (1, 1))
        self.assertEqual((baseline["api_attempts"], constrained["api_attempts"]), (2, 1))
        self.assertEqual((baseline["valid"], constrained["valid"]), (1, 0))
        self.assertEqual(constrained["results"][0]["status"], "parse_error")
        self.assertEqual(baseline["results"][0]["attempts"][1]["request_id"], "mock-id")
        self.assertEqual(baseline["usage"]["total_tokens"], 15)
        self.assertIsNone(baseline["cost_estimate"]["estimated_cost"])
        requests_sent = [call.kwargs for call in post.call_args_list]
        self.assertTrue(all(req["timeout"] == 2 and req["json"]["max_tokens"] == 64
                            for req in requests_sent))
        self.assertNotIn("EXACTLY 1", requests_sent[0]["json"]["messages"][-1]["content"])
        self.assertIn("EXACTLY 1", requests_sent[-1]["json"]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
