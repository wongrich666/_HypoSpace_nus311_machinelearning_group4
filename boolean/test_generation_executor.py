"""Offline C-layer checks; no model account or benchmark answer set is used."""

import unittest
from unittest.mock import Mock, patch

import requests

from boolean_benchmark import BooleanBenchmarkRefined, boolean_operator_complexity
from boolean_dataset import BooleanObservation
from modules.generation_executor import generate_boolean_one, operator_count
from modules.llm_interface import OpenRouterLLM


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def query_with_usage(self, prompt):
        self.prompts.append(prompt)
        return next(self.replies)


def payload(text, *, finish_reason="stop", tokens=8, retryable=None):
    item = {
        "response": text,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": tokens - 3,
            "total_tokens": tokens,
        },
        "cost": 0.0,
    }
    if retryable is not None:
        item["retryable"] = retryable
    return item


def benchmark_without_answers():
    benchmark = BooleanBenchmarkRefined.__new__(BooleanBenchmarkRefined)
    benchmark.variables = ["x", "y"]
    benchmark.operators = {"AND", "OR", "NOT"}
    benchmark.max_depth = 2
    benchmark.complete_dataset = {"metadata": {"mechanistic_opts": {
        "apply_commutativity": True,
        "apply_idempotence_and_or": True,
        "flatten_associativity": True,
    }}}
    return benchmark


class GenerationExecutorTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = benchmark_without_answers()
        self.observations = [
            BooleanObservation({"x": 1, "y": 1}, 1),
            BooleanObservation({"x": 1, "y": 0}, 0),
        ]

    def test_real_boolean_prompt_parser_and_constraint(self):
        llm = FakeLLM([payload("Expression: x AND y")])
        result = generate_boolean_one(
            self.benchmark, llm, self.observations, ["x OR y"], 1,
        )
        self.assertIn("EXACTLY 1 operator occurrences", result.prompt)
        self.assertIn("Expression: x OR y", result.prompt)
        self.assertEqual(result.candidate, "x AND y")
        self.assertTrue(result.obeyed_operator_count)
        self.assertEqual(result.actual_operator_count, 1)
        self.assertEqual(result.usage["total_tokens"], 8)
        self.assertTrue(self.benchmark.validate_expression(
            result.candidate, self.observations,
        )[0])

    def test_retry_counts_attempts_but_returns_one_candidate(self):
        llm = FakeLLM([
            payload("Error querying OpenRouter: HTTP 429", retryable=True),
            payload("Expression: x AND y", tokens=10),
        ])
        records = []
        result = generate_boolean_one(
            self.benchmark, llm, self.observations, [], 1,
            max_attempts=2, retry_delay_seconds=0, on_attempt=records.append,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(result.usage["total_tokens"], 18)
        self.assertEqual(llm.prompts[0], llm.prompts[1])

    def test_truncation_is_not_parsed_or_retried(self):
        llm = FakeLLM([payload("Expression: x AND y", finish_reason="length")])
        result = generate_boolean_one(
            self.benchmark, llm, self.observations, [], 1, max_attempts=2,
        )
        self.assertEqual(result.status, "truncated")
        self.assertIsNone(result.candidate)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.usage["total_tokens"], 8)

    def test_malformed_provider_response_is_api_error(self):
        llm = FakeLLM([payload("Error parsing OpenRouter response: KeyError")])
        result = generate_boolean_one(
            self.benchmark, llm, self.observations, [], 1,
        )
        self.assertEqual(result.status, "api_error")
        self.assertIsNone(result.candidate)

    def test_stop_before_call(self):
        llm = FakeLLM([])
        result = generate_boolean_one(
            self.benchmark, llm, self.observations, [], 1,
            should_stop=lambda: True,
        )
        self.assertEqual(result.status, "stopped")
        self.assertEqual(llm.prompts, [])

    def test_operator_counter_matches_existing_benchmark(self):
        for expression in ("x", "NOT x", "x AND y", "NOR(x,y)",
                           "NOT (x OR y)", "(x XOR y) OR NOT x"):
            self.assertEqual(operator_count(expression),
                             boolean_operator_complexity(expression))

    @patch("modules.llm_interface.requests.post")
    def test_boolean_client_sends_timeout_and_records_finish_reason(self, post):
        reply = Mock()
        reply.headers = {"x-request-id": "request-test"}
        reply.json.return_value = {
            "choices": [{"message": {"content": "Expression: x AND y"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5,
                      "total_tokens": 8},
        }
        post.return_value = reply
        llm = OpenRouterLLM(model="test-model", api_key="test-key",
                           request_timeout=12)
        result = llm.query_with_usage("Boolean task")
        self.assertEqual(post.call_args.kwargs["timeout"], 12)
        self.assertIn("Boolean logic", post.call_args.kwargs["json"]["messages"][0]["content"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["request_id"], "request-test")

    @patch("modules.llm_interface.requests.post")
    def test_timeout_is_retryable(self, post):
        post.side_effect = requests.exceptions.Timeout("private request details")
        llm = OpenRouterLLM(model="test-model", api_key="test-key",
                           request_timeout=12)
        result = llm.query_with_usage("Boolean task")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["response"], "Error querying OpenRouter: Timeout")


if __name__ == "__main__":
    unittest.main()
