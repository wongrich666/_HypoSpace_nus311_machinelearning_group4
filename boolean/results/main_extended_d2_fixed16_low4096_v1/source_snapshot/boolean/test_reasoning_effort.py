import unittest
from unittest.mock import patch

from boolean_benchmark import BooleanBenchmarkRefined
from modules.generation_executor import generate_boolean_one
from modules.llm_interface import OpenRouterLLM


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {"x-request-id": "req-test"}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _message(content, reasoning, finish_reason="stop"):
    return {
        "id": "chat-test",
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content, "reasoning_content": reasoning},
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


class ReasoningEffortTest(unittest.TestCase):
    def _client(self):
        client = OpenRouterLLM(
            model="deepseek-v4.1-flash-1017",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2048,
            request_timeout=60,
            reasoning_effort="low",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        return client

    def test_low_effort_enters_the_http_request(self):
        client = self._client()
        seen = {}

        def post(url, headers, json, timeout):
            seen["url"] = url
            seen["json"] = json
            seen["timeout"] = timeout
            self.assertNotIn("test-key", str(json))
            return _Response(_message("Expression: x AND y", "Expression: NOT x"))

        client.session.post = post
        payload = client.query_with_usage("observations")
        self.assertEqual(seen["timeout"], 60)
        self.assertEqual(seen["json"]["reasoning_effort"], "low")
        self.assertEqual(seen["json"]["max_tokens"], 2048)
        self.assertEqual(seen["json"]["temperature"], 0.7)
        self.assertEqual(seen["json"]["model"], "deepseek-v4.1-flash-1017")
        self.assertEqual(payload["response"], "Expression: x AND y")
        self.assertEqual(payload["reasoning_content"], "Expression: NOT x")
        self.assertEqual(payload["request_parameters"]["reasoning_effort"], "low")

    def test_reasoning_text_is_not_parsed_as_the_answer(self):
        client = self._client()
        client.session.post = lambda url, headers, json, timeout: _Response(
            _message("", "Expression: NOT x")
        )
        benchmark = BooleanBenchmarkRefined.__new__(BooleanBenchmarkRefined)
        with patch.object(benchmark, "create_prompt", return_value="prompt"):
            result = generate_boolean_one(benchmark, client, [], [])
        self.assertEqual(result.reasoning_content, "Expression: NOT x")
        self.assertEqual(result.response, "")
        self.assertIsNone(result.candidate)
        self.assertEqual(result.status, "parse_error")
        self.assertEqual(result.attempts[0]["content"], "")
        self.assertEqual(result.attempts[0]["reasoning_content"], "Expression: NOT x")

    def test_final_content_is_parsed_when_reasoning_disagrees(self):
        client = self._client()
        client.session.post = lambda url, headers, json, timeout: _Response(
            _message("Expression: x AND y", "Expression: NOT x")
        )
        benchmark = BooleanBenchmarkRefined.__new__(BooleanBenchmarkRefined)
        with patch.object(benchmark, "create_prompt", return_value="prompt"):
            result = generate_boolean_one(benchmark, client, [], [])
        self.assertEqual(result.candidate, "x AND y")
        self.assertEqual(result.reasoning_content, "Expression: NOT x")


if __name__ == "__main__":
    unittest.main()
