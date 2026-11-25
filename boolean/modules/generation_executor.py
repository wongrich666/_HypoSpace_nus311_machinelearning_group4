"""Execute one Boolean candidate requested by an exploration controller.

This module handles prompting and API accounting only. The Boolean benchmark
remains responsible for validation, mechanistic identity, and scoring.
"""

import re
from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional


_OPERATORS = ("NOR", "XOR", "AND", "OR", "NOT")


def operator_count(formula: str) -> int:
    """Match boolean_benchmark.boolean_operator_complexity for prompt checks."""
    text = formula.upper()
    count = 0
    for op in _OPERATORS:
        found = re.findall(rf"\b{op}\b", text)
        count += len(found)
        if found:
            text = re.sub(rf"\b{op}\b", " ", text)
    return count


@dataclass
class BooleanGenerationResult:
    status: str
    target_operator_count: Optional[int]
    prompt: str = ""
    response: Optional[str] = None
    reasoning_content: str = ""
    candidate: Optional[str] = None
    actual_operator_count: Optional[int] = None
    obeyed_operator_count: Optional[bool] = None
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    })
    cost: float = 0.0
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def _transient_error(message: str) -> bool:
    message = message.lower()
    return any(term in message for term in (
        "timeout", "timed out", "connection", "ssl", "rate limit", "rate_limit",
        "429", "500", "502", "503", "504", "server error",
    ))


def generate_boolean_one(
    benchmark: Any,
    llm: Any,
    observations: List[Any],
    prior_hypotheses: List[str],
    target_operator_count: Optional[int] = None,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    max_attempts: int = 1,
    retry_delay_seconds: float = 0.5,
    on_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> BooleanGenerationResult:
    """Request one candidate; retries are API attempts within that slot.

    ``llm`` must expose ``query_with_usage(prompt)``. The caller sets the
    request timeout and output-token limit on the LLM client. A failed,
    truncated, or malformed answer still consumes the candidate slot.
    """
    if target_operator_count is not None and (
        isinstance(target_operator_count, bool)
        or not isinstance(target_operator_count, int)
        or target_operator_count < 0
    ):
        raise ValueError("target_operator_count must be a nonnegative integer or None")
    if max_attempts < 1 or retry_delay_seconds < 0:
        raise ValueError("invalid retry settings")

    result = BooleanGenerationResult("stopped", target_operator_count)
    if should_stop():
        return result
    result.prompt = benchmark.create_prompt(
        observations, prior_hypotheses, target_complexity=target_operator_count,
    )

    for number in range(1, max_attempts + 1):
        if should_stop():
            return result
        started = monotonic()
        payload: Dict[str, Any] = {}
        error: Optional[str] = None
        try:
            payload = llm.query_with_usage(result.prompt)
            if not isinstance(payload, dict):
                raise TypeError("query_with_usage must return a dictionary")
            response = payload.get("response")
            if isinstance(response, str) and response.startswith((
                "Error querying", "Error parsing OpenRouter response",
            )):
                error = response
            elif response is not None and not isinstance(response, str):
                error = "invalid API response text"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            payload = {}

        usage = payload.get("usage") or {}
        for key in result.usage:
            result.usage[key] += int(usage.get(key) or 0)
        cost = float(payload.get("cost") or 0)
        result.cost += cost
        attempt = {
            "attempt": number,
            "duration_seconds": round(monotonic() - started, 3),
            "status": "error" if error else "ok",
            "error": error,
            "request_id": payload.get("request_id"),
            "finish_reason": payload.get("finish_reason"),
            "content": payload.get("response") if isinstance(payload.get("response"), str) else "",
            "reasoning_content": payload.get("reasoning_content") or "",
            "request_parameters": payload.get("request_parameters"),
            "usage": {key: int(usage.get(key) or 0) for key in result.usage},
            "cost": cost,
        }
        result.attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(dict(attempt))

        if error:
            result.status = "api_error"
            retryable = payload.get("retryable", _transient_error(error))
            if number < max_attempts and retryable:
                if not should_stop():
                    sleep(retry_delay_seconds * (2 ** (number - 1)))
                continue
            return result

        result.response = payload.get("response") if isinstance(payload.get("response"), str) else ""
        result.reasoning_content = payload.get("reasoning_content") or ""
        if payload.get("finish_reason") == "length":
            result.status = "truncated"
            return result
        try:
            result.candidate = benchmark.parse_llm_response(result.response)
        except Exception as exc:
            result.status = "parse_error"
            attempt["parse_error"] = f"{type(exc).__name__}: {exc}"
            return result
        if not result.candidate:
            result.status = "parse_error"
            return result
        result.status = "ok"
        result.actual_operator_count = operator_count(result.candidate)
        if target_operator_count is not None:
            result.obeyed_operator_count = (
                result.actual_operator_count == target_operator_count
            )
        return result

    return result
