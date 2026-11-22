"""One-candidate causal generation with observable request accounting.

The controller supplies a target edge count; validation and graph identity stay
with the benchmark's existing parser, validator, and CausalGraph implementation.
"""

from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GenerationResult:
    status: str
    target_edge_count: Optional[int]
    prompt: str = ""
    response: Optional[str] = None
    candidate: Any = None
    actual_edge_count: Optional[int] = None
    obeyed_edge_count: Optional[bool] = None
    cost: float = 0.0
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    })
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def _transient_error(message: str) -> bool:
    message = message.lower()
    return any(term in message for term in (
        "timeout", "timed out", "connection", "rate limit", "rate_limit",
        "429", "500", "502", "503", "504", "server error",
    ))


def generate_one(
    llm: Any,
    prompt_builder: Callable[[Optional[int]], str],
    parser: Callable[[str], Any],
    target_edge_count: Optional[int] = None,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.5,
    on_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> GenerationResult:
    """Execute one requested candidate, counting every actual API attempt.

    ``llm`` must implement ``query_with_usage(prompt)``. The caller configures
    its output token limit and request timeout. ``parser`` should be the
    official benchmark parser; this function never decides validity or novelty.
    """
    if target_edge_count is not None and target_edge_count < 0:
        raise ValueError("target_edge_count must be nonnegative")
    if max_attempts < 1 or retry_delay_seconds < 0:
        raise ValueError("invalid retry settings")
    result = GenerationResult(status="stopped", target_edge_count=target_edge_count)
    if should_stop():
        return result
    result.prompt = prompt_builder(target_edge_count)
    for attempt_number in range(1, max_attempts + 1):
        if should_stop():
            result.status = "stopped"
            return result
        started = monotonic()
        response = None
        error = None
        usage = {}
        payload = {}
        try:
            payload = llm.query_with_usage(result.prompt)
            response = payload.get("response")
            usage = payload.get("usage") or {}
            if isinstance(response, str) and response.startswith("Error querying"):
                error = response
            elif response is None or isinstance(response, str):
                response = response or ""
            else:
                error = "invalid API response text"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        for key in result.usage:
            result.usage[key] += int(usage.get(key) or 0)
        result.cost += float(payload.get("cost") or 0)
        record = {
            "attempt": attempt_number,
            "duration_seconds": round(monotonic() - started, 3),
            "status": "error" if error else "ok",
            "error": error,
            "request_id": payload.get("request_id"),
            "finish_reason": payload.get("finish_reason"),
            "usage": {key: int(usage.get(key) or 0) for key in result.usage},
            "cost": float(payload.get("cost") or 0),
        }
        result.attempts.append(record)
        if on_attempt is not None:
            on_attempt(record)
        if error:
            result.status = "api_error"
            retryable = payload.get("retryable", _transient_error(error))
            if attempt_number < max_attempts and retryable:
                if not should_stop():
                    sleep(retry_delay_seconds * (2 ** (attempt_number - 1)))
                continue
            return result
        result.response = response
        if payload.get("finish_reason") != "length":
            result.candidate = parser(response)
        result.status = "ok" if result.candidate is not None else "parse_error"
        if result.candidate is not None:
            result.actual_edge_count = len(result.candidate.edges)
            if target_edge_count is not None:
                result.obeyed_edge_count = result.actual_edge_count == target_edge_count
        return result
    return result


def generate_causal_one(
    benchmark: Any,
    llm: Any,
    observations: List[Dict[str, Any]],
    prior_hypotheses: List[Any],
    target_edge_count: Optional[int] = None,
    **kwargs: Any,
) -> GenerationResult:
    """Use the existing causal prompt and parser without changing scoring."""
    return generate_one(
        llm,
        lambda count: benchmark.create_prompt(
            observations, prior_hypotheses, target_complexity=count
        ),
        benchmark.parse_llm_response,
        target_edge_count,
        **kwargs,
    )
