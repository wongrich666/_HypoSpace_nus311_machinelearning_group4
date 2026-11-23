"""Online feedback: observations and previously valid hashes only; no answer set."""
import json


def evaluate_candidate(generation, observations, seen_valid_hashes, validator):
    """Consume one parsed GenerationResult and update per-question valid hashes.

    Use the official parser upstream and the official observation validator here.
    A duplicate remains valid but earns zero reward. Target compliance is separate.
    """
    candidate = generation.candidate
    reason = None
    detail = None
    valid = False
    graph_hash = None
    new_valid = False
    if generation.status != "ok" or candidate is None:
        reason = generation.status if generation.status != "ok" else "parse_error"
        if generation.attempts:
            last = generation.attempts[-1]
            detail = last.get("parse_error") or last.get("error")
            if last.get("finish_reason") == "length":
                reason = "truncated_output"
    else:
        graph_hash = candidate.get_hash()
        valid = bool(validator(candidate, observations))
        if not valid:
            reason = "observation_mismatch"
        else:
            new_valid = graph_hash not in seen_valid_hashes
            seen_valid_hashes.add(graph_hash)
    return {
        "status": generation.status,
        "is_valid": valid,
        "failure_reason": reason,
        "failure_detail": detail,
        "canonical_hash": graph_hash,
        "is_new_valid": new_valid,
        "is_duplicate_valid": valid and not new_valid,
        "reward": int(new_valid),
        "target_edge_count": generation.target_edge_count,
        "actual_edge_count": generation.actual_edge_count,
        "obeyed_edge_count": generation.obeyed_edge_count,
        "usage": dict(generation.usage),
        "api_attempt_count": len(generation.attempts),
    }


class CandidateJSONLLogger:
    """Append and flush one completed candidate, including post-parse exceptions.

    Single-writer sink. Open a separate file per run; it is not a resume mechanism.
    I/O failures propagate so a run cannot silently claim its log was persisted.
    """
    def __init__(self, path):
        self.path = path

    def __call__(self, record):
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
