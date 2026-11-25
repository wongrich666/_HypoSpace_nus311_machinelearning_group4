"""Member A request journaling, replay and unknown-response reconciliation, ported from the frozen FAAE runner."""
import os
import hashlib
import json
from contextlib import contextmanager

BUDGET=16
MAX_ATTEMPTS=3
UNKNOWN="interrupted_unknown_response"
TOKENS=("prompt_tokens","completion_tokens","total_tokens")


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class JournalClient:
    """Durable intent before each paid attempt; ambiguous interruption never reruns."""
    def __init__(self, client, path, completed):
        self.client, self.path = client, path
        self.slot, self.attempt = completed, 0

    def query_with_usage(self, prompt):
        self.attempt += 1
        if self.slot >= BUDGET or self.attempt > MAX_ATTEMPTS:
            raise RuntimeError("Request budget exhausted")
        entry = {"query_index": self.slot, "attempt": self.attempt,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        append_json(self.path, {**entry, "event": "started"})
        try:
            response = self.client.query_with_usage(prompt)
        except Exception as exc:
            append_json(self.path, {**entry, "event": "returned", "error_type": type(exc).__name__, "error_message": str(exc)})
            raise
        append_json(self.path, {**entry, "event": "returned", "payload": response})
        return response


def require(value, message):
    if not value:
        raise ValueError(message)


def immutable(path, data):
    if path.exists():
        require(read_json(path) == data, f"Frozen record mismatch: {path.name}")
    else:
        atomic_json(path, data)


@contextmanager
def exclusive(path):
    import msvcrt
    with path.open("a+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def inspect(rows, events, qid):
    """Validate observed requests; only the last attempt of the next slot may be open."""
    require(len(rows) <= 16, "Too many slots")
    attempts = {}
    for e in events:
        i, j = e["query_index"], e["attempt"]
        require(0 <= i < 16 and 1 <= j <= 3, "Budget exceeded")
        group = attempts.setdefault(i, [])
        if e["event"] == "started":
            require(j == len(group)+1 and (not group or group[-1][1] is not None), "Repeated or missing start")
            group.append([e,None])
        else:
            require(e["event"] == "returned" and j == len(group) and group[-1][1] is None, "Unexpected return")
            require(e["prompt_sha256"] == group[-1][0]["prompt_sha256"], "Return prompt mismatch")
            group[-1][1] = e
    require(sorted(attempts) == list(range(len(attempts))) and len(rows) <= len(attempts) <= len(rows)+1, "Invalid slot prefix")
    for i, row in enumerate(rows):
        require(row["query_index"] == i and row["observation_set_id"] == qid, "Wrong slot identity")
        group = attempts[i]
        require([a["attempt"] for a in row["attempts"]] == list(range(1,len(group)+1)), "Wrong attempt count")
        require(all(e["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest() for e,_ in group), "Prompt changed")
        missing = [n for n,(_,ret) in enumerate(group,1) if ret is None]
        if row["status"] == UNKNOWN:
            require(missing == [len(group)] and row["candidate"] is None and row["feedback"]["reward"] == 0 and
                    all(row["usage"][k] is None for k in TOKENS) and row["attempts"][-1]["cost"] is None, "Invalid unknown disposition")
        else:
            require(not missing, "Missing return on completed slot")
    return attempts.get(len(rows), [])


class RecoveryFault(BaseException):
    """An integrity error must not enter the frozen client's API retry path."""


class Transport:
    def __init__(self, client, path, slot, pending):
        self.client = JournalClient(client, path, slot)
        self.pending = list(pending)
        self.client.attempt = len(pending)
        self.unknown = None

    def query_with_usage(self, prompt):
        if not self.pending:
            return self.client.query_with_usage(prompt)
        start, ret = self.pending.pop(0)
        if hashlib.sha256(prompt.encode()).hexdigest() != start["prompt_sha256"]:
            raise RecoveryFault("Reconstructed prompt differs from original request")
        if ret is None:
            self.unknown = start
            # Delivers a terminal zero-reward event through the existing evaluator.
            # No HTTP request or fake 'returned' journal entry is emitted.
            return {"response":"Error querying: interrupted_unknown_response", "retryable":False}
        if "payload" in ret:
            return ret["payload"]
        if "error_message" not in ret:
            raise RecoveryFault("Saved exception lacks replay information")
        error_class = type(ret["error_type"], (Exception,), {})
        raise error_class(ret["error_message"])


def reconcile_row(row, start, hashes, path, method):
    attempt = row["attempts"][-1]
    null_usage = dict.fromkeys(TOKENS)
    observation = dict(**hashes, method=method, question=row["observation_set_id"], slot=row["query_index"]+1,
                       original_start=start, reason="Recorded request start without durable response", status=UNKNOWN,
                       candidate=None, reward=0, billed_tokens=None, billed_cost=None)
    immutable(path, observation)
    attempt.update(status=UNKNOWN, usage=null_usage, error=UNKNOWN, cost=None, content=None,
                   reasoning_content=None, duration_seconds=None, finish_reason=None, request_id=None)
    row.update(status=UNKNOWN, candidate=None, response=None, reasoning_content=None, usage=null_usage,
               cost=None, disposition_audit=path.name)
    row["feedback"].update(status=UNKNOWN, failure_reason=UNKNOWN, failure_detail=None, usage=null_usage,
                           is_valid=False, is_new_valid=False, is_duplicate_valid=False, reward=0)
    for e in row["decision_log"]:
        if e.get("event") == "update":
            e["failure_reason"] = UNKNOWN
