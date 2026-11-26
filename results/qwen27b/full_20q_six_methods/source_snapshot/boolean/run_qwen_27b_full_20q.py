"""Qwen 3.5-27B formal Boolean six-method evaluation (20q x 16 slots).

Uses the same held-out question_ids as DeepSeek boolean-main-six-methods-thinking-off-v1.
Model: qwen3.5-27b, enable_thinking=false, temp=0.7, max_tokens=4096, seed=42.
Supports --resume / per-question skip and candidate-log checkpointing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
BOOLEAN = REPO / "boolean"
sys.path.insert(0, str(BOOLEAN))
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(key, None)

from boolean_benchmark import BooleanBenchmarkRefined, setup_llm
from member_a_journal import (
    Transport, append_json, atomic_json, digest, exclusive, immutable,
    inspect, jsonl, read_json, reconcile_row, require, RecoveryFault,
)
from member_a_memory_sde_benchmark import MemorySDEBenchmark
from member_a_sde_benchmark import SDEBenchmark
from modules.exploration_controller import BooleanAdaptiveController
from modules.experiment_manifest import freeze_questions
from modules.llm_interface import direct_http_session

PROTOCOL_PATH = BOOLEAN / "experiments" / "qwen27b_full_20q_six_methods_v1.json"
DATASET = BOOLEAN / "datasets" / "boolean_extended_d2.json"
MODEL = "qwen3.5-27b"
PLAIN_METHODS = ("memory", "uniform", "round_robin", "adaptive")
SDE_METHODS = ("adaptive_sde", "memory_sde")
METHODS = (
    "memory",
    "adaptive",
    "adaptive_sde",
    "memory_sde",
    "uniform",
    "round_robin",
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or ""
    if not key:
        # Fall back to Windows User environment (common on self-hosted workers).
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as hive:
                key, _ = winreg.QueryValueEx(hive, "DASHSCOPE_API_KEY")
        except Exception:
            key = ""
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY in the environment before running")
    os.environ["DASHSCOPE_API_KEY"] = key
    return key


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["n_samples"] != 20 or protocol["n_queries"] != 16 or protocol["seed"] != 42:
        raise SystemExit("protocol must be frozen 20x16 seed=42")
    if len(protocol["question_ids"]) != 20:
        raise SystemExit("protocol question_ids must contain exactly 20 ids")
    if protocol["max_attempts_per_slot"] != 3:
        raise SystemExit("Frozen protocol requires at most three attempts per slot")
    if protocol.get("enable_thinking") is not False:
        raise SystemExit("enable_thinking must be false")
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise SystemExit("Dataset fingerprint differs from the frozen protocol")
    if protocol["model"] != MODEL:
        raise SystemExit(f"protocol model must be {MODEL}")
    return protocol


def probe_model(key: str, model: str) -> None:
    session = direct_http_session()
    response = session.get(
        BASE_URL + "/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    print("PROBE_STATUS", response.status_code, flush=True)
    if response.status_code != 200:
        raise SystemExit("model list probe failed")
    names = [item.get("id") for item in response.json().get("data", [])]
    print("PROBE_HAS_MODEL", model in names, "MODEL_COUNT", len(names), flush=True)
    if model not in names:
        raise SystemExit("target model missing from provider list")


def make_client(key: str, protocol: dict):
    llm = setup_llm(
        "dashscope",
        model=MODEL,
        api_key=key,
        temperature=protocol["temperature"],
        max_tokens=protocol["max_tokens"],
        base_url=BASE_URL,
        request_timeout=protocol["request_timeout"],
        enable_thinking=False,
    )
    body = llm.request_body("parameter-check")
    expected = {
        "model": MODEL,
        "temperature": 0.7,
        "max_tokens": 4096,
        "enable_thinking": False,
    }
    actual = {k: body.get(k) for k in expected}
    if actual != expected or "reasoning_effort" in body:
        raise SystemExit(f"client request body mismatch: {actual}")
    if llm.request_timeout != 60 or llm.session.trust_env:
        raise SystemExit("client timeout/proxy settings invalid")
    print("REQUEST_BODY", actual, "TIMEOUT", llm.request_timeout, flush=True)
    return llm


def public_dataset(data: dict, ids: list[str]) -> dict:
    metadata = {k: data["metadata"][k] for k in ("variables", "operators", "max_depth", "mechanistic_opts")}
    lookup = {q["observation_set_id"]: q for g in data["datasets_by_n_observations"].values() for q in g}
    questions = []
    for qid in ids:
        q = {k: lookup[qid][k] for k in ("observation_set_id", "observations", "n_observations")}
        q["ground_truth_expressions"] = []
        questions.append(q)
    return {"metadata": metadata, "datasets_by_n_observations": {"public_only": questions}}


def load_log(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_plain_method(benchmark, llm, out: Path, index: int, method: str, question, protocol, manifest):
    result_path = out / method / f"q{index}.json"
    log_path = out / method / f"q{index}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        print("SKIP", result_path.relative_to(out), flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    prior = load_log(log_path)
    qid = question["observation_set_id"]
    if any(row.get("observation_set_id") != qid for row in prior):
        raise SystemExit("Candidate log belongs to another question")
    if prior and prior[-1]["query_index"] + 1 != len(prior):
        raise SystemExit("candidate log is not a contiguous prefix")
    print("START", method, index, qid, "resume_from", len(prior), flush=True)

    def sink(row, log_path=log_path, index=index, method=method):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        usage = row.get("usage") or {}
        print(
            "PROGRESS", method, index, row["query_index"] + 1, row["status"],
            "reward", row["feedback"]["reward"],
            "completion", usage.get("completion_tokens", 0),
            flush=True,
        )
        if row["status"] == "api_error":
            raise RuntimeError("API error after retries; stop before spending more")

    result = benchmark.run_benchmark(
        llm,
        n_queries_per_sample=protocol["n_queries"],
        seed=protocol["seed"],
        verbose=False,
        checkpoint_dir=str(out / "checkpoints" / method),
        max_retries=protocol["max_attempts_per_slot"],
        run_id=f"qwen_full_{method}_q{index}_{uuid.uuid4().hex}",
        decoding=method,
        selected_questions=[question],
        on_candidate_complete=sink,
        prior_completions=prior,
        adaptive_parameters=protocol["adaptive_parameters"],
    )
    samples = result.get("per_sample_results", [])
    if len(samples) != 1 or len(samples[0].get("request_log", [])) != protocol["n_queries"]:
        raise RuntimeError("Incomplete run: refusing to save as completed")
    result["experiment_manifest"] = manifest
    result["provider_base_url"] = BASE_URL
    result["model"] = MODEL
    result["enable_thinking"] = False
    result["resumed_logged_candidates"] = len(prior)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE", method, index, flush=True)
    return result


def run_sde_group(bench, group, question, method, client, protocol, mode, manifest_hash):
    qid = question["observation_set_id"]
    if method == "adaptive_sde":
        seed = int(hashlib.sha256(json.dumps([protocol["seed"], qid]).encode()).hexdigest()[:16], 16)
        params = protocol["adaptive_parameters"]
        ctrl = BooleanAdaptiveController(
            3,
            seed=seed,
            strategy="adaptive",
            forget_factor=params["forget_factor"],
            temperature=params["controller_temperature"],
            explore_bonus=params["explore_bonus"],
        )
    elif method == "memory_sde":
        seed = None
        ctrl = None
    else:
        raise ValueError(f"not an SDE method: {method}")
    group.mkdir(parents=True, exist_ok=True)
    rows = jsonl(group / "candidates.jsonl")
    if len(rows) == protocol["n_queries"] and not inspect(rows, jsonl(group / "attempts.jsonl"), qid):
        print("SKIP", method, qid, "already complete", flush=True)
        return rows
    pending = inspect(rows, jsonl(group / "attempts.jsonl"), qid)
    transport = Transport(client, group / "attempts.jsonl", len(rows), pending)

    def sink(row):
        if transport.pending:
            raise RecoveryFault("Unconsumed saved API attempts")
        row.update(method=method, mode=mode, controller_seed=seed)
        if transport.unknown:
            reconcile_row(
                row,
                transport.unknown,
                {"protocol_sha256": manifest_hash},
                group / f"interruption_slot_{row['query_index'] + 1}.json",
                method,
            )
            transport.unknown = None
        append_json(group / "candidates.jsonl", row)
        transport.client.slot += 1
        transport.client.attempt = 0
        print(
            "PROGRESS", method, qid, row["query_index"] + 1, row["status"],
            "reward", row["feedback"]["reward"],
            flush=True,
        )
        if row["status"] == "api_error":
            raise RuntimeError("API error slot persisted; inspect provider")

    print("START", method, qid, "resume_from", len(rows), flush=True)
    bench.evaluate_group(question, transport, ctrl, True, rows, sink)
    all_rows = jsonl(group / "candidates.jsonl")
    require(
        len(all_rows) == protocol["n_queries"] and not inspect(all_rows, jsonl(group / "attempts.jsonl"), qid),
        f"Incomplete {method} group",
    )
    return all_rows


def score_plain(method: str, results: list[dict], ids: list[str], dataset: dict, n_samples: int, n_queries: int) -> dict:
    refs = {
        q["observation_set_id"]: {
            ast.literal_eval(e["mechanistic_key"]) for e in q["ground_truth_expressions"]
        }
        for block in dataset["datasets_by_n_observations"].values()
        for q in block
        if q["observation_set_id"] in ids
    }
    questions = {}
    counts = Counter()
    curves = []
    for sample in results:
        qid = sample["observation_set_id"]
        ref = refs[qid]
        seen = set()
        curve = []
        for i, row in enumerate(sample["request_log"], 1):
            feedback = row["feedback"]
            counts["slots"] += 1
            if feedback.get("is_valid"):
                key = ast.literal_eval(feedback["mechanistic_key"])
                seen.add(key)
            counts["valid"] += int(bool(feedback.get("is_valid")))
            counts["novel"] += int(bool(feedback.get("is_new_valid")))
            counts["duplicates"] += int(bool(feedback.get("is_duplicate_valid")))
            counts["truncated"] += int(row.get("status") == "truncated")
            for attempt in row.get("attempts") or []:
                counts["attempts"] += 1
            curve.append(len(seen & ref))
        questions[qid] = {
            "found": curve[-1],
            "total": len(ref),
            "curve": curve,
            "recovery_curve": [n / len(ref) for n in curve],
            "recovery": curve[-1] / len(ref),
            "truncations": sum(1 for row in sample["request_log"] if row.get("status") == "truncated"),
        }
        curves.append(questions[qid]["recovery_curve"])
    found = sum(q["found"] for q in questions.values())
    slots_total = n_samples * n_queries
    return {
        "method": method,
        "questions": questions,
        "counts": dict(counts),
        "macro_recovery": mean(c[-1] for c in curves),
        "micro_recovery": found / sum(len(refs[q]) for q in ids),
        "recovery_at": {str(k): mean(c[k - 1] for c in curves) for k in (4, 8, 16)},
        "truncation_rate": counts["truncated"] / slots_total,
        "truncations_per_question": counts["truncated"] / n_samples,
    }


def score_sde(method: str, groups: dict, ids: list[str], dataset: dict, n_samples: int, n_queries: int) -> dict:
    refs = {
        q["observation_set_id"]: {
            ast.literal_eval(e["mechanistic_key"]) for e in q["ground_truth_expressions"]
        }
        for block in dataset["datasets_by_n_observations"].values()
        for q in block
        if q["observation_set_id"] in ids
    }
    questions = {}
    counts = Counter()
    curves = []
    for qid in ids:
        rows = groups[qid]
        seen = set()
        curve = []
        for i, row in enumerate(rows, 1):
            feedback = row["feedback"]
            counts["slots"] += 1
            if feedback["is_valid"]:
                key = ast.literal_eval(feedback["mechanistic_key"])
                seen.add(key)
            counts["valid"] += int(feedback["is_valid"])
            counts["novel"] += int(feedback["is_new_valid"])
            counts["duplicates"] += int(feedback["is_duplicate_valid"])
            counts["truncated"] += int(row["status"] == "truncated")
            for attempt in row.get("attempts") or []:
                counts["attempts"] += 1
            curve.append(len(seen & refs[qid]))
        questions[qid] = {
            "found": curve[-1],
            "total": len(refs[qid]),
            "curve": curve,
            "recovery_curve": [n / len(refs[qid]) for n in curve],
            "recovery": curve[-1] / len(refs[qid]),
            "truncations": sum(1 for row in rows if row["status"] == "truncated"),
        }
        curves.append(questions[qid]["recovery_curve"])
    found = sum(q["found"] for q in questions.values())
    slots_total = n_samples * n_queries
    return {
        "method": method,
        "questions": questions,
        "counts": dict(counts),
        "macro_recovery": mean(c[-1] for c in curves),
        "micro_recovery": found / sum(len(refs[q]) for q in ids),
        "recovery_at": {str(k): mean(c[k - 1] for c in curves) for k in (4, 8, 16)},
        "truncation_rate": counts["truncated"] / slots_total,
        "truncations_per_question": counts["truncated"] / n_samples,
    }


def write_method_bundle(out: Path, method: str, metrics: dict, payload: dict) -> Path:
    path = out / f"qwen-27b-full-20q-16-{method}.json"
    path.write_text(
        json.dumps(
            {"method": method, "model": MODEL, "enable_thinking": False, "metrics": metrics, "payload": payload},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)

    if os.environ.get("PYTHONHASHSEED") != "1":
        raise SystemExit("Start Python with PYTHONHASHSEED=1 to match operator order")

    out = args.out.resolve()
    protocol = load_protocol(args.protocol.resolve())
    ids = protocol["question_ids"]
    dataset = read_json(DATASET)
    n_samples = protocol["n_samples"]
    n_queries = protocol["n_queries"]

    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
    elif not (out / "manifest.json").exists() and any(out.iterdir()):
        raise SystemExit("Refusing unrelated output directory")

    settings = {
        "protocol": protocol["protocol_id"],
        "purpose": protocol["purpose"],
        "dataset": "datasets/boolean_extended_d2.json",
        "dataset_sha256": protocol["dataset_sha256"],
        "question_ids": ids,
        "n_samples": n_samples,
        "n_queries": n_queries,
        "seed": 42,
        "methods": list(args.methods),
        "planned_methods": list(METHODS),
        "candidate_slots_total": n_samples * n_queries * len(METHODS),
        "model": MODEL,
        "temperature": 0.7,
        "max_tokens": 4096,
        "enable_thinking": False,
        "reasoning_effort": None,
        "request_timeout": 60,
        "base_url": BASE_URL,
        "adaptive_parameters": protocol["adaptive_parameters"],
        "max_attempts_per_slot": 3,
        "paired_deepseek_protocol_id": protocol.get("paired_deepseek_protocol_id"),
        "code_revision": git_revision(),
    }

    with exclusive(out / "run.lock"):
        if not (out / "manifest.json").exists():
            immutable(out / "manifest.json", settings)
            shutil.copy2(args.protocol.resolve(), out / "protocol.json")
        else:
            existing = read_json(out / "manifest.json")
            locked = (
                "model", "temperature", "max_tokens", "enable_thinking", "seed",
                "n_queries", "n_samples", "question_ids", "base_url", "adaptive_parameters",
            )
            if any(existing.get(k) != settings.get(k) for k in locked) and not args.resume:
                raise SystemExit("Existing manifest settings differ; use a new --out or --resume")

        snapshot = out / "source_snapshot"
        if not snapshot.exists():
            snapshot.mkdir(parents=True)
            for rel in (
                "boolean/run_qwen_27b_full_20q.py",
                "boolean/experiments/qwen27b_full_20q_six_methods_v1.json",
                "boolean/boolean_benchmark.py",
                "boolean/modules/llm_interface.py",
                "boolean/modules/exploration_controller.py",
                "boolean/modules/generation_executor.py",
                "boolean/member_a_sde_benchmark.py",
                "boolean/member_a_structure.py",
                "boolean/member_a_memory_sde_benchmark.py",
                "boolean/member_a_memory_structure.py",
            ):
                target = snapshot / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / rel, target)

        print("FROZEN", ids, flush=True)
        print(
            "CONFIG",
            {
                "model": MODEL,
                "enable_thinking": False,
                "temperature": 0.7,
                "max_tokens": 4096,
                "seed": 42,
                "n_samples": n_samples,
                "n_queries": n_queries,
                "methods": list(args.methods),
            },
            flush=True,
        )
        if args.dry_run:
            print("DRY_RUN_OK", flush=True)
            return

        key = api_key()
        probe_model(key, MODEL)
        llm = make_client(key, protocol)

        full_benchmark = BooleanBenchmarkRefined(str(DATASET))
        if full_benchmark.max_operator_count != 3:
            raise SystemExit(f"expected public operator bound 3, got {full_benchmark.max_operator_count}")
        if ", ".join(full_benchmark.operators) != "AND, OR, NOT":
            raise SystemExit("Prompt operator order mismatch")

        freeze_settings = {
            "dataset_sha256": settings["dataset_sha256"],
            "dataset": settings["dataset"],
            "n_samples": n_samples,
            "n_queries": n_queries,
            "seed": 42,
            "methods": list(METHODS),
            "model": MODEL,
            "temperature": 0.7,
            "max_tokens": 4096,
            "enable_thinking": False,
            "adaptive_parameters": protocol["adaptive_parameters"],
        }
        freeze_path = out / "question_manifest.json"
        if not freeze_path.exists():
            freeze_path.write_text(
                json.dumps({"schema_version": 1, "settings": freeze_settings, "question_ids": ids}, indent=2) + "\n",
                encoding="utf-8",
            )
        questions, question_manifest = freeze_questions(full_benchmark, freeze_path, freeze_settings)
        if [q["observation_set_id"] for q in questions] != ids:
            raise SystemExit("frozen question IDs differ from protocol")

        summary = {
            "model": MODEL,
            "enable_thinking": False,
            "protocol": protocol["protocol_id"],
            "question_ids": ids,
            "n_samples": n_samples,
            "n_queries": n_queries,
            "methods": {},
        }
        if (out / "summary.json").exists():
            summary = read_json(out / "summary.json")
            summary.setdefault("methods", {})
        status_path = out / "status.json"
        atomic_json(status_path, {"status": "running", "methods": list(args.methods)})

        for method in METHODS:
            if method not in args.methods:
                continue
            if method in PLAIN_METHODS:
                method_results = []
                for index, question in enumerate(questions):
                    result = run_plain_method(
                        full_benchmark, llm, out, index, method, question, protocol, question_manifest
                    )
                    method_results.append(result["per_sample_results"][0])
                metrics = score_plain(method, method_results, ids, dataset, n_samples, n_queries)
                write_method_bundle(out, method, metrics, {"per_sample_results": method_results})
                summary["methods"][method] = metrics
                print(
                    "METHOD_DONE", method,
                    "macro_recovery", round(metrics["macro_recovery"], 4),
                    "truncation_rate", round(metrics["truncation_rate"], 4),
                    flush=True,
                )
            else:
                public_path = out / "public_tasks.json"
                if not public_path.exists():
                    immutable(public_path, public_dataset(dataset, ids))
                public = read_json(public_path)
                public_questions = public["datasets_by_n_observations"]["public_only"]
                bench = (
                    SDEBenchmark(str(public_path))
                    if method == "adaptive_sde"
                    else MemorySDEBenchmark(str(public_path))
                )
                require(", ".join(bench.operators) == "AND, OR, NOT", f"{method} operator order mismatch")
                groups = {}
                for i, question in enumerate(public_questions):
                    group = out / method / f"q{i}"
                    rows = run_sde_group(
                        bench, group, question, method, llm, protocol, "live", digest(out / "manifest.json")
                    )
                    groups[question["observation_set_id"]] = rows
                    print("COMPLETE", method, i, flush=True)
                metrics = score_sde(method, groups, ids, dataset, n_samples, n_queries)
                write_method_bundle(
                    out,
                    method,
                    metrics,
                    {"groups": {qid: groups[qid] for qid in ids}},
                )
                summary["methods"][method] = metrics
                print(
                    "METHOD_DONE", method,
                    "macro_recovery", round(metrics["macro_recovery"], 4),
                    "truncation_rate", round(metrics["truncation_rate"], 4),
                    flush=True,
                )

            summary_path = out / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "completed_methods": sorted(summary["methods"]),
                    "summary": str(summary_path),
                },
            )

        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        atomic_json(status_path, {"status": "complete", "summary": str(summary_path)})
        print("SUMMARY", summary_path, flush=True)


if __name__ == "__main__":
    main()
