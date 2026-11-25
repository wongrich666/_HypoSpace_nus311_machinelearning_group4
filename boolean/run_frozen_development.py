"""Fixed-budget Boolean extended_d2 development run. Does not overwrite old results."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import argparse
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOLEAN = REPO / "boolean"
sys.path.insert(0, str(BOOLEAN))
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(key, None)

from boolean_benchmark import BooleanBenchmarkRefined, setup_llm
from modules.experiment_manifest import freeze_questions
from modules.llm_interface import direct_http_session

SOURCE_FILES = [
    "boolean/boolean_benchmark.py",
    "boolean/boolean_dataset.py",
    "boolean/modules/llm_interface.py",
    "boolean/modules/generation_executor.py",
    "boolean/modules/exploration_controller.py",
    "boolean/modules/experiment_manifest.py",
    "boolean/test_nor_validation.py",
    "boolean/test_reasoning_effort.py",
    "boolean/run_frozen_development.py",
    "boolean/experiments/development_budget.json",
]
BUDGET_PATH = BOOLEAN / "experiments" / "development_budget.json"
DATASET = BOOLEAN / "datasets" / "boolean_extended_d2.json"
METHODS = ["memory", "uniform", "round_robin", "adaptive"]
SELECTABLE = METHODS + ["stratified"]


def api_key():
    if os.environ.get("DASHSCOPE_API_KEY"):
        return os.environ["DASHSCOPE_API_KEY"]
    raise SystemExit("Set DASHSCOPE_API_KEY in the environment before running")


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def source_version():
    """Hash the working tree. Git HEAD does not contain these uncommitted edits."""
    files = {}
    combined = hashlib.sha256()
    for relative in SOURCE_FILES:
        data = (REPO / relative).read_bytes()
        work_hash = hashlib.sha256(data).hexdigest()
        try:
            head_data = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
            head_hash = hashlib.sha256(head_data).hexdigest()
        except Exception:
            head_hash = None
        files[relative] = {
            "worktree_sha256": work_hash,
            "head_sha256": head_hash,
            "differs_from_head": work_hash != head_hash,
        }
        combined.update(relative.encode())
        combined.update(b"\0")
        combined.update(work_hash.encode())
    return {
        "git_head": git_revision(),
        "git_head_contains_these_edits": not any(item["differs_from_head"] for item in files.values()),
        "combined_worktree_sha256": combined.hexdigest(),
        "files": files,
    }


def save_snapshot(out, version):
    snapshot = out / "source_snapshot"
    if snapshot.exists():
        return
    snapshot.mkdir(parents=True)
    for relative in SOURCE_FILES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    (out / "source_version.json").write_text(json.dumps(version, indent=2) + "\n", encoding="utf-8")


def load_budget():
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    if budget["planned_methods"] != METHODS or budget["candidate_slots_total"] != 320:
        raise SystemExit("development budget is not the approved 4-method, 320-slot protocol")
    if budget["n_samples"] != 5 or budget["n_queries"] != 16 or budget["seed"] != 42:
        raise SystemExit("development budget does not match the frozen validation settings")
    if budget["max_attempts_per_slot"] != 3:
        raise SystemExit("Frozen protocol requires at most three attempts per slot")
    return budget


def settings_from(budget):
    return {
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "dataset": "datasets/boolean_extended_d2.json",
        "n_samples": budget["n_samples"],
        "n_queries": budget["n_queries"],
        "seed": budget["seed"],
        "methods": budget["planned_methods"],
        "candidate_slots_total": budget["candidate_slots_total"],
        "model": budget["model"],
        "temperature": budget["temperature"],
        "max_tokens": budget["max_tokens"],
        "max_attempts_per_slot": budget["max_attempts_per_slot"],
        "request_timeout": budget["request_timeout"],
        "reasoning_effort": budget["reasoning_effort"],
        "base_url": budget["base_url"],
        "adaptive_parameters": budget["adaptive_parameters"],
        "method_definitions": budget["method_definitions"],
        "actions": [0, 1, 2, 3],
        "validator": "fixed_not_or_composition",
        "code_revision": git_revision(),
        "source_version": source_version(),
    }


def probe(key, base_url, model):
    session = direct_http_session()
    response = session.get(
        base_url + "/models",
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


def load_log(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-queries', type=int, required=True)
    parser.add_argument('--questions', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--methods', nargs='+', choices=SELECTABLE, default=METHODS)
    parser.add_argument('--dry-run', action='store_true', help='Validate protocol without API requests')
    args = parser.parse_args()
    out = args.out.resolve()
    budget = load_budget()
    if args.n_queries != budget['n_queries'] or budget['max_tokens'] != 4096:
        raise SystemExit('Frozen protocol requires --n-queries 16 and max_tokens 4096')
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != budget['dataset_sha256']:
        raise SystemExit('Dataset fingerprint differs from the frozen protocol')
    indices, methods = args.questions, args.methods
    if len(set(indices)) != len(indices) or len(set(methods)) != len(methods):
        raise SystemExit('Duplicate question indices or methods are not allowed')
    out.mkdir(parents=True, exist_ok=True)
    benchmark = BooleanBenchmarkRefined(str(DATASET))
    if benchmark.max_operator_count != 3:
        raise SystemExit(f"expected public operator bound 3, got {benchmark.max_operator_count}")
    manifest_path = out / "manifest.json"
    settings = settings_from(budget)
    if settings["reasoning_effort"] != "low" or settings["request_timeout"] != 60:
        raise SystemExit("reasoning effort and timeout stay low and 60 seconds")
    if settings["max_tokens"] != budget["max_tokens"]:
        raise SystemExit("manifest max_tokens does not match the requested limit")
    settings["protocol_id"] = budget["protocol_id"]
    settings["seed_scope"] = budget["seed_scope"]
    if settings["source_version"]["git_head_contains_these_edits"]:
        print("SOURCE_MATCHES_HEAD", settings["code_revision"], flush=True)
    else:
        print("SOURCE_AHEAD_OF_HEAD", settings["code_revision"], flush=True)
    save_snapshot(out, settings["source_version"])
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "settings": settings,
            "question_ids": budget["question_ids"],
        }, indent=2) + "\n", encoding="utf-8")
    questions, manifest = freeze_questions(benchmark, manifest_path, settings)
    if manifest["question_ids"] != budget["question_ids"]:
        raise SystemExit("frozen question IDs differ from the archived 2048 run")
    print("FROZEN", manifest["question_ids"], flush=True)
    print(
        "MAX_TOKENS", settings["max_tokens"],
        "TIMEOUT", settings["request_timeout"],
        "REASONING_EFFORT", settings["reasoning_effort"],
        "SOURCE", settings["source_version"]["combined_worktree_sha256"],
        flush=True,
    )
    if args.dry_run:
        print('DRY_RUN_OK', indices, methods, flush=True)
        return
    key = api_key()
    probe(key, budget['base_url'], budget['model'])
    llm = setup_llm(
        "dashscope", model=budget["model"], api_key=key,
        temperature=budget["temperature"], max_tokens=settings["max_tokens"],
        base_url=budget["base_url"], request_timeout=budget["request_timeout"],
        reasoning_effort=budget["reasoning_effort"],
    )
    body = llm.request_body("parameter-check")
    if body.get("reasoning_effort") != "low" or body.get("max_tokens") != settings["max_tokens"] or llm.request_timeout != 60:
        raise SystemExit("client would not send the frozen request parameters")
    print("REQUEST_BODY", {key: body[key] for key in ("model", "temperature", "max_tokens", "reasoning_effort")}, "TIMEOUT", llm.request_timeout, flush=True)
    if llm.session.trust_env:
        raise SystemExit("generation client is still trusting proxy settings")
    for index in indices:
        for method in methods:
            result_path = out / f"q{index}_{method}.json"
            log_path = out / f"q{index}_{method}.jsonl"
            if result_path.exists():
                print("SKIP", result_path.name, flush=True)
                continue
            prior = load_log(log_path)
            if any(row.get('observation_set_id') != budget['question_ids'][index] for row in prior):
                raise SystemExit('Candidate log belongs to another question')
            if prior and prior[-1]["query_index"] + 1 != len(prior):
                raise SystemExit("candidate log is not a contiguous prefix")
            print("START", index, method, "resume_from", len(prior), flush=True)

            def sink(row, log_path=log_path, index=index, method=method):
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                usage = row.get("usage") or {}
                duration = sum(float(item.get("duration_seconds") or 0) for item in row.get("attempts") or [])
                print(
                    "PROGRESS", index, method, row["query_index"] + 1, row["status"],
                    "reward", row["feedback"]["reward"],
                    "completion", usage.get("completion_tokens", 0),
                    "attempts", len(row.get("attempts") or []),
                    "seconds", round(duration, 1),
                    flush=True,
                )
                if row["status"] == "api_error":
                    raise RuntimeError("API error after retries; stop before spending more")

            result = benchmark.run_benchmark(
                llm, n_queries_per_sample=budget["n_queries"], seed=budget["seed"],
                verbose=False, checkpoint_dir=str(out / "checkpoints"), max_retries=budget["max_attempts_per_slot"],
                run_id=f"q{index}_{method}_{uuid.uuid4().hex}",
                decoding=method, selected_questions=[questions[index]],
                on_candidate_complete=sink, prior_completions=prior,
                adaptive_parameters=budget["adaptive_parameters"],
            )
            samples = result.get('per_sample_results', [])
            if len(samples) != 1 or len(samples[0].get('request_log', [])) != budget['n_queries']:
                raise RuntimeError('Incomplete run: refusing to save as completed')
            result["experiment_manifest"] = manifest
            result["provider_base_url"] = budget["base_url"]
            result["resumed_logged_candidates"] = len(prior)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("COMPLETE", index, method, flush=True)


if __name__ == "__main__":
    main()
