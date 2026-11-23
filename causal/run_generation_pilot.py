"""Compare causal generation with and without an explicit edge-count target."""

import argparse
import json
from pathlib import Path
from random import Random

import yaml

from modules.generation_executor import generate_causal_one
from run_causal_benchmark import CausalBenchmarkEnhanced, build_cost_estimate, setup_llm


def run_pilot(benchmark, llm, questions, target_edge_count, max_attempts, pricing=None):
    """Give both arms one candidate slot per question and log API attempts separately."""
    results = {"unconstrained": [], "constrained": []}
    for index, question in enumerate(questions):
        observations = question["observations"]
        arms = list(results)
        if index % 2:
            arms.reverse()
        for arm in arms:
            target = target_edge_count if arm == "constrained" else None
            result = generate_causal_one(
                benchmark, llm, observations, [], target, max_attempts=max_attempts
            )
            graph = result.candidate
            results[arm].append({
                "question_id": question.get("observation_set_id", index),
                "call_order": arms.index(arm) + 1,
                "status": result.status,
                "target_edge_count": target,
                "actual_edge_count": result.actual_edge_count,
                "obeyed_edge_count": result.obeyed_edge_count,
                "matches_requested_count": result.actual_edge_count == target_edge_count,
                "valid": benchmark.validate_hypothesis(graph, observations) if graph is not None else None,
                "candidate": graph.to_dict() if graph is not None else None,
                "candidate_hash": graph.get_hash() if graph is not None else None,
                "response": result.response,
                "usage": result.usage,
                "attempts": result.attempts,
            })

    report = {}
    for arm, rows in results.items():
        usage = {key: sum(row["usage"][key] for row in rows)
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        report[arm] = {
            "candidate_slots": len(rows),
            "api_attempts": sum(len(row["attempts"]) for row in rows),
            "api_errors": sum(row["status"] == "api_error" for row in rows),
            "parsed": sum(row["status"] == "ok" for row in rows),
            "matching_edge_count": sum(row["matches_requested_count"] for row in rows),
            "valid": sum(row["valid"] is True for row in rows),
            "usage": usage,
            "cost_estimate": build_cost_estimate(
                pricing, usage["prompt_tokens"], usage["completion_tokens"]
            ),
            "results": rows,
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-edge-count", required=True, type=int)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=33550336)
    args = parser.parse_args(argv)
    if args.n_samples < 1 or args.max_attempts < 1:
        parser.error("sample and attempt counts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    # Use only public metadata and observations for the generation pilot.
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    metadata = dataset.get("metadata", {})
    benchmark = CausalBenchmarkEnhanced()
    benchmark.nodes = metadata.get("nodes", [])
    if not benchmark.nodes:
        parser.error("dataset metadata must list the graph nodes")
    dag_bound = len(benchmark.nodes) * (len(benchmark.nodes) - 1) // 2
    declared_max = metadata.get("max_edges")
    benchmark.max_edges = dag_bound if declared_max is None else min(declared_max, dag_bound)
    if not 0 <= args.target_edge_count <= benchmark.max_edges:
        parser.error(f"target edge count must be between 0 and {benchmark.max_edges}")
    if "datasets_by_n_observations" in dataset:
        pool = [question for group in dataset["datasets_by_n_observations"].values()
                for question in group]
    else:
        pool = dataset.get("datasets", dataset.get("sampled_datasets", []))
    if args.n_samples > len(pool):
        parser.error("dataset does not contain enough questions")
    questions = Random(args.seed).sample(pool, args.n_samples)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm_config = config["llm"]
    if llm_config["type"] != "dashscope":
        parser.error("this pilot expects a DashScope configuration")
    model = llm_config["models"]["dashscope"]
    llm = setup_llm(
        "dashscope", model=model, temperature=llm_config.get("temperature", 0.7),
        base_url=llm_config.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=llm_config.get("max_tokens", 2048),
        request_timeout=llm_config.get("request_timeout", 60.0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run_pilot(benchmark, llm, questions, args.target_edge_count,
                       args.max_attempts, config.get("pricing"))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({
            "model": model, "seed": args.seed, "dataset": str(args.dataset),
            "target_edge_count": args.target_edge_count, "max_attempts": args.max_attempts,
            "arms": report,
        }, output, indent=2)


if __name__ == "__main__":
    main()
