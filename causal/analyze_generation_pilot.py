"""Analyze a paired generation pilot using ground truth only after generation."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_questions(dataset):
    if "datasets_by_n_observations" in dataset:
        groups = dataset["datasets_by_n_observations"].values()
        return {q["observation_set_id"]: q for group in groups for q in group}
    return {q["observation_set_id"]: q
            for q in dataset.get("datasets", dataset.get("sampled_datasets", []))}


def summarize(report, questions):
    target = report["target_edge_count"]
    by_arm = report["arms"]
    summary = {}
    for arm, data in by_arm.items():
        rows = data["results"]
        feasibility = []
        for row in rows:
            question = questions[row["question_id"]]
            valid_edge_counts = {
                len(graph["edges"]) for graph in question["ground_truth_graphs"]
            }
            feasibility.append(target in valid_edge_counts)
        summary[arm] = {
            "questions": len(rows),
            "feasible_target_questions": sum(feasibility),
            "valid_on_feasible": sum(
                feasible and row["valid"] is True
                for row, feasible in zip(rows, feasibility)
            ),
            "target_match_on_feasible": sum(
                feasible and row["matches_requested_count"]
                for row, feasible in zip(rows, feasibility)
            ),
            "valid_on_infeasible": sum(
                not feasible and row["valid"] is True
                for row, feasible in zip(rows, feasibility)
            ),
            "statuses_on_infeasible": dict(Counter(
                row["status"] for row, feasible in zip(rows, feasibility)
                if not feasible
            )),
            "total_tokens": data["usage"]["total_tokens"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    questions = load_questions(json.loads(args.dataset.read_text(encoding="utf-8")))
    print(json.dumps(summarize(report, questions), indent=2))


if __name__ == "__main__":
    main()
