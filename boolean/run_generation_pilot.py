"""One-question Boolean prompt check for Member C; not a formal benchmark."""

import argparse
import json
import os
from pathlib import Path

from boolean_benchmark import BooleanBenchmarkRefined, setup_llm, load_config
from boolean_dataset import BooleanObservation
from modules.generation_executor import generate_boolean_one


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-operator-count", type=int, required=True)
    parser.add_argument("--question-id", help="Use this observation set; otherwise sample by seed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--output", help="Optional local JSON file; do not commit pilot logs")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without an API call")
    args = parser.parse_args(argv)
    if args.target_operator_count < 0 or args.max_attempts < 1:
        parser.error("target-operator-count must be nonnegative and max-attempts positive")

    benchmark = BooleanBenchmarkRefined(args.dataset)
    if args.question_id:
        matches = [item for item in benchmark.all_observation_sets
                   if item["observation_set_id"] == args.question_id]
        if len(matches) != 1:
            parser.error("question-id must match exactly one observation set")
        question = matches[0]
    else:
        question = benchmark.sample_observation_sets(1, seed=args.seed)[0]
    observations = [BooleanObservation(o["inputs"], o["output"])
                    for o in question["observations"]]

    if args.dry_run:
        print(benchmark.create_prompt(
            observations, [], target_complexity=args.target_operator_count,
        ))
        return 0

    config = load_config(args.config)
    llm_config = config.get("llm", {})
    llm_type = llm_config.get("type", "dashscope")
    model = llm_config.get("models", {}).get(llm_type)
    key_name = {"dashscope": "DASHSCOPE_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(llm_type)
    api_key = llm_config.get("api_keys", {}).get(llm_type) or (
        os.environ.get(key_name) if key_name else None
    )
    if not api_key:
        parser.error(f"missing API key: set {key_name or 'the provider key'}")
    llm = setup_llm(
        llm_type, model=model, api_key=api_key,
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2048),
        base_url=llm_config.get("base_url"),
        request_timeout=llm_config.get("request_timeout", 60.0),
    )
    generation = generate_boolean_one(
        benchmark, llm, observations, [], args.target_operator_count,
        max_attempts=args.max_attempts,
    )
    valid = None
    if generation.candidate is not None:
        valid = benchmark.validate_expression(generation.candidate, observations)[0]
    record = {
        "purpose": "member_c_development_check",
        "question_id": question["observation_set_id"],
        "model": model,
        "temperature": llm_config.get("temperature", 0.7),
        "seed_for_question_selection": args.seed if not args.question_id else None,
        "status": generation.status,
        "target_operator_count": generation.target_operator_count,
        "actual_operator_count": generation.actual_operator_count,
        "obeyed_operator_count": generation.obeyed_operator_count,
        "candidate": generation.candidate,
        "valid": valid,
        "usage": generation.usage,
        "api_attempt_count": len(generation.attempts),
        "attempts": generation.attempts,
    }
    print(json.dumps(record, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
