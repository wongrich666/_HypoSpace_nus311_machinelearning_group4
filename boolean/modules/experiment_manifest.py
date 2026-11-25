"""Freeze the same Boolean questions and public settings across methods."""
import json
from pathlib import Path


def freeze_questions(benchmark, path, settings):
    path = Path(path)
    pool = benchmark.all_observation_sets
    ids = [question.get("observation_set_id") for question in pool]
    if any(item is None for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset requires unique observation_set_id values")
    by_id = dict(zip(ids, pool))
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("settings") != settings:
            raise ValueError("Experiment settings differ from the frozen manifest")
        selected_ids = manifest["question_ids"]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Invalid frozen question IDs")
        if any(item not in by_id for item in selected_ids):
            raise ValueError("Frozen question is missing from the dataset")
    else:
        selected = benchmark.sample_observation_sets(settings["n_samples"], settings["seed"])
        selected_ids = [question["observation_set_id"] for question in selected]
        if len(selected_ids) != settings["n_samples"]:
            raise ValueError("Could not freeze the requested number of questions")
        manifest = {"schema_version": 1, "settings": settings, "question_ids": selected_ids}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    return [by_id[item] for item in selected_ids], manifest
