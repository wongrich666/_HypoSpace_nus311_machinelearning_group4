"""Run the v3 Adaptive/SDE ablation; planning is the default and has no network access."""
import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from member_a_journal import (Transport, append_json, atomic_json, digest, exclusive, immutable,
                              inspect, jsonl, read_json, reconcile_row, require, RecoveryFault)
from boolean_benchmark import setup_llm
from modules.exploration_controller import BooleanAdaptiveController
from member_a_memory_sde_benchmark import MemorySDEBenchmark as SDEBenchmark
from member_a_memory_sde_metrics import evaluate

ROOT=Path(__file__).resolve().parent
TEAM=ROOT/"experiments/development_budget.json"
DATA=ROOT/"datasets/boolean_extended_d2.json"
METHODS=("memory_sde",)
SOURCES=("boolean_benchmark.py","boolean_dataset.py","modules/exploration_controller.py",
         "modules/generation_executor.py","modules/llm_interface.py","experiments/development_budget.json",
         "experiments/adaptive_strategy_manifest.json","member_a_journal.py","member_a_structure.py",
         "member_a_sde_benchmark.py","member_a_sde_metrics.py","run_member_a_sde.py",
         "member_a_memory_structure.py",
         "member_a_memory_sde_benchmark.py",
         "member_a_memory_sde_metrics.py",
         "run_member_a_memory_sde.py")


class OfflineFixture:
    def __init__(self):self.i=0
    def query_with_usage(self,prompt):
        self.i+=1
        outputs=("Expression: x OR y","Expression: y OR x","Expression: NOT (x OR y)","unparseable","Expression: y","Expression: x")
        return dict(response=outputs[(self.i-1)%len(outputs)],finish_reason="length" if self.i%7==0 else "stop",
                    usage=dict(prompt_tokens=3,completion_tokens=2,total_tokens=5))


def public_dataset(data,ids):
    metadata={k:data["metadata"][k] for k in ("variables","operators","max_depth","mechanistic_opts")}
    lookup={q["observation_set_id"]:q for g in data["datasets_by_n_observations"].values() for q in g}
    questions=[]
    for qid in ids:
        q={k:lookup[qid][k] for k in ("observation_set_id","observations","n_observations")}
        q["ground_truth_expressions"]=[]
        questions.append(q)
    return dict(metadata=metadata,datasets_by_n_observations={"public_only":questions})


def run_group(bench, group, question, method, client, team, mode, manifest_hash):
    qid=question["observation_set_id"]
    seed=None
    ctrl=None
    group.mkdir(parents=True,exist_ok=True)
    rows=jsonl(group/"candidates.jsonl")
    pending=inspect(rows,jsonl(group/"attempts.jsonl"),qid)
    transport=Transport(client,group/"attempts.jsonl",len(rows),pending)
    def sink(row):
        if transport.pending:raise RecoveryFault("Unconsumed saved API attempts")
        row.update(method=method,mode=mode,controller_seed=seed)
        if transport.unknown:
            reconcile_row(row,transport.unknown,{"protocol_sha256":manifest_hash},
                          group/f"interruption_slot_{row['query_index']+1}.json",method)
            transport.unknown=None
        append_json(group/"candidates.jsonl",row)
        transport.client.slot+=1;transport.client.attempt=0
        if row["status"]=="api_error": raise RuntimeError("API error slot persisted; inspect provider")
    bench.evaluate_group(question,transport,ctrl,method=="memory_sde",rows,sink)
    all_rows=jsonl(group/"candidates.jsonl")
    require(len(all_rows)==16 and not inspect(all_rows,jsonl(group/"attempts.jsonl"),qid),"Incomplete group")
    return all_rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run",action="store_true");modes.add_argument("--live",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv);out=a.out.resolve(); team=read_json(TEAM)
    require(os.environ.get("PYTHONHASHSEED")=="1","Start Python with PYTHONHASHSEED=1 to match the original operator order")
    require(digest(DATA)==team["dataset_sha256"] and team["protocol_id"].endswith("v3-public-space"),"Wrong dataset or protocol")
    ids=team["question_ids"]
    require(len(set(ids))==5 and (team["seed"],team["n_queries"],team["max_attempts_per_slot"])==(42,16,3),"Frozen budget changed")
    mode="offline_fixture" if a.dry_run else "live"
    manifest=dict(team=team,methods=list(METHODS),mode=mode,slots=80,max_attempts=240,python=platform.python_version(),
                  pythonhashseed=1,source_sha256={s:digest(ROOT/s) for s in SOURCES},
                  guidance="global minimum generated + guided - compliant_new, ties VAR NOT AND OR",
                  controller_seed=None,
                  history="raw",feasibility="off",provider_seed=None,unknown_response="one slot, zero reward, null tokens/cost, no resend")
    if not out.exists():out.mkdir(parents=True,exist_ok=False)
    elif not (out/"manifest.json").exists() and any(out.iterdir()):raise ValueError("Refusing unrelated output directory")
    with exclusive(out/"run.lock"):
        immutable(out/"manifest.json",manifest)
        for name,h in manifest["source_sha256"].items():
            path=out/"source_snapshot"/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():shutil.copyfile(ROOT/name,path)
            require(digest(path)==h,"Source snapshot differs")
        public=public_dataset(read_json(DATA),ids)
        immutable(out/"public_tasks.json",public)
        status_path=out/"status.json"
        if not status_path.exists():immutable(status_path,dict(status="prepared",mode=mode))
        status=read_json(status_path)["status"]
        if not (a.dry_run or a.live):return
        if status=="complete":return
        require(status=="prepared" or a.resume,"Use --resume for an existing run")
        questions=public["datasets_by_n_observations"]["public_only"]
        bench=SDEBenchmark(str(out/"public_tasks.json"))
        require(", ".join(bench.operators)=="AND, OR, NOT","Prompt operator order mismatch")
        for i,q in enumerate(questions):
            for method in METHODS:
                d=out/method/f"q{i}";inspect(jsonl(d/"candidates.jsonl"),jsonl(d/"attempts.jsonl"),q["observation_set_id"])
        config={k:team[k] for k in ("model","base_url","temperature","max_tokens","reasoning_effort","request_timeout")}
        client=OfflineFixture() if a.dry_run else setup_llm("dashscope",**config)
        try:
            atomic_json(status_path,dict(status="running",mode=mode))
            groups={}
            for i,q in enumerate(questions):
                for method in METHODS:
                    groups[method,q["observation_set_id"]]=run_group(bench,out/method/f"q{i}",q,method,client,team,mode,digest(out/"manifest.json"))
            metrics=evaluate(groups,ids,read_json(DATA))
            immutable(out/"metrics.json",dict(mode=mode,measures_real_model=a.live,methods=metrics))
            atomic_json(status_path,dict(status="complete",mode=mode,slots=80,actual_api_attempts=0 if a.dry_run else sum(v["counts"]["attempts"] for v in metrics.values())))
        except BaseException:
            atomic_json(status_path,dict(status="incomplete",mode=mode));raise


if __name__=="__main__":main()

