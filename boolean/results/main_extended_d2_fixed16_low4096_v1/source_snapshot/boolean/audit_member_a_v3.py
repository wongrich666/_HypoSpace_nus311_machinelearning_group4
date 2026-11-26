"""Offline comparison against the actual frozen FAAE sources and recorded prompts."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def probe(source,role,experiment,diag,dest):
    sys.path.insert(0,str(source))
    from boolean_benchmark import BooleanBenchmarkRefined
    from boolean_dataset import BooleanDiscoveryGame,BooleanObservation
    from modules.exploration_controller import BooleanAdaptiveController
    cls=BooleanBenchmarkRefined
    if role=="A":
        from member_a_space import PublicSpaceBenchmark
        cls=PublicSpaceBenchmark
    b=cls(str(experiment/"public_tasks.json"))
    m=read(experiment/"manifest.json")
    forms={e.formula for e in BooleanDiscoveryGame.generate_all_expressions(b.variables,b.operators,b.max_depth,b.mechanistic_opts)}
    forms.update(("NOT x OR y OR (x AND y)","z","x OR x","x AND y AND x","NOR(x,y)","NOT NOT x"))
    grouped={}
    for path,h in read(diag)["source_prefix"].items():
        content=(experiment/path).read_bytes()[:h["bytes"]]
        assert hashlib.sha256(content).hexdigest()==h["sha256"]
        method,qindex=Path(path).parts[-3:-1]
        rows=list(map(json.loads,content.splitlines()))
        grouped.setdefault((method,qindex),[]).extend(rows)
        forms.update(r["candidate"] for r in rows if r["candidate"])
    observations={q["observation_set_id"]:[BooleanObservation(o["inputs"],o["output"]) for o in q["observations"]] for q in b.all_observation_sets}
    outcomes=[(f,repr(b.get_expression_mechanistic_key(f)),b._in_space(f),[b.validate_expression(f,obs)[0] for obs in observations.values()]) for f in sorted(forms)]
    seeds={q:int(hashlib.sha256(json.dumps([42,q]).encode()).hexdigest()[:16],16) for q in m["question_ids"]}
    prompt_match={};decision_match={};feedback_diff=0
    for (method,qi),rows in grouped.items():
        q=m["question_ids"][int(qi[1:])]; history=[]; seen=set()
        ctrl=BooleanAdaptiveController(3,seed=seeds[q])
        for r in rows:
            f=r["candidate"];fb=r["feedback"]
            if method in ("memory","adaptive"):
                key=q+":"+method
                p=b.create_prompt(observations[q],history,r["target_operator_count"])
                pm=prompt_match.setdefault(key,[0,0]);pm[0]+=int(p==r["prompt"]);pm[1]+=1
                if method=="adaptive":
                    chosen=ctrl.select_action({"query_index":r["query_index"],"n_unique_valid":len(seen)})
                    dm=decision_match.setdefault(q,[0,0]);dm[0]+=int(chosen==r["target_operator_count"]);dm[1]+=1
                    ctrl.update(chosen,fb)
            if f:history.append(f)
            valid=bool(f and b.validate_expression(f,observations[q])[0])
            official=repr(b.get_expression_mechanistic_key(f)) if f and b.get_expression_mechanistic_key(f) is not None else None
            new=valid and official not in seen
            feedback_diff+=int((valid,new,int(new),official)!=(fb["is_valid"],fb["is_new_valid"],fb["reward"],fb["mechanistic_key"]))
            if valid:seen.add(official)
    out=dict(seeds=seeds,public_key_count=len(b.public_space_keys),formulas=len(forms),question_validations=len(forms)*len(observations),
             validator_digest=hashlib.sha256(json.dumps(outcomes).encode()).hexdigest(),prompt_match=prompt_match,
             adaptive_action_match=decision_match,observed_feedback_mismatches=feedback_diff)
    dest.write_text(json.dumps(out,indent=2),encoding="utf8")


def main():
    here=Path(__file__).resolve().parent
    if sys.argv[1]=="probe":
        probe(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]),Path(sys.argv[6]));return
    exp=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
    original=exp/"recovery_v1/frozen_source"
    m=read(exp/"manifest.json"); t=read(here/"experiments/development_budget.json")
    assert all(digest(original/k)==v for k,v in m["source_sha256"].items())
    assert all(t[k]==m["model_settings"][k] for k in m["model_settings"])
    assert (t["dataset_sha256"],t["question_ids"],t["adaptive_parameters"],t["seed"],t["n_queries"],t["max_attempts_per_slot"])==(m["dataset_sha256"],m["question_ids"],m["adaptive_parameters"],m["seed"],m["budget_per_question_method"],m["max_attempts_per_slot"])
    for role,src in [("A",original),("D",here)]:
        subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"probe",str(src),role,str(exp),str(out/"diagnostic.json"),str(out/f"audit_{role}.json")],
                       check=True,env={**os.environ,"PYTHONHASHSEED":"1","PYTHONDONTWRITEBYTECODE":"1"})
    a=read(out/"audit_A.json");d=read(out/"audit_D.json")
    assert a==d and d["observed_feedback_mismatches"]==0
    assert all(n==k for n,k in [*d["adaptive_action_match"].values(),*d["prompt_match"].values()])
    result={"role_outputs_identical":True,"actual_frozen_source_verified":True,"dataset_model_budget_seed_ids_match":True,
            "byte_identical":{name:(original/name).read_bytes().replace(b"\r\n",b"\n")==(here/name).read_bytes().replace(b"\r\n",b"\n")
            for name in ("boolean_dataset.py","modules/llm_interface.py","modules/generation_executor.py")},
            "observed_matches":d}
    (out/"compatibility.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("Validated",d["question_validations"],"formula/question decisions",d["formulas"],"formulas")


if __name__=="__main__":main()

