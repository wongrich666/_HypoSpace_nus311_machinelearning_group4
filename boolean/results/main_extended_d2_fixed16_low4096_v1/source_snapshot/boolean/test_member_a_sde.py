import ast
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import run_member_a_sde as run
from boolean_benchmark import BooleanBenchmarkRefined
from modules.exploration_controller import BooleanAdaptiveController
from member_a_structure import features, StructureHistory
from member_a_sde_benchmark import SDEBenchmark
from member_a_journal import UNKNOWN, jsonl, read_json


@pytest.fixture
def task(tmp_path):
    team=read_json(run.TEAM)
    data=run.public_dataset(read_json(run.DATA),team["question_ids"])
    assert all(q["ground_truth_expressions"]==[] for q in data["datasets_by_n_observations"]["public_only"])
    path=tmp_path/"public.json";path.write_text(json.dumps(data))
    return team,path,data["datasets_by_n_observations"]["public_only"][1]


@pytest.mark.parametrize("formula,syntax,key",[
    ("x","VAR","VAR"),("NOT (x OR y)","NOT","NOR"),("NOR(x,y)","NOR","NOR"),
    ("(NOT x) OR y","OR","OR"),("x OR x","OR","VAR"),("(x OR y) OR x","OR","OR"),
    ("y AND x","AND","AND"),("NOT NOT x","NOT","NOT")])
def test_structure(task,formula,syntax,key):
    bench=BooleanBenchmarkRefined(str(task[1]))
    actual=bench.get_expression_mechanistic_key(formula)
    f=features({"candidate":formula,"feedback":{"mechanistic_key":repr(actual)}})
    assert f["syntax_root"]==syntax and f["mechanism_root"]==key


def test_no_cross_credit_and_deterministic_failed_suggestion():
    state=StructureHistory()
    for i in range(6):
        root=state.choose(2)["suggested_root"]
        state.observe(2,root,{"candidate":None,"feedback":{"mechanistic_key":None,"is_new_valid":False}})
    assert all(state.counts[2,h]["guided"]==2 for h in ("AND","NOT","OR"))
    state.observe(2,"NOT",{"candidate":"x OR (NOT y)","actual_operator_count":2,
                           "feedback":{"mechanistic_key":None,"is_new_valid":True}})
    assert state.counts[2,"OR"]["new_valid"]==1 and not state.counts[2,"NOT"]["compliant_new"]


def test_off_matches_v3_and_on_accepts_off_action_exact_text(task,tmp_path):
    team,path,q=task
    seed=int(hashlib.sha256(json.dumps([42,q["observation_set_id"]]).encode()).hexdigest()[:16],16)
    class Fixed:
        def query_with_usage(self,prompt):
            return dict(response="Expression: y",usage=dict(total_tokens=2))
    official=BooleanBenchmarkRefined(str(path)).evaluate_single_observation_set(Fixed(),q,n_queries=16,verbose=False,
        controller=BooleanAdaptiveController(3,seed=seed),max_retries=3)
    bench=SDEBenchmark(str(path))
    baseline=run.run_group(bench,tmp_path/"off",q,"adaptive",Fixed(),team,"offline_fixture","test")
    for a,b in zip(official["request_log"],baseline):
        for k in ("prompt","decision_log","feedback","target_operator_count","candidate","response"):
            assert a[k]==b[k]
    guided=run.run_group(bench,tmp_path/"on",q,"adaptive_sde",Fixed(),team,"offline_fixture","test")
    assert all(r["candidate"]=="y" and r["feedback"]["is_valid"] for r in guided)
    assert sum(r["feedback"]["reward"] for r in guided)==1
    assert any(r["structure"]["root_compliance"] is False for r in guided)
    for a,b in zip(baseline,guided):
        assert a["decision_log"]==b["decision_log"] and b["prompt"].startswith(a["prompt"]+"\nWithin")
    assert guided[0]["structure"]["history"]==StructureHistory().choose(guided[0]["target_operator_count"])["history"]


def test_resume_preserves_structure_and_unknown_slot(task,tmp_path):
    team,path,q=task
    class Client:
        def __init__(self,interrupt):self.n=0;self.interrupt=interrupt
        def query_with_usage(self,prompt):
            self.n+=1
            if self.n==3:
                if self.interrupt:raise KeyboardInterrupt()
                return dict(response="",finish_reason="length",usage=dict(total_tokens=1))
            return dict(response="Expression: y",usage=dict(total_tokens=1))
    def group(d,c):
        return run.run_group(SDEBenchmark(str(path)),tmp_path/d,q,"adaptive_sde",c,team,"offline_fixture","test")
    reference=group("ref",Client(False))
    c=Client(True)
    with pytest.raises(KeyboardInterrupt):group("res",c)
    rows=group("res",c)
    assert len(rows)==c.n==16 and rows[2]["status"]==UNKNOWN and rows[2]["usage"]["total_tokens"] is None
    for a,b in zip(reference[3:],rows[3:]):
        assert a["prompt"]==b["prompt"] and a["structure"]==b["structure"] and a["decision_log"]==b["decision_log"]
    group("res",c)
    assert c.n==16 and len([e for e in jsonl(tmp_path/"res/attempts.jsonl") if e["event"]=="started"])==16


def test_retry_cap_and_failures(task,tmp_path):
    team,path,q=task
    class Client:
        calls=0
        def query_with_usage(self,prompt):
            self.calls+=1
            return {"response":"Error querying: HTTP 503","retryable":True}
    c=Client()
    with patch("modules.generation_executor.sleep"),pytest.raises(RuntimeError,match="API error slot"):
        run.run_group(SDEBenchmark(str(path)),tmp_path/"r",q,"adaptive_sde",c,team,"offline_fixture","test")
    assert c.calls==3 and len(jsonl(tmp_path/"r/candidates.jsonl"))==1



def test_saved_exception_replay_preserves_error(tmp_path):
    from member_a_journal import JournalClient, Transport, inspect, jsonl

    class TimeoutClient:
        def query_with_usage(self, prompt):
            raise TimeoutError("provider timed out")

    path = tmp_path / "attempts.jsonl"
    journal = JournalClient(TimeoutClient(), path, 0)

    with pytest.raises(TimeoutError):
        journal.query_with_usage("test prompt")

    events = jsonl(path)
    pending = inspect([], events, "test-question")
    replay = Transport(TimeoutClient(), path, 0, pending)

    with pytest.raises(Exception) as exc:
        replay.query_with_usage("test prompt")

    assert type(exc.value).__name__ == "TimeoutError"
    assert str(exc.value) == "provider timed out"
    assert jsonl(path) == events
