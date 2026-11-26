"""Read a prefix of finalized FAAE rows; no model, validation or reference input."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from member_a_structure import features


def summary(rows):
    n=len(rows)
    c=Counter()
    first=None; streak=longest=0
    for r in rows:
        fb=r["feedback"]
        c["valid"]+=int(fb["is_valid"]); c["novel"]+=int(fb["is_new_valid"])
        c["duplicate"]+=int(fb["is_duplicate_valid"])
        c["invalid_parsed"]+=int(bool(r["candidate"]) and not fb["is_valid"])
        c["truncated"]+=int(r["status"]=="truncated")
        c["targeted"]+=int(r["target_operator_count"] is not None)
        c["compliant"]+=int(r["obeyed_operator_count"] is True)
        c["comparable"]+=int(r["obeyed_operator_count"] is not None)
        if fb["is_valid"] and first is None:first=r["query_index"]+1
        streak=0 if fb["is_new_valid"] else streak+1
        longest=max(longest,streak)
    return dict(slots=n,**c,valid_rate=c["valid"]/n,novel_yield=c["novel"]/n,duplicate_per_slot=c["duplicate"]/n,
                compliance_all=c["compliant"]/c["targeted"] if c["targeted"] else None,
                first_valid_slot=first,longest_run_without_new=longest,
                status=dict(Counter(r["status"] for r in rows)),failure_reason=dict(Counter(r["feedback"]["failure_reason"] for r in rows if r["feedback"]["failure_reason"])))


def strata(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[str(key(r))].append(r)
    return {k:summary(v) for k,v in groups.items()}


def diagnose(base,out):
    m=json.loads((base/"manifest.json").read_text()); groups=[]; evidence={}
    for method in m["methods"]:
        for i,q in enumerate(m["question_ids"]):
            rows=[]
            for directory in (base,base/"recovery_v1"):
                path=directory/method["label"]/f"q{i}"/"candidates.jsonl"
                if not path.exists():continue
                data=path.read_bytes(); data=data[:data.rfind(b"\n")+1]
                evidence[path.relative_to(base).as_posix()]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                rows.extend(json.loads(line) for line in data.splitlines())
            if not rows:continue
            assert [r["query_index"] for r in rows]==list(range(len(rows))) and all(r["observation_set_id"]==q for r in rows)
            for r in rows:r["structure"]=features(r)
            groups.append(dict(method=method["label"],question=q,**summary(rows),
                syntax=strata(rows,lambda r:r["structure"]["syntax_root"]),mechanism=strata(rows,lambda r:r["structure"]["mechanism_root"]),
                target_k=strata(rows,lambda r:r["target_operator_count"]),actual_k=strata(rows,lambda r:r["actual_operator_count"]),
                target_actual=strata(rows,lambda r:(r["target_operator_count"],r["actual_operator_count"])),
                k_root=strata(rows,lambda r:(r["actual_operator_count"],r["structure"]["root_family"]))))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf8") as f:json.dump(dict(groups=groups,source_prefix=evidence),f,indent=2)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("base",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();diagnose(a.base,a.out)

