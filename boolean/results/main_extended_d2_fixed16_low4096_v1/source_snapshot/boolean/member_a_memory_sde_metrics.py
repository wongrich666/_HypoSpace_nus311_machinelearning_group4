"""Offline scoring entry point. Reference keys enter only after all 160 rows exist."""
import ast
from collections import Counter
from statistics import mean
from member_a_journal import require, TOKENS, UNKNOWN


def evaluate(groups, ids, dataset):
    require(len(groups)==5 and all(len(g)==16 for g in groups.values()),"Scoring requires all 80 slots")
    refs={q["observation_set_id"]:{ast.literal_eval(e["mechanistic_key"]) for e in q["ground_truth_expressions"]}
          for block in dataset["datasets_by_n_observations"].values() for q in block if q["observation_set_id"] in ids}
    metrics={}
    for method in ("memory_sde",):
        questions={}; counts=Counter(); actual=Counter(); suggested=Counter(); joined=Counter(); known=Counter()
        for q in ids:
            seen=set();curve=[];first=None
            for i,r in enumerate(groups[method,q],1):
                f=r["feedback"]; feature=r["structure"]; h=feature["root_family"]; s=feature["suggested_root"]
                require(r["query_index"]==i-1 and r["observation_set_id"]==q,"Slot identity mismatch")
                counts["slots"]+=1
                if f["is_valid"]:
                    key=ast.literal_eval(f["mechanistic_key"])
                    require(f["is_new_valid"]==(key not in seen),"Novelty disagreement")
                    seen.add(key); first=first or i
                counts["valid"]+=int(f["is_valid"]); counts["novel"]+=int(f["is_new_valid"])
                counts["duplicates"]+=int(f["is_duplicate_valid"]); counts["truncated"]+=int(r["status"]=="truncated")
                counts["invalid_parsed"]+=int(bool(r["candidate"]) and not f["is_valid"])
                counts["op_complied"]+=int(r["obeyed_operator_count"] is True)
                counts["op_measurable"]+=int(r["obeyed_operator_count"] is not None)
                if h:actual[h]+=1
                if s:
                    suggested[s]+=1
                    counts["root_complied"]+=int(feature["root_compliance"] is True)
                    if feature["root_compliance"] is True:
                        joined[s]+=int(f["is_new_valid"])
                for a in r["attempts"]:
                    counts["attempts"]+=1
                    if a["status"]==UNKNOWN or not (a.get("usage") or {}).get("total_tokens"):counts["missing_usage"]+=1
                    for k in TOKENS:known[k]+=(a.get("usage") or {}).get(k) or 0
                curve.append(len(seen & refs[q]))
            questions[q]=dict(found=curve[-1],total=len(refs[q]),curve=curve,first_valid_slot=first,
                              recovery_curve=[n/len(refs[q]) for n in curve])
        cs=[d["recovery_curve"] for d in questions.values()]
        found=sum(d["found"] for d in questions.values())
        tokens=None if counts["missing_usage"] else known["total_tokens"]
        ratio=lambda n,d:n/d if d else None
        metrics[method]=dict(questions=questions,counts=dict(counts),macro_recovery=mean(c[-1] for c in cs),
            micro_recovery=found/sum(len(refs[q]) for q in ids),recovery_auc=mean(mean(c) for c in cs),
            recovery_at={str(k):mean(c[k-1] for c in cs) for k in (4,8,16)},actual_root_distribution=dict(actual),
            root_family_coverage=len(actual),suggested_root_distribution=dict(suggested),
            compliant_new_yield_by_suggestion={s:joined[s]/n for s,n in suggested.items()},
            root_compliance_all_guided=ratio(counts["root_complied"],sum(suggested.values())),
            operator_compliance_all=None,operator_compliance_given_parsed=ratio(counts["op_complied"],counts["op_measurable"]),
            duplicate_rate_among_valid=ratio(counts["duplicates"],counts["valid"]),invalid_parsed_per_slot=counts["invalid_parsed"]/80,
            nonvalid_per_slot=(80-counts["valid"])/80,truncation_rate=counts["truncated"]/80,
            observed_usage=dict(known),total_tokens=tokens,tokens_per_new_valid=ratio(tokens,counts["novel"]) if tokens is not None else None,
            tokens_per_recovered=ratio(tokens,found) if tokens is not None else None)
    return metrics

