import sys, os, json, argparse, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_snapshot'/'boolean'))
from member_a_journal import *
from run_member_a_sde import public_dataset, OfflineFixture
from boolean_benchmark import setup_llm
from modules.exploration_controller import BooleanAdaptiveController
from member_a_sde_benchmark import SDEBenchmark
from member_a_memory_sde_benchmark import MemorySDEBenchmark
def run_group(bench, group, question, method, client, team, mode, manifest_hash):
    qid=question["observation_set_id"]
    seed=int(hashlib.sha256(json.dumps([team["seed"],qid]).encode()).hexdigest()[:16],16)
    params=team["adaptive_parameters"]
    ctrl=BooleanAdaptiveController(3,seed=seed,strategy=(method if method in ("uniform","round_robin") else "adaptive"),forget_factor=params["forget_factor"],
        temperature=params["controller_temperature"],explore_bonus=params["explore_bonus"])
    if method in ("memory", "memory_sde"):
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
    bench.evaluate_group(question,transport,ctrl,method in ("adaptive_sde","memory_sde"),rows,sink)
    all_rows=jsonl(group/"candidates.jsonl")
    require(len(all_rows)==16 and not inspect(all_rows,jsonl(group/"attempts.jsonl"),qid),"Incomplete group")
    return all_rows



class BudgetClient:
    def __init__(self, client, path):
        self.client, self.path = client, path
    def query_with_usage(self, prompt):
        ledger=read_json(self.path) if self.path.exists() else {'reserved_cny':0,'attempts':0}
        # Conservative UTF-8 byte allowance for input plus full output cap.
        reserve=((len(prompt.encode('utf-8'))+1024)*2.188+4096*8.75)/1_000_000
        if ledger['reserved_cny']+reserve>150:
            raise SystemExit('Budget guard stopped before next API request')
        ledger['reserved_cny']+=reserve;ledger['attempts']+=1
        atomic_json(self.path,ledger)
        return self.client.query_with_usage(prompt)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--questions',nargs='+',type=int,default=list(range(20)))
    p.add_argument('--methods',nargs='+',default=['memory','adaptive','adaptive_sde','memory_sde'])
    args=p.parse_args()
    require(os.environ.get('PYTHONHASHSEED')=='1','Set PYTHONHASHSEED=1 before Python')
    team=read_json(ROOT/'protocol.json')
    require(all(m in team['planned_methods'] for m in args.methods),'Unknown method')
    require(all(i in range(20) for i in args.questions),'Invalid question index')
    for name,h in team['source_sha256'].items():require(digest(ROOT/'source_snapshot'/'boolean'/name)==h,'Source changed')
    data=read_json(ROOT/'source_snapshot'/'boolean'/'datasets/boolean_extended_d2.json')
    require(digest(ROOT/'source_snapshot'/'boolean'/'datasets/boolean_extended_d2.json')==team['dataset_sha256'],'Dataset changed')
    mode='offline_fixture' if args.dry_run else 'live'
    out=ROOT/('offline_fixture_budget_v1' if args.dry_run else mode);out.mkdir(exist_ok=True)
    immutable(out/'manifest.json',dict(protocol=team,mode=mode,runner_sha256=digest(Path(__file__))))
    public=public_dataset(data,team['question_ids']);immutable(out/'public_tasks.json',public)
    if args.dry_run:client=OfflineFixture()
    else:
        key=os.environ.get('DASHSCOPE_API_KEY')
        if not key:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as reg:key=winreg.QueryValueEx(reg,'DASHSCOPE_API_KEY')[0]
        client=setup_llm('dashscope',api_key=key,**{k:team[k] for k in ('model','base_url','temperature','max_tokens','reasoning_effort','request_timeout')})
    if not args.dry_run:client=BudgetClient(client,out/'budget_ledger.json')
    with exclusive(out/'run.lock'):
        for i in args.questions:
            q=public['datasets_by_n_observations']['public_only'][i]
            for method in args.methods:
                bench=(MemorySDEBenchmark if method in ('memory','memory_sde') else SDEBenchmark)(str(out/'public_tasks.json'))
                rows=run_group(bench,out/method/f'q{i}',q,method,client,team,mode,digest(out/'manifest.json'))
                print('COMPLETE',i,method,'slots',len(rows),'truncated',sum(x['status']=='truncated' for x in rows),'attempts',sum(len(x['attempts']) for x in rows),flush=True)
if __name__=='__main__':main()
