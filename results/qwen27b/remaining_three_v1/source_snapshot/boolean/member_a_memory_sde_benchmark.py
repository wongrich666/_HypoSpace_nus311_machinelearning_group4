"""Thin prompt and logging adapter over the v3 generator and evaluator."""
import hashlib
from boolean_benchmark import BooleanBenchmarkRefined
from boolean_dataset import BooleanObservation
from member_a_memory_structure import MemoryStructureHistory as StructureHistory


class MemorySDEBenchmark(BooleanBenchmarkRefined):
    def create_prompt(self, observations, prior_hypotheses, target_complexity=None):
        base=super().create_prompt(observations,prior_hypotheses,target_complexity)
        if not self.enabled:
            self.decision={"suggested_root":None,"history":{},"scores":{}}
            return base
        self.decision=self.structures.choose(target_complexity)
        root=self.decision["suggested_root"]
        return base+f"\nPrefer a {root} root if it fits the observations. Another root is allowed."

    def evaluate_group(self, question, client, controller, enabled, rows, sink):
        self.enabled=enabled
        self.structures=StructureHistory()
        obs=[BooleanObservation(o["inputs"],o["output"]) for o in question["observations"]]
        prior=[]
        def decorate(row):
            tracking=self.structures.observe(row["target_operator_count"],self.decision["suggested_root"],row)
            return {**self.decision,**tracking}
        for row in rows:
            prompt=self.create_prompt(obs,prior,row["target_operator_count"])
            if prompt!=row["prompt"] or decorate(row)!=row["structure"]:
                raise ValueError("Structure/prompt replay mismatch before issuing more slots")
            if row["candidate"]:prior.append(row["candidate"])
        def record(row):
            row["prompt_sha256"]=hashlib.sha256(row["prompt"].encode()).hexdigest()
            row["structure"]=decorate(row)
            sink(row)
        return self.evaluate_single_observation_set(client,question,n_queries=16,max_retries=3,verbose=False,
            decoding="memory",controller=controller,prior_completions=rows,on_candidate_complete=record)

