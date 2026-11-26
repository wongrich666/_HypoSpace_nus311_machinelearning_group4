import unittest
from unittest.mock import patch
from boolean_benchmark import BooleanBenchmarkRefined

class FixedSlotsTest(unittest.TestCase):
    def test_sixteen_slots_failures_are_not_replaced(self):
        b=BooleanBenchmarkRefined.__new__(BooleanBenchmarkRefined)
        q={'n_observations':0, 'observations':[], 'ground_truth_expressions':[], 'observation_set_id':'test'}
        class Model:
            calls=0
            def query_with_usage(self,prompt):
                self.calls+=1
                if self.calls==1: raise TimeoutError('timed out')
                return {'response':'bad', 'usage':{'total_tokens':2}}
        m=Model()
        with patch.object(b,'create_prompt',return_value='test'), patch.object(b,'parse_llm_response',return_value=None), patch('modules.generation_executor.sleep'):
            r=b.evaluate_single_observation_set(m,q,n_queries=16,max_retries=3,verbose=False)
        self.assertEqual(m.calls,17)
        self.assertEqual(r['n_queries'],16)
        self.assertEqual(len(r['request_log']),16)
        self.assertEqual(r['api_attempt_count'],17)
        self.assertEqual(r['n_valid'],0)
        self.assertTrue(all(x['status']=='parse_error' for x in r['request_log']))

    def test_missing_budget_rejected_before_requests(self):
        b=BooleanBenchmarkRefined.__new__(BooleanBenchmarkRefined)
        with self.assertRaises(ValueError): b.run_benchmark(None)
        with self.assertRaises(ValueError): b.evaluate_single_observation_set(None,{},n_queries=0)

if __name__=='__main__': unittest.main()

