import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_causal_benchmark as runner

class UnifiedEntryTest(unittest.TestCase):
    def test_cli_all_methods_share_questions_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'dataset.json'
            config = root / 'config.yaml'
            config.write_text('llm: {type: dashscope, models: {dashscope: fake}, temperature: 0.7}', encoding='utf-8')
            questions = []
            for i in range(2):
                questions.append({'observation_set_id': str(i), 'n_observations': 1,
                    'n_compatible_graphs': 50 + i,
                    'observations': [{'string': 'Perturb A: B=1', 'perturbed_node': 'A', 'effects': {'A': 0, 'B': 1}}],
                    'ground_truth_graphs': [{'nodes': ['A', 'B'], 'edges': [['A', 'B']]}]})
            dataset.write_text(json.dumps({'metadata': {'nodes': ['A','B'], 'max_edges': 1}, 'datasets': questions}))
            class FakeLLM:
                def get_name(self): return 'fake'
                def query_with_usage(self, prompt):
                    return {'response': 'Graph: A->B', 'usage': {'total_tokens': 2}}
            common = ['--dataset', str(dataset), '--config', str(config), '--n-samples', '2', '--n-queries', '3',
                      '--seed', '7', '--experiment-manifest', str(root/'freeze.json'), '--checkpoint-dir', str(root/'checkpoints'), '--quiet']
            frozen_ids = None
            for method in ['memory', 'stratified', 'uniform', 'round_robin', 'adaptive']:
                output = root / (method + '.json')
                with patch.object(runner, 'setup_llm', return_value=FakeLLM()), contextlib.redirect_stdout(io.StringIO()):
                    runner.main(common + ['--decoding', method, '--output', str(output)])
                report = json.loads(output.read_text())
                rows = report['per_sample_results']
                ids = [r['observation_set_id'] for r in rows]
                if frozen_ids is None: frozen_ids = ids
                self.assertEqual(ids, frozen_ids)
                self.assertEqual([r['n_queries'] for r in rows], [3,3])
                logs = [json.loads(x) for x in Path(report['candidate_log_path']).read_text().splitlines()]
                self.assertEqual(len(logs), 6)
                if method in ['uniform','round_robin','adaptive']:
                    for row in rows:
                        self.assertEqual(len(row['decision_log']), 6)
                        self.assertTrue(all(x == 0 for x in row['decision_log'][0]['counts'].values()))
                        self.assertEqual(row['request_log'][0]['feedback']['reward'], 1)
                    self.assertEqual(len(logs[0]['decision_log']), 2)
                self.assertEqual(report['uses_ground_truth_distribution'], method == 'stratified')
            with patch.object(runner, 'setup_llm') as setup, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    runner.main(common + ['--n-queries', '4'])
                setup.assert_not_called()

    def test_cli_requires_explicit_budget(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['--dataset','unused','--config','unused','--seed','7','--experiment-manifest','unused'])

    def test_python_entry_rejects_oracle_budget(self):
        benchmark = runner.CausalBenchmarkEnhanced()
        with self.assertRaises(ValueError): benchmark.run_benchmark(None)

if __name__ == '__main__': unittest.main()
