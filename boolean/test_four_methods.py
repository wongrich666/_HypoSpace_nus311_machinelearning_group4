import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_frozen_development as runner
from modules.exploration_controller import BooleanAdaptiveController
from modules.llm_interface import OpenRouterLLM


class Fake(OpenRouterLLM):
    def query_with_usage(self, prompt):
        self.calls = getattr(self, 'calls', 0) + 1
        # First slot exhausts its three attempts; second is truncated.
        if self.calls <= 3:
            raise TimeoutError('timed out')
        return {'response': 'Expression: NOT x',
                'finish_reason': 'length' if self.calls == 4 else 'stop',
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


class FourMethodsTest(unittest.TestCase):
    def test_baseline_actions_ignore_rewards_and_reset(self):
        for strategy in ('uniform', 'round_robin'):
            controllers = [BooleanAdaptiveController(3, seed=42, strategy=strategy) for _ in range(2)]
            sequences = []
            for c, reward in zip(controllers, (False, True)):
                actions = []
                for i in range(16):
                    action = c.select_action({'query_index': i})
                    actions.append(action)
                    c.update(action, {'is_new_valid': reward})
                sequences.append(actions)
                c.reset()
                self.assertEqual(c.select_action(), actions[0])
            self.assertEqual(*sequences)
            if strategy == 'round_robin':
                self.assertEqual(sequences[0], [0, 1, 2, 3] * 4)

    def test_failure_slots_and_attempt_cap_for_all_methods(self):
        benchmark = runner.BooleanBenchmarkRefined(str(runner.DATASET))
        q = benchmark.all_observation_sets[0]
        for method in runner.METHODS:
            model = Fake(api_key='fake')
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch('modules.generation_executor.sleep'):
                result = benchmark.run_benchmark(model, selected_questions=[q],
                    n_queries_per_sample=16, seed=42, decoding=method, max_retries=3,
                    checkpoint_dir=directory, verbose=False)
            rows = result['per_sample_results'][0]['request_log']
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows[0]['status'], 'api_error')
            self.assertEqual(len(rows[0]['attempts']), 3)
            self.assertEqual(rows[1]['status'], 'truncated')
            self.assertEqual(model.calls, 18)
            self.assertTrue(all(len(row['attempts']) <= 3 for row in rows))

    def test_launcher_all_five_questions_four_methods(self):
        class Success(Fake):
            def query_with_usage(self, prompt):
                return {'response': 'Expression: NOT x', 'finish_reason': 'stop',
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        with tempfile.TemporaryDirectory() as directory:
            args = ['runner', '--out', directory, '--n-queries', '16']
            with patch.object(sys, 'argv', args), patch.object(runner, 'api_key', return_value='fake'), patch.object(runner, 'probe'), patch.object(runner, 'source_version', return_value={'git_head_contains_these_edits': True, 'combined_worktree_sha256': 'fake'}), patch.object(runner, 'git_revision', return_value='fake'), patch.object(runner, 'save_snapshot'), patch.object(runner, 'setup_llm', side_effect=lambda *a, **k: Success(**k)), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(manifest['settings']['candidate_slots_total'], 320)
            self.assertEqual(manifest['settings']['max_attempts_per_slot'], 3)
            self.assertEqual(manifest['settings']['methods'], runner.METHODS)
            for i, qid in enumerate(manifest['question_ids']):
                for method in runner.METHODS:
                    result = json.loads((Path(directory) / f'q{i}_{method}.json').read_text())
                    sample = result['per_sample_results'][0]
                    rows = sample['request_log']
                    self.assertEqual(len(rows), 16)
                    self.assertTrue(all(row['observation_set_id'] == qid for row in rows))
                    if method != 'memory':
                        self.assertEqual(sample['decision_log'][0]['counts'], {'0': 0, '1': 0, '2': 0, '3': 0})
                        self.assertTrue(all(len(row['decision_log']) == 2 for row in rows))
                        self.assertTrue(all(row['target_operator_count'] in range(4) for row in rows))
                    if method == 'round_robin':
                        self.assertEqual([row['target_operator_count'] for row in rows], [0,1,2,3]*4)


if __name__ == '__main__':
    unittest.main()
