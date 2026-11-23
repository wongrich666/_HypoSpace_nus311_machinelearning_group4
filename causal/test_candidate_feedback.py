import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_causal_benchmark import CausalBenchmarkEnhanced
from modules.candidate_feedback import CandidateJSONLLogger


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.benchmark = CausalBenchmarkEnhanced()
        self.benchmark.nodes = ['A', 'B']
        self.benchmark.max_edges = 1
        self.question = {
            'observation_set_id': 'q1',
            'observations': [{'string': 'Perturb A: B=1', 'perturbed_node': 'A',
                              'effects': {'A': 0, 'B': 1}}],
            'ground_truth_graphs': [{'nodes': ['A', 'B'], 'edges': [['A', 'B']]}],
        }

    def model(self, responses):
        class FakeModel:
            def query_with_usage(self, prompt):
                response = next(responses)
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, dict):
                    return response
                return {'response': response, 'usage': {'total_tokens': 3}}
        return FakeModel()

    def test_controller_loop_retry_invalid_duplicate_and_jsonl(self):
        events = []
        class Controller:
            def select_action(controller, state):
                self.assertNotIn('ground_truth_graphs', state)
                events.append('select')
                return 0

            def update(controller, action, feedback):
                events.append('update')
                self.assertEqual(action, 0)
                self.assertNotIn('recovery_rate', feedback)
                feedback['reward'] = 999  # Must not mutate persisted feedback.

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidates.jsonl'
            logger = CandidateJSONLLogger(path)
            def completed(record):
                events.append('log')
                logger(record)
            with patch('modules.generation_executor.sleep'):
                result = self.benchmark.evaluate_single_observation_set(
                    self.model(iter([TimeoutError('timed out'), 'bad format',
                                     'Graph: No edges', 'Graph: A->B', 'Graph: A->B'])),
                    self.question, n_queries=4, max_retries=2, verbose=False,
                    controller=Controller(), on_candidate_complete=completed,
                )
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(events, ['select', 'update', 'log'] * 4)
        self.assertEqual(result['api_attempt_count'], 5)
        self.assertEqual(result['n_queries'], 4)
        self.assertEqual(result['n_valid'], 2)
        self.assertEqual(result['n_unique_valid'], 1)
        self.assertEqual(result['recovery_rate'], 1)
        self.assertEqual([r['feedback']['reward'] for r in rows], [0, 0, 1, 0])
        self.assertEqual(rows[1]['feedback']['failure_reason'], 'observation_mismatch')
        self.assertFalse(rows[2]['feedback']['obeyed_edge_count'])
        self.assertTrue(rows[2]['feedback']['is_valid'])
        self.assertEqual(rows[2]['feedback']['canonical_hash'], rows[3]['feedback']['canonical_hash'])
        self.assertEqual(rows[0]['api_attempt_count'], 2)
        self.assertEqual(rows[0]['usage']['total_tokens'], 3)
        self.assertIsNone(rows[0]['candidate'])

    def test_parser_exception_in_completion_log(self):
        completed = []
        with patch.object(self.benchmark, 'parse_llm_response', side_effect=ValueError('broken parse')):
            result = self.benchmark.evaluate_single_observation_set(
                self.model(iter(['Graph: A->B'])), self.question, n_queries=1,
                on_candidate_complete=completed.append, verbose=False,
            )
        self.assertEqual(result['n_valid'], 0)
        self.assertIn('broken parse', completed[0]['feedback']['failure_detail'])
        self.assertIn('parse_error', completed[0]['attempts'][0])

    def test_truncation_and_exhausted_api_error_consume_slots(self):
        result = self.benchmark.evaluate_single_observation_set(
            self.model(iter([{'response': 'Graph: A->B', 'finish_reason': 'length'},
                             RuntimeError('permanent failure')])),
            self.question, n_queries=2, max_retries=1, verbose=False,
        )
        self.assertEqual([r['feedback']['failure_reason'] for r in result['request_log']],
                         ['truncated_output', 'api_error'])
        self.assertEqual(result['api_attempt_count'], 2)
        self.assertEqual(result['n_valid'], 0)

    def test_feedback_independent_of_ground_truth_and_resets_per_question(self):
        rewards = []
        for truth in [self.question['ground_truth_graphs'], []]:
            question = dict(self.question, ground_truth_graphs=truth)
            result = self.benchmark.evaluate_single_observation_set(
                self.model(iter(['Graph: A->B'])), question, n_queries=1, verbose=False,
            )
            rewards.append(result['request_log'][0]['feedback'])
        self.assertEqual(rewards[0], rewards[1])
        self.assertEqual(rewards[1]['reward'], 1)

    def test_reject_oracle_controller_mix(self):
        with self.assertRaises(ValueError):
            self.benchmark.evaluate_single_observation_set(
                None, self.question, decoding='stratified', controller=object(),
            )


if __name__ == '__main__':
    unittest.main()
