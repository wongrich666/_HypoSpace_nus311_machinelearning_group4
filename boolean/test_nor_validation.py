import unittest
from pathlib import Path
from boolean_benchmark import BooleanBenchmarkRefined
from boolean_dataset import BooleanDiscoveryGame, BooleanExpression, BooleanObservation

class NorValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=BooleanBenchmarkRefined(str(Path(__file__).parent/'datasets/boolean_extended_d2.json'))

    def test_all_34_enumerated_expressions_are_in_space(self):
        expressions=BooleanDiscoveryGame.generate_all_expressions(self.b.variables,self.b.operators,self.b.max_depth,self.b.mechanistic_opts)
        self.assertEqual(len(expressions),34)
        for expression in expressions:
            with self.subTest(formula=expression.formula): self.assertTrue(self.b._in_space(expression.formula))

    def test_all_ground_truths_match_public_observations(self):
        nor_count=0
        key=self.b.get_expression_mechanistic_key('NOT (x OR y)')
        for q in self.b.all_observation_sets:
            obs=[BooleanObservation(o['inputs'],o['output']) for o in q['observations']]
            for gt in q['ground_truth_expressions']:
                with self.subTest(question=q['observation_set_id'],formula=gt['formula']):
                    self.assertTrue(self.b.validate_expression(gt['formula'],obs)[0])
                nor_count+=self.b.get_expression_mechanistic_key(gt['formula'])==key
        self.assertEqual(nor_count,15)

    def test_public_space_counterexample_rejected(self):
        formula = 'NOT x OR y OR (x AND y)'
        self.assertEqual(len(self.b.public_space_keys), 34)
        self.assertFalse(self.b._in_space(formula))
        self.assertFalse(self.b.validate_expression(formula, [BooleanObservation({'x':0,'y':1},1)])[0])
        self.assertTrue(self.b._in_space('x'))
        self.assertTrue(self.b._in_space('y'))

    def test_key_unchanged_and_constraints_retained(self):
        self.assertEqual(self.b.get_expression_mechanistic_key('NOT (x OR y)'),('NOR',(('VAR','x'),('VAR','y'))))
        self.assertNotEqual(self.b.get_expression_mechanistic_key('NOT (x OR y)'),self.b.get_expression_mechanistic_key('NOT x AND NOT y'))
        self.assertFalse(self.b._in_space('NOT (x OR NOT y)'))
        self.assertFalse(self.b.validate_expression('NOT (x OR y)',[BooleanObservation({'x':0,'y':0},0)])[0])
        ast=BooleanExpression('NOT (x OR y)',self.b.variables,self.b.operators).sympy_expr
        self.assertTrue(self.b._check_operators_in_ast(ast,{'NOR'}))
        self.assertFalse(self.b._check_operators_in_ast(ast,{'NOT'}))
        self.assertFalse(self.b._check_operators_in_ast(ast,{'OR'}))

if __name__=='__main__': unittest.main()
