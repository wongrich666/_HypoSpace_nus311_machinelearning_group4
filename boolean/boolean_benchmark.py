import sys
import json
import argparse
import copy
import hashlib
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation, BooleanDiscoveryGame
from modules.generation_executor import generate_boolean_one
from modules.exploration_controller import BooleanAdaptiveController

_BOOL_OPS = ("NOR", "XOR", "AND", "OR", "NOT")


def public_operator_bound(metadata: Dict) -> int:
    """Maximum operator count from public depth, not from the answer set.

    A full binary expression of depth d contains 2^d - 1 operators.
    """
    depth = metadata.get("max_depth") if isinstance(metadata, dict) else None
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("Dataset metadata needs a nonnegative public max_depth")
    bound = (1 << depth) - 1
    print(f"Using public operator bound from max_depth={depth}: max_operator_count={bound}")
    return bound


def boolean_operator_complexity(formula: str) -> int:
    """Count operator occurrences. A bare variable has complexity 0."""
    text = formula.upper()
    count = 0
    for op in _BOOL_OPS:
        found = re.findall(rf"\b{op}\b", text)
        count += len(found)
        if found:
            text = re.sub(rf"\b{op}\b", " ", text)
    return count


def build_cost_estimate(
    pricing_config: Optional[Dict],
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict:
    """Estimate only from a verified model price. An unknown price is null, not 0."""
    pricing = pricing_config or {}
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    verified = (
        bool(pricing.get("verified"))
        and input_price is not None
        and output_price is not None
    )
    estimated = None
    if verified:
        estimated = (
            prompt_tokens * float(input_price) + completion_tokens * float(output_price)
        ) / 1_000_000
    return {
        "estimated_cost": estimated,
        "currency": pricing.get("currency") or "CNY",
        "pricing_source": pricing.get("source") if verified else None,
        "pricing_verified": verified,
    }


def decoding_metadata(decoding: str) -> Dict:
    """Label how the query budget was chosen. The CLI value stays memory or stratified."""
    if decoding == "stratified":
        return {
            "decoding": "stratified",
            "method_label": "Oracle-informed complexity-stratified decoding",
            "allocation_source": "ground_truth_complexity_histogram",
            "uses_ground_truth_distribution": True,
        }
    if decoding == "memory":
        return {
            "decoding": "memory",
            "method_label": "Memory-based decoding",
            "allocation_source": "none",
            "uses_ground_truth_distribution": False,
        }
    if decoding in ("uniform", "round_robin"):
        return {
            "decoding": decoding,
            "method_label": f"Non-learning operator-count {decoding}",
            "allocation_source": "public_operator_counts",
            "uses_ground_truth_distribution": False,
        }
    if decoding == "adaptive":
        return {
            "decoding": "adaptive",
            "method_label": "Online operator-count adaptive decoding",
            "allocation_source": "validation_feedback",
            "uses_ground_truth_distribution": False,
        }
    return {
        "decoding": decoding,
        "method_label": None,
        "allocation_source": None,
        "uses_ground_truth_distribution": None,
    }


def allocate_complexity_budget(complexities: List[int], n_queries: int) -> List[Optional[int]]:
    """Split n_queries across complexity levels in proportion to their counts.

    Uses the largest-remainder method so the assigned counts sum to n_queries.
    Levels with count 0 are omitted. Returns the query schedule in increasing
    complexity. None marks an unconstrained (memory-based) query.
    """
    if n_queries <= 0:
        return []
    counts = Counter(complexities)
    if not counts:
        return [None] * n_queries
    levels = sorted(counts)
    total = sum(counts.values())
    raw = {c: n_queries * counts[c] / total for c in levels}
    floors = {c: int(raw[c]) for c in levels}
    remainder = n_queries - sum(floors.values())
    order = sorted(
        levels,
        key=lambda c: (raw[c] - floors[c], counts[c], -c),
        reverse=True,
    )
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    schedule: List[Optional[int]] = []
    for c in levels:
        schedule.extend([c] * floors[c])
    return schedule


class BooleanBenchmarkRefined:
    def __init__(self, complete_dataset_path: str):
        """
        Initialize benchmark with a complete dataset.
        
        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']
        self.max_operator_count = public_operator_bound(self.metadata)
        # Scope the exhaustive membership check to the audited extended_d2 grammar.
        # Enumerate from public metadata only, never per-question answer sets.
        self.public_space_keys = None
        if (set(self.variables) == {"x", "y"}
                and self.operators == {"AND", "OR", "NOT"} and self.max_depth == 2):
            self.public_space_keys = frozenset(
                expr.mechanistic_key(**self.mechanistic_opts)
                for expr in BooleanDiscoveryGame.generate_all_expressions(
                    self.variables, self.operators, self.max_depth, self.mechanistic_opts)
            )

        
        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)
        
        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(self, observations: List[BooleanObservation], 
                     prior_hypotheses: List[str],
                     target_complexity: Optional[int] = None) -> str:
        # Build obs_block, prior_block, operators_str ...
                # Format observations
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)
        
        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"
        
        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR'  in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")
        
        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        complexity_line = ""
        if target_complexity is not None:
            complexity_line = (
                f"7. Complexity constraint: the expression must contain EXACTLY {target_complexity} "
                "operator occurrences. Count each AND, OR, NOT, XOR, and NOR as one operator. "
                "A variable alone has complexity 0.\n            "
            )

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {obs_block}

            Prior expressions generated (avoid repeating any expression 
            that is equivalent under the rules below):
            {prior_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(self.variables)}
            2. Use ONLY these operators: {', '.join(self.operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {self.max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.
            {complexity_line}
            Output format: 
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators. 
            - Use lowercase for variables: x, y
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression
            
            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)
            
            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
        return prompt
    
    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None
        
        response = response.strip()
        
        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)
            
            # Remove markdown backticks
            expr = expr.strip('`')
            
            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')
            
            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)
            
            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')
            
            return expr.strip()
        
        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr
        
        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND','OR','NOT','XOR','NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned
        
        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None
    
    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            if expr.sympy_expr is None:
                return False
            
            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False
            
            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False
            
            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False
            
            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False
            if (self.public_space_keys is not None
                    and expr.mechanistic_key(**self.mechanistic_opts) not in self.public_space_keys):
                return False

            
            return True
        except:
            return False
    
    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0
        
        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0
        
        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)
        
        return 0
    
    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # A NOR key does not require a primitive NOR in the grammar.
            # Otherwise validate NOT and its OR subtree as ordinary operators.
            if (isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or)
                    and 'NOR' in allowed_ops):
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True
    
    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Optional[Dict]]:
        """Validate if expression is consistent with observations and meets constraints."""
        if not self._in_space(expr_str):
            return False, None
        try:
            # Now create expression and validate
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            # Check if expression uses only allowed variables
            if expr.sympy_expr is not None:
                allowed_symbols = {sp.symbols(v) for v in self.variables}
                free_symbols = expr.sympy_expr.free_symbols
                if not free_symbols.issubset(allowed_symbols):
                    return False, None
                
                # Check operators using AST traversal
                if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                    return False, None
                
                # Disallow constants (True/False) to maintain consistent space
                for node in sp.preorder_traversal(expr.sympy_expr):
                    if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                        return False, None
            
            # Check expression depth using AST
            if expr.sympy_expr is not None:
                ast_depth = self._get_ast_depth(expr.sympy_expr)
                if ast_depth > self.max_depth:
                    return False, None
            
            # Check consistency with observations
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    return False, None
            
            return True, expr.truth_table
        except:
            return False, None
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def get_expression_depth(self, expr_str: str) -> int:
        """Calculate the depth of a Boolean expression."""
        # Simple heuristic: count maximum nesting level of parentheses
        max_depth = 0
        current_depth = 0
        
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Also check for operators without parentheses
        if max_depth == 0:
            # Count operators to estimate depth
            expr_upper = expr_str.upper()
            if any(op in expr_upper for op in ['AND','OR','NOT','XOR','NOR']):
                max_depth = 1
        
        return max_depth
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5,
        decoding: str = "memory",
        controller=None,
        on_candidate_complete=None,
        prior_completions=None,
    ) -> Dict:
        """
        Evaluate LLM on a single observation set.
        
        Returns:
            Dictionary with evaluation results for this observation set
        """
        if isinstance(n_queries, bool) or not isinstance(n_queries, int) or n_queries <= 0:
            raise ValueError("A positive fixed candidate budget is required")
        if controller is not None and decoding != "memory":
            raise ValueError("controller cannot be combined with oracle stratified decoding")
        # Extract observations and ground truths
        observations = [
            BooleanObservation(obs['inputs'], obs['output']) 
            for obs in observation_set['observations']
        ]
        
        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            # Extract mechanistic key for structural comparison
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                # Convert string representation back to tuple if needed (safe parsing)
                gt_mechanistic_keys.append(ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)
        
        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_mechanistic_keys = set()
        unique_valid_expressions = []
        all_unique_mechanistic_keys = set()
        unique_all_expressions = []
        parse_success_count = 0
        in_space_count = 0
        
        # Track token usage and costs
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Track errors
        errors = []  # List of error details
        error_counts = {}  # Count of each error type

        if decoding == "stratified":
            complexities = [
                boolean_operator_complexity(gt.get("formula", ""))
                for gt in ground_truth_expressions
            ]
            schedule = allocate_complexity_budget(complexities, n_queries)
        else:
            schedule = [None] * n_queries
        
        request_log = []
        seen_valid_keys = set()
        prior_completions = list(prior_completions or [])
        if any(row.get("query_index") != index for index, row in enumerate(prior_completions)):
            raise ValueError("prior completions must be a contiguous 0-based prefix")
        if len(prior_completions) > n_queries:
            raise ValueError("prior completions exceed the fixed candidate budget")

        def absorb(index, hypothesis_str, feedback, usage, attempts, record):
            nonlocal parse_success_count, in_space_count
            nonlocal total_prompt_tokens, total_completion_tokens, total_tokens, total_cost
            usage = usage or {}
            total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            total_completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            total_cost += sum(float(attempt.get("cost") or 0) for attempt in attempts or [])
            request_log.append(record)
            for attempt in attempts or []:
                if attempt.get("error"):
                    error_type = self._classify_error(attempt["error"])
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    errors.append({
                        "query_index": index, "attempt": attempt.get("attempt"),
                        "error_message": attempt["error"], "error_type": error_type,
                    })
            if hypothesis_str:
                parse_success_count += 1
                all_hypotheses.append(hypothesis_str)
                if self._in_space(hypothesis_str):
                    in_space_count += 1
                    all_mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if all_mech_key and all_mech_key not in all_unique_mechanistic_keys:
                        all_unique_mechanistic_keys.add(all_mech_key)
                        unique_all_expressions.append(hypothesis_str)
                if feedback.get("is_valid"):
                    valid_hypotheses.append(hypothesis_str)
                    mech_key = feedback.get("mechanistic_key")
                    if mech_key and mech_key not in unique_mechanistic_keys:
                        unique_mechanistic_keys.add(mech_key)
                        unique_valid_expressions.append(hypothesis_str)
            if feedback.get("is_valid") and feedback.get("mechanistic_key"):
                seen_valid_keys.add(feedback["mechanistic_key"])

        for row in prior_completions:
            logged = copy.deepcopy(row.get("feedback") or {})
            hypothesis_str = row.get("candidate")
            if controller is not None:
                state = {
                    "query_index": row["query_index"], "n_queries": n_queries,
                    "observations": [obs.to_string() for obs in observations],
                    "variables": list(self.variables),
                    "max_operator_count": self.max_operator_count,
                    "n_unique_valid": len(seen_valid_keys),
                }
                replay_action = controller.select_action(state)
                if replay_action != row.get("target_operator_count"):
                    raise ValueError("replayed controller action does not match the candidate log")
                controller.update(replay_action, logged)
            schedule[row["query_index"]] = row.get("target_operator_count")
            absorb(row["query_index"], hypothesis_str, logged, row.get("usage"), row.get("attempts"), copy.deepcopy(row))

        for i in range(len(prior_completions), n_queries):
            target_complexity = schedule[i]
            decision_start = len(getattr(controller, "decision_log", []))
            if controller is not None:
                state = {
                    "query_index": i, "n_queries": n_queries,
                    "observations": [obs.to_string() for obs in observations],
                    "variables": list(self.variables),
                    "max_operator_count": self.max_operator_count,
                    "n_unique_valid": len(seen_valid_keys),
                }
                target_complexity = controller.select_action(state)
                if (
                    isinstance(target_complexity, bool) or not isinstance(target_complexity, int)
                    or target_complexity < 0 or target_complexity > self.max_operator_count
                ):
                    raise ValueError("controller action must be an allowed operator count")
                schedule[i] = target_complexity
            generation = generate_boolean_one(
                self, llm, observations, all_hypotheses, target_complexity,
                max_attempts=max_retries,
            )
            hypothesis_str = generation.candidate
            mech_key = self.get_expression_mechanistic_key(hypothesis_str) if hypothesis_str else None
            is_valid = False
            failure_reason = None
            failure_detail = None
            if generation.status != "ok" or not hypothesis_str:
                failure_reason = "truncated_output" if generation.status == "truncated" else (
                    generation.status if generation.status != "ok" else "parse_error"
                )
                if generation.attempts:
                    last = generation.attempts[-1]
                    failure_detail = last.get("parse_error") or last.get("error")
            elif not self._in_space(hypothesis_str):
                failure_reason = "out_of_space"
            else:
                is_valid, _truth_table = self.validate_expression(hypothesis_str, observations)
                if not is_valid:
                    failure_reason = "observation_mismatch"
            logged_key = None if mech_key is None else repr(mech_key)
            is_new = bool(is_valid and logged_key and logged_key not in seen_valid_keys)
            feedback = {
                "status": generation.status,
                "is_valid": is_valid,
                "failure_reason": failure_reason,
                "failure_detail": failure_detail,
                "mechanistic_key": logged_key,
                "is_new_valid": is_new,
                "is_duplicate_valid": bool(is_valid and not is_new),
                "reward": int(is_new),
                "target_operator_count": target_complexity,
                "actual_operator_count": generation.actual_operator_count,
                "obeyed_operator_count": generation.obeyed_operator_count,
                "usage": dict(generation.usage),
                "api_attempt_count": len(generation.attempts),
            }
            if controller is not None:
                controller.update(target_complexity, copy.deepcopy(feedback))
            record = {
                "query_index": i,
                "observation_set_id": observation_set["observation_set_id"],
                "status": generation.status,
                "candidate": hypothesis_str,
                "prompt": generation.prompt,
                "response": generation.response,
                "reasoning_content": generation.reasoning_content,
                "target_operator_count": target_complexity,
                "actual_operator_count": generation.actual_operator_count,
                "obeyed_operator_count": generation.obeyed_operator_count,
                "usage": generation.usage,
                "attempts": generation.attempts,
                "feedback": feedback,
                "decision_log": copy.deepcopy(getattr(controller, "decision_log", [])[decision_start:]),
            }
            absorb(i, hypothesis_str, feedback, generation.usage, generation.attempts, record)
            if on_candidate_complete is not None:
                on_candidate_complete(copy.deepcopy(record))
        
        # Calculate metrics
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        in_space_rate = in_space_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_expressions) / n_queries if n_queries > 0 else 0
        recovery_rate = 0
        
        # Check recovery against ground truths using mechanistic keys (structural matching)
        recovered_gts = set()
        recovered_gt_keys = set()
        
        # Collect mechanistic keys of generated expressions
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)
        
        # Check which ground truths were recovered (structurally)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)
                recovered_gt_keys.add(gt_key)
        
        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0
        
        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),  # Unique among valid hypotheses
            'n_unique_all': len(unique_all_expressions),  # Unique among in-space hypotheses
            'n_recovered_gts': len(recovered_gts),
            'request_log': request_log,
            'api_attempt_count': sum(len(r['attempts']) for r in request_log),
            'parse_success_rate': parse_success_rate,  # Diagnostic: how many parsed
            'in_space_rate': in_space_rate,  # Diagnostic: how many met space constraints
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,  # Now using unique in-space / n_queries
            'recovery_rate': recovery_rate,
            'decoding': decoding,
            'complexity_schedule': [c for c in schedule if c is not None] if decoding == "stratified" else None,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            # Token usage and cost tracking
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            # Error tracking
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            }
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None,
        decoding: str = "memory",
        pricing_config: Optional[Dict] = None,
        selected_questions=None,
        on_candidate_complete=None,
        prior_completions=None,
        max_retries: int = 3,
        adaptive_parameters: Optional[Dict] = None,
    ) -> Dict:
        """
        Run the benchmark with sampling.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with complete benchmark results
        """
        if isinstance(n_queries_per_sample, bool) or not isinstance(n_queries_per_sample, int) or n_queries_per_sample <= 0:
            raise ValueError("Explicit positive n_queries_per_sample required; answer-count budgets disabled")
        if decoding not in ("memory", "stratified", "uniform", "round_robin", "adaptive"):
            raise ValueError("Unknown decoding method")
        if decoding in ("uniform", "round_robin", "adaptive") and seed is None:
            raise ValueError("Adaptive runs require a fixed seed")
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and safe filename
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"
        
        sampled_sets = list(selected_questions) if selected_questions is not None else self.sample_observation_sets(n_samples, seed)
        n_samples = len(sampled_sets)
        print(f"\nRunning refined Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Decoding: {decoding}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        if prior_completions and n_samples != 1:
            raise ValueError("prior completions can resume only one question")
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        
        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Initialize error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])

                # RESTORE TOKEN AND COST TOTALS
                if 'total_token_usage' in checkpoint_data:
                    total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                    total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                if 'total_cost' in checkpoint_data:
                    total_cost = checkpoint_data['total_cost']
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")
            
            try:
                # Determine number of queries for this observation set
                n_queries = n_queries_per_sample
                controller = None
                controller_seed = None
                if decoding in ("uniform", "round_robin", "adaptive"):
                    params = adaptive_parameters or {}
                    question_id = obs_set["observation_set_id"]
                    controller_seed = int(hashlib.sha256(
                        json.dumps([seed, question_id]).encode()
                    ).hexdigest()[:16], 16)
                    controller = BooleanAdaptiveController(
                        self.max_operator_count,
                        seed=controller_seed,
                        strategy=decoding,
                        forget_factor=params.get("forget_factor", 0.9),
                        temperature=params.get("controller_temperature", 1.0),
                        explore_bonus=params.get("explore_bonus", 0.15),
                    )
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False, max_retries=max_retries,
                    decoding="memory" if controller is not None else decoding,
                    controller=controller, on_candidate_complete=on_candidate_complete,
                    prior_completions=prior_completions,
                )
                result["decoding"] = decoding
                result["controller_seed"] = controller_seed
                result["decision_log"] = copy.deepcopy(getattr(controller, "decision_log", []))
                result["max_operator_count"] = self.max_operator_count
                if controller is not None:
                    result["method_parameters"] = {
                        "strategy": controller.strategy,
                        "actions": list(range(self.max_operator_count + 1)),
                        "forget_factor": controller.forget_factor,
                        "controller_temperature": controller.temperature,
                        "explore_bonus": controller.explore_bonus,
                    }
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                
                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                
                # Save checkpoint after each sample
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'decoding': decoding,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_token_usage': {
                        'prompt_tokens': total_prompt_tokens,
                        'completion_tokens': total_completion_tokens,
                        'total_tokens': total_tokens
                    },
                    'total_cost': total_cost,
                    'total_errors': len(all_errors),
                    'error_types': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            except RuntimeError:
                raise
            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            **decoding_metadata(decoding),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost_estimate': build_cost_estimate(
                pricing_config, total_prompt_tokens, total_completion_tokens
            ),
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        print(f"Method: {final_results['method_label']}")
        print(f"Uses ground-truth distribution: {final_results['uses_ground_truth_distribution']}")
        
        for metric_name, metric_key in [('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nLedger cost (default USD rate in the client, not a vendor bill):")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        estimate = final_results['cost_estimate']
        if estimate['estimated_cost'] is None:
            print("  Estimated cost: unknown")
        else:
            print(
                f"  Estimated cost: {estimate['estimated_cost']:.4f} {estimate['currency']}"
                f" ({estimate['pricing_source']})"
            )
        
        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")
        
        print("=" * 50)
        
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""
    
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "dashscope":
        api_key = kwargs.get('api_key') or os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("DashScope API key required (set DASHSCOPE_API_KEY)")

        return OpenRouterLLM(
            model=kwargs.get('model', 'qwen-plus'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 2048),
            base_url=kwargs.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            request_timeout=kwargs.get('request_timeout', 60.0),
            reasoning_effort=kwargs.get('reasoning_effort'),
        )
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run refined Boolean discovery benchmark with sampling")
    parser.add_argument("--dataset", required=True, help="Path to complete Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, required=True, help="Fixed candidate slots per question; development protocol uses 16")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--decoding", choices=["memory", "stratified", "uniform", "round_robin", "adaptive"], default="memory",
                        help="memory, oracle stratified, or online adaptive operator-count control")
    
    args = parser.parse_args()
    if args.n_queries <= 0:
        parser.error("--n-queries must be positive")
    
    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    
    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229',
            'dashscope': 'qwen-plus',
        }
        model = default_models.get(llm_type)
    
    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    if not api_key:
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'dashscope': 'DASHSCOPE_API_KEY',
        }
        if llm_type in env_vars:
            api_key = os.environ.get(env_vars[llm_type])
    
    temperature = config.get('llm', {}).get('temperature', 0.7)
    base_url = config.get('llm', {}).get('base_url')
    max_tokens = config.get('llm', {}).get('max_tokens', 2048)
    checkpoint_dir = config.get('benchmark', {}).get('checkpoint', 'checkpoints')
    run_id = config.get('benchmark', {}).get('run_id', None)
    verbose = config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    dataset_name = Path(args.dataset).stem
    model_name = Path(model).stem if model else llm_type
    output_pattern = config.get('benchmark', {}).get(
        'output_pattern', 'results/{decoding}/{dataset_name}_{model}.json'
    )
    if "{decoding}" not in output_pattern:
        output_pattern = output_pattern.replace("results/", "results/{decoding}/", 1)
    output = output_pattern.format(
        dataset_name=dataset_name,
        model=model_name,
        llm_type=llm_type,
        decoding=args.decoding,
    )

    # Initialize benchmark
    benchmark = BooleanBenchmarkRefined(args.dataset)
     
    # Print configuration
    print("\n" + "=" * 60)
    print("REFINED BOOLEAN BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Queries per sample: {args.n_queries}")
    print(f"Seed: {args.seed}")
    print(f"Decoding: {args.decoding}")
    print(f"Output: {output}")
    print("=" * 60)
    
    # Set up LLM
    llm = setup_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens,
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        decoding=args.decoding,
        pricing_config=config.get("pricing"),
    )
    
    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()