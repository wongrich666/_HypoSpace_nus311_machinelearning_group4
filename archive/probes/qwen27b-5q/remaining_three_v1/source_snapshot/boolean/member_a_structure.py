"""SDE uses generated syntax for guidance; official identity is logged separately."""
import ast
import re
from collections import Counter, defaultdict
from copy import deepcopy

ROOTS = ("VAR", "NOT", "AND", "OR")


def syntax_root(formula):
    if not isinstance(formula, str) or not formula.strip():
        return None
    symbols = {"AND": "&", "OR": "|", "NOT": "~", "XOR": "^"}
    text = re.sub(r"\b(AND|OR|NOT|XOR)\b", lambda m: symbols[m[0]], formula.strip(), flags=0)
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if isinstance(node, ast.Name) and node.id in ("x", "y"):
        return "VAR"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "NOR":
        return "NOR"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return "NOT"
    if isinstance(node, ast.BinOp):
        return {ast.BitAnd: "AND", ast.BitOr: "OR", ast.BitXor: "XOR"}.get(type(node.op))
    return None


def features(row):
    syntax = syntax_root(row.get("candidate"))
    key = row["feedback"].get("mechanistic_key")
    if isinstance(key, str):
        key = ast.literal_eval(key)
    return {"syntax_root":syntax, "root_family":"NOT" if syntax=="NOR" else syntax,
            "mechanism_root":key[0] if key else None, "actual_k":row.get("actual_operator_count")}


class StructureHistory:
    """Question-local counts. No benchmark, model, question or answer-set argument."""
    def __init__(self):
        self.counts = defaultdict(Counter)

    def choose(self,k):
        roots = ("VAR",) if k==0 else ROOTS[1:]
        history = {root:dict(self.counts[k,root]) for root in roots}
        # Under-covered actual roots first, balanced by unsuccessful past guidance.
        scores = {root: self.counts[k,root]["generated"] + self.counts[k,root]["guided"]
                       - self.counts[k,root]["compliant_new"] for root in roots}
        root=min(roots,key=scores.get)
        return {"suggested_root":root,"scores":scores,"history":history}

    def observe(self, k, suggested, row):
        f=features(row)
        actual=f["root_family"]
        new=bool(row["feedback"]["is_new_valid"])
        if actual:
            cell=self.counts[f["actual_k"],actual]
            cell["generated"]+=1
            cell["new_valid"]+=int(new)
        complied=actual==suggested if actual and suggested else None
        if suggested:
            cell=self.counts[k,suggested]
            cell["guided"]+=1
            cell["root_compliant"]+=int(complied is True)
            cell["compliant_new"]+=int(complied is True and f["actual_k"]==k and new)
        return {**f,"root_compliance":complied}

