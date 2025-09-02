from typing import List, Any
import numpy as np
import z3
import sympy as sp
import subprocess
import sympy as sp
from sympy.printing.mathematica import mathematica_code
import tempfile, os
import itertools

def wolfram_tensor_equiv(t1, t2) -> bool:
    """
    Check element-wise equivalence of two SymPy tensors (Matrix / N-d array)
    by batching all equalities into a single WolframScript evaluation.

    Returns:
        True iff Simplify[And[e1==f1, e2==f2, ...]] is True.
    """
    # Basic shape checks
    if not hasattr(t1, "shape") or not hasattr(t2, "shape"):
        raise TypeError("Inputs must be SymPy Matrix/N-dim arrays with a .shape.")
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")

    # Build all element equalities and convert to Wolfram code
    eqs_wl = []
    idx_ranges = [range(s) for s in t1.shape]
    for idx in itertools.product(*idx_ranges):
        a = t1[idx] if len(idx) > 1 else t1[idx[0]]
        b = t2[idx] if len(idx) > 1 else t2[idx[0]]
        eq = sp.Eq(a, b)
        eqs_wl.append(mathematica_code(eq))

    # Empty tensors are trivially equivalent
    if not eqs_wl:
        return True

    # Use And@@(Simplify /@ {...}) so each equality is simplified individually,
    # then conjoined. Print[...] gives a clean True/False output.
    wl_code = "Print[And@@(Simplify /@ {" + ",".join(eqs_wl) + "})];\n"

    # Write to a temp .wl file (avoids command-length limits)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wl") as f:
        f.write(wl_code.encode("utf-8"))
        f.flush()
        path = f.name

    try:
        result = subprocess.run(
            ["wolframscript", "-file", path],
            capture_output=True, text=True, check=True
        )
        out = result.stdout.strip()
        if out == "True":
            return True
        if out == "False":
            return False
        raise RuntimeError(f"Unexpected Wolfram output: {out}\nCode was:\n{wl_code}")
    finally:
        os.remove(path)
        
        
def wolfram_equiv(expr1, expr2):
    code1 = mathematica_code(expr1)
    code2 = mathematica_code(expr2)

    wl_code = f"Print[Simplify[{code1} == {code2}]];"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wl") as f:
        f.write(wl_code.encode("utf-8"))
        f.flush()
        path = f.name

    try:
        result = subprocess.run(
            ["wolframscript", "-file", path],
            capture_output=True, text=True, check=True
        )
    finally:
        os.remove(path)

    return result.stdout.strip() == "True"
    

def z3_to_python_number(val: z3.ExprRef):
    if z3.is_int_value(val):
        return val.as_long()
    elif z3.is_rational_value(val):
        num = val.numerator_as_long()
        den = val.denominator_as_long()
        return num / den  # float
    else:
        raise TypeError(f"Unsupported Z3 value type: {val}")


def concrete_z3(zt, model: z3.ModelRef) :
    if type(zt) is z3.ArithRef:
        return model.evaluate(zt, model_completion=True)
    flat_result = [
        z3_to_python_number(model.evaluate(x, model_completion=True))
        for x in zt.ravel()
    ]
    result = np.array(flat_result).reshape(zt.shape)
    return result


def print_check(result, expect, dscp=""):
    icon = "⚠️ " if result == z3.unknown else ("✅" if result == expect else "🚨")
    print(f"   {icon} {dscp} | Result: {result}, Expect: {expect}")


def create_name_tensor(shape, prefix):
    """Recursively create a tensor with the given dynamic shape."""
    if len(shape) == 1:
        return np.array([f"{prefix}[{i}]" for i in range(shape[0])])
    return np.array(
        [
            create_name_tensor(shape[1:], prefix=f"{prefix}[{i}]")
            for i in range(shape[0])
        ]
    )


def create_tensor(shape, prefix: str, dtype: object):
    names = create_name_tensor(shape, prefix)
    return create_tensors_from_names(names, dtype)


def create_tensors_from_names(names, dtype):
    return np.vectorize(lambda x: dtype(x))(names)


def equalize_tensors(ts: List[np.ndarray]) -> List:
    unique([type(t) for t in ts])
    constraints = []
    t0 = ts[0]
    if type(t0) is not np.ndarray:
        constraints = [t0 == t for t in ts[1:]]
    else:
        shape = unique([t.shape for t in ts])
        for t in ts[1:]:
            constraints.extend(t0[i] == t[i] for i in np.ndindex(shape))
    return constraints


def unique(l: List[Any]):
    if len(l) == 1:
        return l[0]
    assert len(set(l)) == 1, f"{set(l)}"
    return l[0]


def z3_tactic_check_unsat(constraints) -> bool:
    g = z3.Goal()
    g.add(*constraints)
    simplify = z3.Tactic("solve-eqs")
    simplified_goal = simplify(g)
    all_unsat = all(str(subgoal) == "[False]" for subgoal in simplified_goal)
    return all_unsat


def sp_tensor_view(T):
    """Return (kind, flat_list, shape, rebuild) for Matrix/Array/np(object)/scalar."""
    if isinstance(T, sp.MatrixBase):
        shape = (T.rows, T.cols)
        flat = [T[i, j] for i in range(T.rows) for j in range(T.cols)]
        def rebuild(vals): return sp.Matrix(shape[0], shape[1], vals)
        return "matrix", flat, shape, rebuild
    if isinstance(T, sp.NDimArray):
        shape = T.shape
        flat = list(T)
        def rebuild(vals): return sp.Array(vals).reshape(shape)
        return "array", flat, shape, rebuild
    if isinstance(T, np.ndarray):
        shape = T.shape
        flat = list(T.flat)
        def rebuild(vals):
            out = np.empty(shape, dtype=object)
            out.flat[:] = vals
            return out
        return "numpy", flat, shape, rebuild
    # scalar
    def rebuild(vals): return vals[0]
    return "scalar", [T], (), rebuild

def sp_build_subs_from_pairs(pairs):
    """
    Pairs: list[(A, B)] with same shape; returns {sym_in_B: sym_in_A}.
    This encodes 'B ≡ A' as a substitution B→A.
    """
    subs = {}
    for A, B in pairs:
        _, flatA, _, _ = sp_tensor_view(A)
        _, flatB, _, _ = sp_tensor_view(B)
        assert len(flatA) == len(flatB)
        for a, b in zip(flatA, flatB):
            subs[b] = a
    return subs

def sp_apply_subs_tensor(T, subs):
    kind, flat, shape, rebuild = sp_tensor_view(T)
    flat2 = [ (e.xreplace(subs) if isinstance(e, sp.Basic) else e) for e in flat ]
    return rebuild(flat2)

def sp_tensors_equal_symbolic(A, B, use_wolfram=False):
    """Strict: elementwise sp.simplify(a-b) == 0."""
    if use_wolfram:
        return wolfram_tensor_equiv(A, B)
    _, flatA, _, _ = sp_tensor_view(A)
    _, flatB, _, _ = sp_tensor_view(B)
    if len(flatA) != len(flatB): return False
    for a, b in zip(flatA, flatB):
        if isinstance(a, sp.Basic) or isinstance(b, sp.Basic):
            if sp.simplify(a - b) != 0:
                return False
        else:
            if a != b:
                return False
    return True