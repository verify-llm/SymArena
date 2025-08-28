import z3
import sympy as sp
import numpy as np
from utils import create_tensor, equalize_tensors, print_check, z3_tactic_check_unsat
from operators.z3_attn import self_attention


def test_z3_sumsqare():
    print("🚀 z3 Sum Square")
    
    x1 = z3.Real("x1")
    x2 = z3.Real("x2")

    y1 = (x1 + x2) ** 2
    y2 = x1 ** 2 + x2 ** 2 + 2 * x1 * x2

    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")
    
    tactic_result = z3_tactic_check_unsat([y1_neq_y2])
    print_check(tactic_result, True, "tactic")
    

def test_sympy_sumsquare():
    print("🌿 SymPy Sum Square")

    x1, x2 = sp.symbols('x1 x2', real=True)

    y1 = (x1 + x2)**2
    y2 = x1**2 + x2**2 + 2*x1*x2

    diff = sp.simplify(y1 - y2)
    print_check(diff == 0, True, "simplify(y1 - y2) == 0")
    
    
def test_z3_squarediv():
    print("🚀 z3 Square Div")
    
    x1 = z3.Real("x1")
    x2 = z3.Real("x2")

    y1 = (x1 ** 2 - x2 ** 2)  / (x1 - x2)
    y2 = x1 + x2

    x1_lt_x2 = x1 > x2
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_lt_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_lt_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")
    
    tactic_result = z3_tactic_check_unsat([y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_sympy_squarediv():
    print("🌿 SymPy Square Div")

    x1, x2 = sp.symbols('x1 x2', real=True)

    y1 = (x1 ** 2 - x2 ** 2)  / (x1 - x2)
    y2 = x1 + x2

    diff = sp.simplify(y1 - y2)
    print_check(diff == 0, True, "simplify(y1 - y2) == 0")


def test_z3_selfattn_tp():
    print("🚀 z3 Self_attention with tensor partitioning")

    """
    # require: N = h * d
    @nnscaler.register_op('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
    y1 = self_attention(query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask)
    """
    h = 2
    L = E = N = d = 1
    query_shape = (L, N, E)
    qkv_proj_shape = (3 * h * d, E)
    qkv_bias_shape = (3 * h * d,)
    out_proj_shape = (E, h * d)
    half = 3 * h * d // 2

    query1 = create_tensor(query_shape, "query1", z3.Real)
    qkv_proj1 = create_tensor(qkv_proj_shape, "qkv_proj1", z3.Real)
    qkv_bias1 = create_tensor(qkv_bias_shape, "qkv_bias1", z3.Real)
    out_proj1 = create_tensor(out_proj_shape, "out_proj1", z3.Real)

    query2 = query1
    qkv_proj2a = qkv_proj1[:half, :]
    qkv_proj2b = qkv_proj1[half:, :]
    qkv_bias2a = qkv_bias1[:half]
    qkv_bias2b = qkv_bias1[half:]
    out_proj2a = out_proj1[:, : h * d // 2]
    out_proj2b = out_proj1[:, h * d // 2 :]

    scale = 1
    mask = True
    y1 = self_attention(query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask)
    y2a = self_attention(
        query2, qkv_proj2a, qkv_bias2a, out_proj2a, h // 2, scale, mask
    )
    y2b = self_attention(
        query2, qkv_proj2b, qkv_bias2b, out_proj2b, h // 2, scale, mask
    )
    y2 = y2a + y2b

    input_eq = []
    output_eq = equalize_tensors([y1, y2])
    output_neq = z3.Not(z3.And(output_eq))

    solver = z3.Solver()
    solver.add(input_eq)
    solver.add(output_eq)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(input_eq)
    solver.add(output_neq)
    result = solver.check()
    print_check(result, z3.unsat, "solver")
    
    tactic_result = z3_tactic_check_unsat(input_eq + [output_neq])
    print_check(tactic_result, True, "tactic")
    

if __name__ == "__main__":
    test_z3_sumsqare()
    test_sympy_sumsquare()
    test_z3_squarediv()
    test_sympy_squarediv()
    test_z3_selfattn_tp()
