import z3
import sympy as sp
import numpy as np
from utils import (
    create_tensor,
    equalize_tensors,
    print_check,
    z3_tactic_check_unsat,
    concrete_z3,
    sp_tensor_view,
    sp_build_subs_from_pairs,
    sp_apply_subs_tensor,
    sp_tensors_equal_symbolic,
)
from operators.z3_attn import (
    self_attention,
    bw_softmax,
    softmax,
    bw_self_attention,
    standard_attention,
    flash_attention,
    arrmax,
    flash_attention_sympy,
)



def test_z3_selfattn_bw(zoom=1):
    print(f"🚀 z3 Self_attention Backward Zoom={zoom}")

    """
    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # require: N = h * d
    """

    L = E = N = h = d = 2 * zoom
    query_shape = (L, N, E)
    qkv_proj_shape = (3 * h * d, E)
    qkv_bias_shape = (3 * h * d,)
    out_proj_shape = (E, h * d)

    query1 = create_tensor(query_shape, "query1", z3.Real)
    qkv_proj1 = create_tensor(qkv_proj_shape, "qkv_proj1", z3.Real)
    qkv_bias1 = create_tensor(qkv_bias_shape, "qkv_bias1", z3.Real)
    out_proj1 = create_tensor(out_proj_shape, "out_proj1", z3.Real)
    g1 = create_tensor(query_shape, "g1", z3.Real)

    query2 = create_tensor(query_shape, "query2", z3.Real)
    qkv_proj2 = create_tensor(qkv_proj_shape, "qkv_proj2", z3.Real)
    qkv_bias2 = create_tensor(qkv_bias_shape, "qkv_bias2", z3.Real)
    out_proj2 = create_tensor(out_proj_shape, "out_proj2", z3.Real)
    g2 = create_tensor(query_shape, "g2", z3.Real)

    scale = 1
    mask = False
    # g_query, g_qkv_proj, g_qkv_bias, g_out_proj
    y11, y12, y13, y14 = bw_self_attention(
        g1, query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask
    )
    y21, y22, y23, y24 = bw_self_attention(
        g2, query2, qkv_proj2, qkv_bias2, out_proj2, h, scale, mask
    )

    input_eq = (
        equalize_tensors([g1, g2])
        + equalize_tensors([query1, query2])
        + equalize_tensors([qkv_proj1, qkv_proj2])
        + equalize_tensors([qkv_bias1, qkv_bias2])
        + equalize_tensors([out_proj1, out_proj2])
    )

    output_eq = (
        equalize_tensors([y11, y21])
        + equalize_tensors([y12, y22])
        + equalize_tensors([y13, y23])
        + equalize_tensors([y14, y24])
    )
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


def test_z3_flashattn1(zoom=1):
    print(f"🚀 z3 Flash Attention 1 Zoom={zoom}")
    N = 1 * zoom
    d = 1 * zoom
    M = 4 * zoom
    
    # N = 2 * zoom
    # d = 2 * zoom
    # M = 16 * zoom

    Q = create_tensor((N, d), "Q", z3.Real)
    K = create_tensor((N, d), "K", z3.Real)
    V = create_tensor((N, d), "V", z3.Real)

    y1 = standard_attention(Q, K, V, N, d)
    y2 = flash_attention(Q, K, V, N, d, M)

    input_eq = []
    output_eq = equalize_tensors([y1, y2])
    output_neq = z3.Not(z3.And(output_eq))

    solver = z3.Solver()
    solver.check()
    solver.add(input_eq)
    solver.add(output_eq)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.check()
    solver.add(input_eq)
    solver.add(output_neq)
    result = solver.check()
    print_check(result, z3.unsat, "solver")
    if result == z3.sat:
        model = solver.model()
        print("Q", concrete_z3(Q, model))
        print("K", concrete_z3(K, model))
        print("V", concrete_z3(V, model))
        print("y1", concrete_z3(y1, model))
        print("y2", concrete_z3(y2, model))


def test_sympy_max():
    print("🌿 SymPy Max over Array")
    x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6") 
    expr = sp.Max(sp.Max(x1, x2, x3), sp.Max(x4, x5), x6) - sp.Max(x1,x2,x3,x4,x5,x6)
    is_zero = sp.simplify(expr) == 0
    print_check(is_zero, True, "Max")


def test_z3_max():
    print("🚀 z3 Max over Array")
    x1, x2, x3, x4, x5, x6 = z3.Reals("x1 x2 x3 x4 x5 x6")
    
    expr1 = arrmax([arrmax([x1, x2, x3]), arrmax([x4, x5]), x6])
    expr2 = arrmax([x1,x2,x3,x4,x5,x6])
    
    output_eq = expr1 == expr2
    output_neq = expr1 != expr2
    
    solver = z3.Solver()
    solver.add(output_eq)
    result = solver.check()
    print_check(result, z3.sat, "Max")
    
    solver = z3.Solver()
    solver.add(output_neq)
    result = solver.check()
    print_check(result, z3.unsat, "Max")
        

def np_sympy_equal(A: np.ndarray, B: np.ndarray) -> bool:
    if A.shape != B.shape:
        return False
    return all(sp.simplify(a - b) == 0 for a, b in zip(A.flat, B.flat))


def test_sympy_flashattn1(zoom=1):
    print(f"🌿 Sympy Flash Attention 1 Zoom={zoom}")
    N = 2 * zoom
    d = 2 * zoom
    M = 16 * zoom
    
    # N = 1 * zoom
    # d = 1 * zoom
    # M = 4 * zoom

    Q = create_tensor((N, d), "Q", sp.Symbol)
    K = create_tensor((N, d), "K", sp.Symbol)
    V = create_tensor((N, d), "V", sp.Symbol)

    y1 = standard_attention(Q, K, V, N, d)
    y2 = flash_attention_sympy(Q, K, V, N, d, M)

    is_zero = np_sympy_equal(y1, y2)
    print_check(is_zero, True, "flashattn")
    
    if not is_zero:
        print("⚠️ Symbolic check inconclusive → testing with concrete values...")

        # collect all symbols from Q,K,V
        symbols = list(Q.flat) + list(K.flat) + list(V.flat)

        # assign sequential integers (1,2,3,...)
        vals = {s: i+1 for i, s in enumerate(symbols)}

        y1_val = sp.Matrix(y1.tolist()).evalf(subs=vals)
        y2_val = sp.Matrix(y2.tolist()).evalf(subs=vals)

        print("Substitution:", vals)
        print("y1 =", y1_val)
        print("y2 =", y2_val)

        # check numeric equality
        is_zero = all(abs(a - b) < 1e-9 for a, b in zip(y1_val, y2_val))
        


def test_sympy_selfattn_bw(zoom=1):
    print(f"🌿 z3 Self_attention Backward Zoom={zoom}")

    """
    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # require: N = h * d
    """

    L = E = N = h = d = 2 * zoom
    query_shape = (L, N, E)
    qkv_proj_shape = (3 * h * d, E)
    qkv_bias_shape = (3 * h * d,)
    out_proj_shape = (E, h * d)

    query1 = create_tensor(query_shape, "query1", sp.Symbol)
    qkv_proj1 = create_tensor(qkv_proj_shape, "qkv_proj1", sp.Symbol)
    qkv_bias1 = create_tensor(qkv_bias_shape, "qkv_bias1", sp.Symbol)
    out_proj1 = create_tensor(out_proj_shape, "out_proj1", sp.Symbol)
    g1 = create_tensor(query_shape, "g1", sp.Symbol)

    query2 = create_tensor(query_shape, "query2", sp.Symbol)
    qkv_proj2 = create_tensor(qkv_proj_shape, "qkv_proj2", sp.Symbol)
    qkv_bias2 = create_tensor(qkv_bias_shape, "qkv_bias2", sp.Symbol)
    out_proj2 = create_tensor(out_proj_shape, "out_proj2", sp.Symbol)
    g2 = create_tensor(query_shape, "g2", sp.Symbol)

    scale = 1
    mask = False
    # g_query, g_qkv_proj, g_qkv_bias, g_out_proj
    y11, y12, y13, y14 = bw_self_attention(
        g1, query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask
    )
    y21, y22, y23, y24 = bw_self_attention(
        g2, query2, qkv_proj2, qkv_bias2, out_proj2, h, scale, mask
    )

    # --- SymPy-equivalent of 'input_eq' via substitution map (second → first)
    pairs = [
        (g1,        g2),
        (query1,    query2),
        (qkv_proj1, qkv_proj2),
        (qkv_bias1, qkv_bias2),
        (out_proj1, out_proj2),
    ]
    subs = sp_build_subs_from_pairs(pairs)

    # Apply constraints by substitution, then compare outputs
    y21_s = sp_apply_subs_tensor(y21, subs)
    y22_s = sp_apply_subs_tensor(y22, subs)
    y23_s = sp_apply_subs_tensor(y23, subs)
    y24_s = sp_apply_subs_tensor(y24, subs)

    ok1 = sp_tensors_equal_symbolic(y11, y21_s)
    ok2 = sp_tensors_equal_symbolic(y12, y22_s)
    ok3 = sp_tensors_equal_symbolic(y13, y23_s)
    ok4 = sp_tensors_equal_symbolic(y14, y24_s)

    all_ok = ok1 and ok2 and ok3 and ok4
    print_check(all_ok, True, "bw_self_attention equivalence (SymPy)")

    
if __name__ == "__main__":
    # op with nonlinear but not Max
    print("Use Z3 to check backward_self_attention, which contains nonlinear ops.\n"
          "When comparing whether the equivalence has a solution, it reports unknown.\n"
          "When comparing whether the equivalence always hold, it passes the check.\n"
          "But if we scale up, time explodes. By zoom=3, it runs forever.")
    test_z3_selfattn_bw(zoom=1)
    test_z3_selfattn_bw(zoom=2)
    # test_z3_selfattn_bw(zoom=3) # hangs or slow
    print()
    
    # op with nonlinear but not Max
    print("Use SymPy to check backward_self_attention, which contains nonlinear ops.\n"
          "SymPy's run time is similar to Z3. zoom=1 finishes fast; zoom=2 takes much longer time.")
    test_sympy_selfattn_bw(zoom=1)
    test_sympy_selfattn_bw(zoom=2)
    # test_sympy_selfattn_bw(zoom=3) # hangs or slow
    print()
    
    # compare how z3 and sympy is capable of Max semantics
    print("Use Z3 and SymPy to check global Max V.S. aggregation of partial max.\n"
          "The result shows both tools can pass in this prototype.")
    test_z3_max()
    test_sympy_max()
    print()
    
    # compare how z3 and sympy is capable of flash attention
    print("Use Z3 to check flash attention, which contains nonlinear ops and Max.\n"
          "Z3 only passes for the smallest scale (input tensors have only 1 element).\n"
          "SymPy can understand complicated Max, and checks larger inputs.\n"
          "Note SymPy's version also has larger Block Sizes (Bc. Br).")
    test_z3_flashattn1(zoom=1)
    # test_z3_flashattn1(zoom=2) # hangs or slow
    test_sympy_flashattn1(zoom=1)
    test_sympy_flashattn1(zoom=2)
    test_sympy_flashattn1(zoom=3)
    print()
    
