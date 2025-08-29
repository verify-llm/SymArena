import z3
import numpy as np
from utils import (
    create_tensor,
    equalize_tensors,
    print_check,
    z3_tactic_check_unsat,
    concrete_z3,
)
from operators.z3_attn import (
    self_attention,
    bw_softmax,
    softmax,
    bw_self_attention,
    standard_attention,
    flash_attention,
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


if __name__ == "__main__":
    # test_z3_sanity()
    # test_z3_linear()
    # test_z3_divsqrt()
    # test_z3_divexp()
    # test_z3_selfattn()
    # test_z3_selfattn_tp()
    # test_z3_softmax()
    # test_z3_softmax_bw()
    # test_z3_selfattn_bw()
    test_z3_flashattn1()
    # test_z3_flashattn2()
