import z3
import numpy as np
from utils import create_tensor, equalize_tensors, print_check, z3_tactic_check_unsat
from operators.z3_attn import self_attention, bw_softmax, softmax, bw_self_attention
from time import time


def test_z3_sanity():
    print("🚀 z3 Sanity")
    shape = (2, 2)

    x1 = create_tensor(shape, "x1", z3.Real)
    x2 = create_tensor(shape, "x2", z3.Real)

    y1 = x1
    y2 = x2

    x1_eq_x2 = equalize_tensors([x1, x2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")

    tactic_result = z3_tactic_check_unsat(x1_eq_x2 + [y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_z3_divsqrt():
    print("🚀 z3 DivSqrt")

    x1 = z3.Real("x1")
    x2 = z3.Real("x2")

    def op(t):
        return t**-0.5

    y1 = op(x1)
    y2 = op(x2)

    x1_eq_x2 = equalize_tensors([x1, x2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")

    tactic_result = z3_tactic_check_unsat(x1_eq_x2 + [y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_z3_divexp():
    print("🚀 z3 DivExp")
    shape = (2, 2)

    x1 = create_tensor(shape, "x1", z3.Real)
    x2 = create_tensor(shape, "x2", z3.Real)

    def op(t):
        return 1 / np.e**t

    y1 = op(x1)
    y2 = op(x2)

    x1_eq_x2 = equalize_tensors([x1, x2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")

    tactic_result = z3_tactic_check_unsat(x1_eq_x2 + [y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_z3_selfattn():
    print("🚀 z3 Self_attention")

    """
    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # require: N = h * d
    """

    L = E = N = h = d = 2
    query_shape = (L, N, E)
    qkv_proj_shape = (3 * h * d, E)
    qkv_bias_shape = (3 * h * d,)
    out_proj_shape = (E, h * d)

    query1 = create_tensor(query_shape, "query1", z3.Real)
    qkv_proj1 = create_tensor(qkv_proj_shape, "qkv_proj1", z3.Real)
    qkv_bias1 = create_tensor(qkv_bias_shape, "qkv_bias1", z3.Real)
    out_proj1 = create_tensor(out_proj_shape, "out_proj1", z3.Real)

    query2 = create_tensor(query_shape, "query2", z3.Real)
    qkv_proj2 = create_tensor(qkv_proj_shape, "qkv_proj2", z3.Real)
    qkv_bias2 = create_tensor(qkv_bias_shape, "qkv_bias2", z3.Real)
    out_proj2 = create_tensor(out_proj_shape, "out_proj2", z3.Real)

    scale = 1
    mask = True
    y1 = self_attention(query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask)
    y2 = self_attention(query2, qkv_proj2, qkv_bias2, out_proj2, h, scale, mask)

    input_eq = (
        equalize_tensors([query1, query2])
        + equalize_tensors([qkv_proj1, qkv_proj2])
        + equalize_tensors([qkv_bias1, qkv_bias2])
        + equalize_tensors([out_proj1, out_proj2])
    )

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


def test_z3_selfattn_tp():
    print("🚀 z3 Self_attention with tensor partitioning")

    """
    # require: N = h * d
    @nnscaler.register_op('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
    y1 = self_attention(query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask)
    """
    h = 4
    L = E = N = d = 2
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


def test_z3_softmax():
    print("🚀 z3 Softmax")
    shape = (2, 2)

    x1 = create_tensor(shape, "x1", z3.Real)
    x2 = create_tensor(shape, "x2", z3.Real)

    y1 = softmax(x1, dim=-1)
    y2 = softmax(x2, dim=-1)

    x1_eq_x2 = equalize_tensors([x1, x2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")

    tactic_result = z3_tactic_check_unsat(x1_eq_x2 + [y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_z3_softmax_bw():
    print("🚀 z3 Softmax Backward")
    shape = (2, 2)

    x1 = create_tensor(shape, "x1", z3.Real)
    x2 = create_tensor(shape, "x2", z3.Real)
    g1 = create_tensor(shape, "g1", z3.Real)
    g2 = create_tensor(shape, "g2", z3.Real)

    y1 = bw_softmax(g1, x1)
    y2 = bw_softmax(g2, x2)

    x1_eq_x2 = equalize_tensors([x1, x2]) + equalize_tensors([g1, g2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat, "solver")

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat, "solver")

    tactic_result = z3_tactic_check_unsat(x1_eq_x2 + [y1_neq_y2])
    print_check(tactic_result, True, "tactic")


def test_z3_selfattn_bw():
    print("🚀 z3 Self_attention Backward")

    """
    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # require: N = h * d
    """

    L = E = N = h = d = 8
    query_shape = (L, N, E)
    qkv_proj_shape = (3 * h * d, E)
    qkv_bias_shape = (3 * h * d,)
    out_proj_shape = (E, h * d)

    print("creating tensors")
    t = time()
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
    print(time() - t)

    print("executing operation")
    t = time()
    scale = 1
    mask = True
    # g_query, g_qkv_proj, g_qkv_bias, g_out_proj
    y11, y12, y13, y14 = bw_self_attention(
        g1, query1, qkv_proj1, qkv_bias1, out_proj1, h, scale, mask
    )
    y21, y22, y23, y24 = bw_self_attention(
        g2, query2, qkv_proj2, qkv_bias2, out_proj2, h, scale, mask
    )
    print(time() - t)

    print("equalizing inputs")
    t = time()
    input_eq = (
        equalize_tensors([g1, g2])
        + equalize_tensors([query1, query2])
        + equalize_tensors([qkv_proj1, qkv_proj2])
        + equalize_tensors([qkv_bias1, qkv_bias2])
        + equalize_tensors([out_proj1, out_proj2])
    )
    print(time() - t)

    print("equalizing outputs")
    t = time()
    output_eq = (
        equalize_tensors([y11, y21])
        + equalize_tensors([y12, y22])
        + equalize_tensors([y13, y23])
        + equalize_tensors([y14, y24])
    )
    output_neq = z3.Not(z3.And(output_eq))
    print(time() - t)

    print("start checking")
    t = time()
    solver = z3.Solver()
    print("adding input_eq")
    solver.add(input_eq)
    print("adding output_eq")
    solver.add(output_eq)
    print("checking")
    result = solver.check()
    print_check(result, z3.sat, "solver")
    print(time() - t)

    t = time()
    solver = z3.Solver()
    print("adding input_eq")
    solver.add(input_eq)
    print("adding output_neq")
    solver.add(output_neq)
    print("checking")
    result = solver.check()
    print_check(result, z3.unsat, "solver")
    print(time() - t)

    t = time()
    tactic_result = z3_tactic_check_unsat(input_eq + [output_neq])
    print_check(tactic_result, True, "tactic")
    print(time() - t)


if __name__ == "__main__":
    # test_z3_sanity()
    # test_z3_divsqrt()
    # test_z3_divexp()
    # test_z3_selfattn()
    # test_z3_selfattn_tp()
    # test_z3_softmax()
    # test_z3_softmax_bw()
    test_z3_selfattn_bw()
