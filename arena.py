import z3
import numpy as np
from utils import create_tensor, equalize_tensors
from operators.z3_attn import self_attention


def print_check(result, expect):
    print(f"Result: {result}, Expect: {expect}, {'✅' if result == expect else '🚨'}")


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
    print_check(result, z3.sat)

    solver = z3.Solver()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat)


def test_z3_divsqrt():
    print("🚀 z3 DivSqrt")
    shape = (2, 2)

    x1 = create_tensor(shape, "x1", z3.Real)
    x2 = create_tensor(shape, "x2", z3.Real)

    def op(t):
        return t**-0.5

    y1 = op(x1)
    y2 = op(x2)

    x1_eq_x2 = equalize_tensors([x1, x2])
    y1_eq_y2 = equalize_tensors([y1, y2])
    y1_neq_y2 = z3.Not(z3.And(y1_eq_y2))

    solver = z3.Solver()
    solver.check()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat)

    solver = z3.Solver()
    solver.check()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat)


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
    solver.check()
    solver.add(x1_eq_x2)
    solver.add(y1_eq_y2)
    result = solver.check()
    print_check(result, z3.sat)

    solver = z3.Solver()
    solver.check()
    solver.add(x1_eq_x2)
    solver.add(y1_neq_y2)
    result = solver.check()
    print_check(result, z3.unsat)


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
    solver.check()
    solver.add(input_eq)
    solver.add(output_eq)
    result = solver.check()
    print_check(result, z3.sat)

    solver = z3.Solver()
    solver.check()
    solver.add(input_eq)
    solver.add(output_neq)
    result = solver.check()
    print_check(result, z3.unsat)


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

    print("start solving sat")
    solver = z3.Solver()
    solver.check()
    solver.add(input_eq)
    solver.add(output_eq)
    result = solver.check()
    print_check(result, z3.sat)

    print("start solving unsat")
    solver = z3.Solver()
    solver.check()
    solver.add(input_eq)
    solver.add(output_neq)
    result = solver.check()
    print_check(result, z3.unsat)


if __name__ == "__main__":
    # test_z3_sanity()
    # test_z3_divsqrt()
    # test_z3_divexp()
    # test_z3_selfattn()
    test_z3_selfattn_tp()
