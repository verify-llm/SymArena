import z3
import numpy as np
from typing import List
import math
import sympy as sp

Z3Tensor = np.ndarray


def linear(input: Z3Tensor, weight: Z3Tensor, bias = None) -> Z3Tensor:
    """torch.nn.functional.linear"""
    result = input @ weight.T
    if bias is not None:
        result += bias
    return result


def transpose(input: Z3Tensor, dim0: int, dim1: int) -> Z3Tensor:
    """torch.transpose"""
    axis = list(range(len(input.shape)))
    axis[dim0], axis[dim1] = axis[dim1], axis[dim0]
    return np.transpose(input, axis)


def softmax(input: Z3Tensor, dim: int = -1, **kwargs) -> Z3Tensor:
    """torch.nn.functional.softmax"""
    e_x = np.e**input
    e_x_sum = np.sum(e_x, axis=dim, keepdims=True)
    return e_x / e_x_sum


def self_attention(
    query: Z3Tensor,
    qkv_proj: Z3Tensor,
    qkv_bias: Z3Tensor,
    out_proj: Z3Tensor,
    h: int,
    scale: float,
    mask: bool = False,
) -> Z3Tensor:
    """examples.nlp.blocks.attention.self_attention"""
    num_head = h
    L, N = query.shape[0], query.shape[1]
    dim_head = qkv_proj.shape[0] // num_head // 3

    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # L N (h d 3) -> L N (h d) 3
    qkv = qkv.reshape(L, N, num_head * dim_head, 3)
    # L N (h d) 3 -> L N (h d) 1, L N (h d) 1, L N (h d) 1
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d
    k = k.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d
    v = v.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d

    q = transpose(q, 0, 1)  # L (N h) d -> (N h) L d
    q = q * scale  # (N h) L d, 1 -> (N h) L d
    k = transpose(k, 0, 1)  # L (N h) d -> (N h) L d
    k = transpose(k, 1, 2)  # (N h) L d -> (N h) d L

    attn: Z3Tensor = q @ k  # (N h) L d, (N h) d L -> (N h) L L

    if mask:
        attn = attn.reshape(N, num_head, L, L)
        ones = np.ones([N, L, L])
        amask = np.tril(ones)
        amask = amask.reshape(N, 1, L, L)
        amask = amask < 0.5
        attn[amask.repeat(attn.shape[1], axis=1)] = -10000
        attn = attn.reshape(N * num_head, L, L)

    attn = softmax(attn, dim=-1)  # (N h) L L -> (N h) L L
    v = transpose(v, 0, 1)  # L (N h) d -> (N h) L d
    output = attn @ v  # (N h) L L, (N h) L d -> (N h) L d
    output = transpose(output, 0, 1)  # (N h) L d -> L (N h) d
    # L (N h) d -> L N (h d)
    output = output.reshape(L, N, num_head * dim_head)
    # L N (h d), E (h d)  -> L N E
    output = linear(output, out_proj, None)
    return output


def bw_self_attention(
    g: Z3Tensor,
    query: Z3Tensor,
    qkv_proj: Z3Tensor,
    qkv_bias: Z3Tensor,
    out_proj: Z3Tensor,
    h: int,
    scale: float,
    mask: bool,
) -> Z3Tensor:
    """examples.nlp.blocks.attention.self_attention"""
    num_head = h
    L, N = query.shape[0], query.shape[1]
    dim_head = qkv_proj.shape[0] // num_head // 3

    # L N E, (h d 3) E -> L N (h d 3)
    qkv1 = linear(query, qkv_proj, qkv_bias)
    # L N (h d 3) -> L N (h d) 3
    qkv2 = qkv1.reshape(L, N, num_head * dim_head, 3)
    # L N (h d) 3 -> L N (h d) 1, L N (h d) 1, L N (h d) 1
    q, k, v = np.split(qkv2, 3, axis=-1)
    q1 = q.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d
    k1 = k.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d
    v1 = v.reshape(L, N * num_head, dim_head)  # L N (h d) 1 -> L (N h) d

    q2 = transpose(q1, 0, 1)  # L (N h) d -> (N h) L d
    q3 = q2 * scale  # (N h) L d, 1 -> (N h) L d
    k2 = transpose(k1, 0, 1)  # L (N h) d -> (N h) L d
    k3 = transpose(k2, 1, 2)  # (N h) L d -> (N h) d L

    attn1: Z3Tensor = q3 @ k3  # (N h) L d, (N h) d L -> (N h) L L

    mask = False  # invalidate mask, as we did not implement its bw
    if mask:
        attn2 = attn1.reshape(N, num_head, L, L)
        ones = np.ones([N, L, L])
        amask1 = np.tril(ones)
        amask2 = amask1.reshape(N, 1, L, L)
        amask3 = amask2 < 0.5
        attn2[amask3] = -10000
        attn3 = attn2.reshape(N * num_head, L, L)
    else:
        attn3 = attn1

    attn4 = softmax(attn3, dim=-1)  # (N h) L L -> (N h) L L
    attn5 = attn4  # (N h) L L -> (N h) L L
    v2 = transpose(v1, 0, 1)  # L (N h) d -> (N h) L d
    output1 = attn5 @ v2  # (N h) L L, (N h) L d -> (N h) L d
    output2 = transpose(output1, 0, 1)  # (N h) L d -> L (N h) d
    # L (N h) d -> L N (h d)
    output3 = output2.reshape(L, N, num_head * dim_head)
    # L N (h d), E (h d)  -> L N E
    _ = linear(output3, out_proj, None)

    """ Backward Gradient Calculation """
    g_output3, g_out_proj = bw_linear(g, output3, out_proj, None)
    g_output2 = g_output3.reshape(L, N * num_head, dim_head)  # L (N h) d <- L N (h d)
    g_output1 = bw_transpose(g_output2, output1, 0, 1)
    g_attn5, g_v2 = bw_matmul(g_output1, attn5, v2)
    g_v1 = bw_transpose(g_v2, v1, 0, 1)
    g_attn4 = g_attn5
    g_attn3 = bw_softmax(g_attn4, attn3, dim=-1)

    if mask:
        g_attn2 = g_attn3.reshape(N, num_head, L, L)
        g_attn2[amask3] = 0
        g_attn1 = g_attn2.reshape(N * num_head, L, L)
    else:
        g_attn1 = g_attn3

    g_q3, g_k3 = bw_matmul(g_attn1, q3, k3)
    g_k2 = bw_transpose(g_k3, k2, 1, 2)
    g_k1 = bw_transpose(g_k2, k1, 0, 1)
    g_q2 = g_q3 * scale
    g_q1 = bw_transpose(g_q2, q1, 0, 1)

    # L N (h d) 1 <- L (N h) d
    g_v = g_v1.reshape(L, N, num_head * dim_head, 1)
    # L N (h d) 1 <- L (N h) d
    g_k = g_k1.reshape(L, N, num_head * dim_head, 1)
    # L N (h d) 1 <- L (N h) d
    g_q = g_q1.reshape(L, N, num_head * dim_head, 1)

    g_qkv2 = np.concatenate([g_q, g_k, g_v], axis=-1)
    # L N (h d 3) <- L N (h d) 3
    g_qkv1 = g_qkv2.reshape(L, N, h * dim_head * 3)
    g_query, g_qkv_proj, g_qkv_bias = bw_linear(g_qkv1, query, qkv_proj, qkv_bias)

    return [g_query, g_qkv_proj, g_qkv_bias, g_out_proj]


def bw_softmax(g, t, dim=-1):
    e_x = np.e**t
    e_x_sum = np.sum(e_x, axis=dim, keepdims=True)
    softmax_output = e_x / e_x_sum
    g_input = softmax_output * (g - np.sum(g * softmax_output, axis=dim, keepdims=True))
    return g_input


def bw_transpose(g: Z3Tensor, input: Z3Tensor, dim0: int, dim1: int) -> Z3Tensor:
    """torch.transpose"""
    axis = list(range(len(g.shape)))
    axis[dim0], axis[dim1] = axis[dim1], axis[dim0]
    return np.transpose(g, axis)


def bw_matmul(g: Z3Tensor, X: Z3Tensor, W: Z3Tensor) -> List[Z3Tensor]:
    """
    Compute the gradients of W given X, W and the gradient of Y.
    Forward computation is Y = X @ W
    Parameters:
    - X: numpy.ndarray with shape (b1, b2, ..., bn, m, k)
    - W: numpy.ndarray with shape (..., bn, k, n)
    - g: numpy.ndarray with shape (b1, b2, ..., bn, m, n)
    Returns:
    - g_X: numpy.ndarray with shape (b1, b2, ..., bn, m, k)
    - g_W: numpy.ndarray with shape (..., bn, k, n)
    """
    # Number of dimensions in X and g_Y
    num_dims_X = X.ndim
    num_dims_W = W.ndim
    assert num_dims_X - 2 <= 26, "einsum notation use single letter to denote dimension"
    assert num_dims_W - 2 <= 26, "einsum notation use single letter to denote dimension"
    batch_dims = "".join(chr(ord("A") + i) for i in range(num_dims_X - 2))
    subscript_X = batch_dims + "mk"
    subscript_W = subscript_X[-num_dims_W:-2] + "kn"
    subscript_g = batch_dims + "mn"

    g_X = np.einsum(f"{subscript_g},{subscript_W}->{subscript_X}", g, W)
    g_W = np.einsum(f"{subscript_X},{subscript_g}->{subscript_W}", X, g)
    return [g_X, g_W]


def bw_linear(
    g: Z3Tensor, input: Z3Tensor, weight: Z3Tensor, bias = None
) -> List[Z3Tensor]:
    """torch.nn.functional.linear"""
    g_input, g_weight = bw_matmul(g, input, weight.T)
    g_weight = g_weight.T
    # g_weight[0][0] += 1 # MUTATION
    grads = [g_input, g_weight]
    if bias is not None:
        g_bias = np.sum(g, axis=tuple(range(g.ndim - 1)))
        grads += [g_bias]
    return grads


def standard_attention(Q: Z3Tensor, K: Z3Tensor, V: Z3Tensor, N: int, d: int):
    """N: seqlen, d: model_dim"""
    assert Q.shape == K.shape == V.shape == (N, d)
    S = Q @ K.T
    P = softmax(S, -1)
    O = P @ V
    return O


def symbmax_scalar(x, y):
    return z3.If(x >= y, x, y)


def arrmax(arr):
    it = iter(arr)
    m = next(it)
    for e in it:
        m = symbmax_scalar(m, e)
    return m


def rowmax(X: Z3Tensor):
    assert X.ndim == 2
    result = []
    for row in X:
        result.append(arrmax(row))
    result = np.array(result)
    assert result.shape == (X.shape[0],)
    return result


def rowsum(X: Z3Tensor):
    assert X.ndim == 2
    return np.sum(X, axis=1)


def vmax(a: Z3Tensor, b: Z3Tensor):
    """Elementwise Z3 max for 1-D arrays (dtype=object)."""
    assert a.ndim == b.ndim == 1 and a.shape == b.shape
    return np.fromiter(
        (symbmax_scalar(x, y) for x, y in zip(a, b)), dtype=object, count=a.shape[0]
    )


def flash_attention(Q: Z3Tensor, K: Z3Tensor, V: Z3Tensor, N: int, d: int, M: int):
    """N: seqlen, d: model_dim, M: on-chip SRAM size"""
    assert Q.shape == K.shape == V.shape == (N, d)

    Bc = math.ceil(M / 4 / d)
    Br = min(Bc, d)
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    print(f"   Bc={Bc}, Br={Br}, Tc={Tc}, Tr={Tr}")

    ZERO = z3.IntVal(0)
    NEG_INF = z3.IntVal(-(10**9))
    O = np.full((N, d), ZERO, dtype=object)
    l = np.full((N,), ZERO, dtype=object)
    m = np.full((N,), NEG_INF, dtype=object)

    e = np.e
    diag = np.diag

    for j in range(0, Tc):
        Kj = K[j * Bc : (j + 1) * Bc, :]
        Vj = V[j * Bc : (j + 1) * Bc, :]
        for i in range(0, Tr):
            Qi = Q[i * Br : (i + 1) * Br, :]
            Oi = O[i * Br : (i + 1) * Br, :]
            li = l[i * Br : (i + 1) * Br]
            mi = m[i * Br : (i + 1) * Br]
            Sij = Qi @ Kj.T
            mij: Z3Tensor = rowmax(Sij)
            # assert mij.shape == (Br,)
            Pij = e ** (Sij - mij[:, None])
            # assert Pij.shape == (Br, Bc)
            lij = rowsum(Pij)
            # assert lij.shape == (Br,)
            mi_new = vmax(mi, mij)
            # assert mi_new.shape == (Br,)
            li_new = e ** (mi - mi_new) * li + e ** (mij - mi_new) * lij
            # assert li_new.shape == (Br,)
            O[i * Br : (i + 1) * Br, :] = (
                e ** (mi - mi_new)[:, None] * (li[:, None] * Oi)
                + e ** (mij - mi_new)[:, None] * (Pij @ Vj)
            ) / li_new[:, None]
            l[i * Br : (i + 1) * Br] = li_new
            m[i * Br : (i + 1) * Br] = mi_new
    return O


def rowmax_sympy(X: Z3Tensor):
    assert X.ndim == 2
    result = []
    for row in X:
        result.append(sp.Max(*row))
    result = np.array(result)
    assert result.shape == (X.shape[0],)
    return result


def vmax_sympy(a: Z3Tensor, b: Z3Tensor):
    """Elementwise Z3 max for 1-D arrays (dtype=object)."""
    assert a.ndim == b.ndim == 1 and a.shape == b.shape
    return np.fromiter(
        (sp.Max(x, y) for x, y in zip(a, b)), dtype=object, count=a.shape[0]
    )

def flash_attention_sympy(Q: Z3Tensor, K: Z3Tensor, V: Z3Tensor, N: int, d: int, M: int):
    """N: seqlen, d: model_dim, M: on-chip SRAM size"""
    assert Q.shape == K.shape == V.shape == (N, d)

    Bc = math.ceil(M / 4 / d)
    Br = min(Bc, d)
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    print(f"   Bc={Bc}, Br={Br}, Tc={Tc}, Tr={Tr}")

    ZERO = 0
    NEG_INF = -(10**9)
    O = np.full((N, d), ZERO, dtype=object)
    l = np.full((N,), ZERO, dtype=object)
    m = np.full((N,), NEG_INF, dtype=object)

    e = np.e

    for j in range(0, Tc):
        Kj = K[j * Bc : (j + 1) * Bc, :]
        Vj = V[j * Bc : (j + 1) * Bc, :]
        for i in range(0, Tr):
            Qi = Q[i * Br : (i + 1) * Br, :]
            Oi = O[i * Br : (i + 1) * Br, :]
            li = l[i * Br : (i + 1) * Br]
            mi = m[i * Br : (i + 1) * Br]
            Sij = Qi @ Kj.T
            mij: Z3Tensor = rowmax_sympy(Sij)
            # assert mij.shape == (Br,)
            Pij = e ** (Sij - mij[:, None])
            # assert Pij.shape == (Br, Bc)
            lij = rowsum(Pij)
            # assert lij.shape == (Br,)
            mi_new = vmax_sympy(mi, mij)
            # assert mi_new.shape == (Br,)
            li_new = e ** (mi - mi_new) * li + e ** (mij - mi_new) * lij
            # assert li_new.shape == (Br,)
            O[i * Br : (i + 1) * Br, :] = (
                e ** (mi - mi_new)[:, None] * (li[:, None] * Oi)
                + e ** (mij - mi_new)[:, None] * (Pij @ Vj)
            ) / li_new[:, None]
            l[i * Br : (i + 1) * Br] = li_new
            m[i * Br : (i + 1) * Br] = mi_new
    return O
