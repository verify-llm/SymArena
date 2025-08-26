import z3
import numpy as np

Z3Tensor = np.ndarray


def linear(input: Z3Tensor, weight: Z3Tensor, bias: Z3Tensor | None = None) -> Z3Tensor:
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


def softmax(input: Z3Tensor, dim: int, **kwargs) -> Z3Tensor:
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
