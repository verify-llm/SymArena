import z3
import sympy as sp
import numpy as np
from utils import create_tensor, equalize_tensors, print_check, z3_tactic_check_unsat, wolfram_equiv, sp_build_subs_from_pairs, sp_apply_subs_tensor, sp_tensors_equal_symbolic, sp_tensor_view
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
from time import time



def test_sympy_max():
    print("🌿 SymPy Max")
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", positive=True) 
    expr1 = sp.Max(sp.Max(x1, x2), sp.Max(x3, x4)) 
    expr2 = sp.Max(x1,x2,x3,x4)
    
    result = wolfram_equiv(expr1, expr2)
    print(result)
    
def test_sympy_selfattn_bw(zoom=1, use_wolfram=True):
    print(f"🌿 z3 Self_attention Backward Zoom={zoom}")

    """
    # L N E, (h d 3) E -> L N (h d 3)
    qkv = linear(query, qkv_proj, qkv_bias)
    # 
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

    # ok1 = sp_tensors_equal_symbolic(y11, y21_s, use_wolfram)
    # ok2 = sp_tensors_equal_symbolic(y12, y22_s, use_wolfram)
    # ok3 = sp_tensors_equal_symbolic(y13, y23_s, use_wolfram)
    # ok4 = sp_tensors_equal_symbolic(y14, y24_s, use_wolfram)
    # all_ok = ok1 and ok2 and ok3 and ok4
    # print_check(all_ok, True, "bw_self_attention equivalence (SymPy)")
    
    print("sympy solving")
    t = time()
    ok1 = sp_tensors_equal_symbolic(y11, y21_s, False)
    ok2 = sp_tensors_equal_symbolic(y12, y22_s, False)
    ok3 = sp_tensors_equal_symbolic(y13, y23_s, False)
    ok4 = sp_tensors_equal_symbolic(y14, y24_s, False)
    all_ok = ok1 and ok2 and ok3 and ok4
    print_check(all_ok, True, "bw_self_attention equivalence (SymPy)")
    print("sympy solved", time() - t)
    
    print("wolfram solving")
    t = time()
    ok1 = sp_tensors_equal_symbolic(y11, y21_s, True)
    ok2 = sp_tensors_equal_symbolic(y12, y22_s, True)
    ok3 = sp_tensors_equal_symbolic(y13, y23_s, True)
    ok4 = sp_tensors_equal_symbolic(y14, y24_s, True)
    all_ok = ok1 and ok2 and ok3 and ok4
    print_check(all_ok, True, "bw_self_attention equivalence (Wolfram)")
    print("wolfram solved", time() - t)
    

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

    print("wolfram solving")
    t = time()
    is_zero = sp_tensors_equal_symbolic(y1, y2, True)
    print_check(is_zero, True, "wolfram checks flashattn")
    print("wolfram solved", time() - t)
    
    print("sympy solving")
    t = time()
    is_zero = sp_tensors_equal_symbolic(y1, y2, False)
    print_check(is_zero, True, "sympy checks flashattn")
    print("sympy solved", time() - t)
    
    
    
    
        
        
if __name__ == "__main__":
    # test_sympy_max()
    # test_sympy_selfattn_bw(zoom=3)
    test_sympy_flashattn1(zoom=2)

