import z3
import sympy as sp
from sympy.matrices.expressions import BlockMatrix, block_collapse
import numpy as np
from utils import create_tensor, equalize_tensors, print_check, z3_tactic_check_unsat
from operators.z3_attn import self_attention
from time import time

def test_col_linear(zoom=1):
    print("🌿 SymPy Column Linear")
    t = time()
    A = sp.MatrixSymbol("A", 4 * zoom, 4 * zoom)
    B1 = sp.MatrixSymbol("B1", 4 * zoom, 2 * zoom)
    B2 = sp.MatrixSymbol("B2", 4 * zoom, 2 * zoom)
    
    # [A*B1 | A*B2]
    Y1 = BlockMatrix([[A*B1, A*B2]])

    # A * [B1 | B2]  → block_collapse distributes A over the block row
    Y2 = block_collapse(A * BlockMatrix([[B1, B2]]))
    
    expr = Y1 - Y2
    is_zero = expr == sp.ZeroMatrix(*expr.shape)
    print_check(is_zero, True, "Column Linear")
    
    


if __name__ == "__main__":
    test_col_linear(zoom=1000000)
    """
    SymPy has MatrixSymbol and Matrix sorts.
    
    MatrixSymbol represent matrix as one symbol.
    MatrixSymbol can scale without side effect (see zoom=1xxxx).
    
    Matrix is concrete matrix with elements to be number or symbolics.
    So its scalability is almost the same as e.g. np.ndarray with Symbol elements.
    
    MatrixSymbol can only represent 2-D matrix, and support basic matrix operations.
    These factors provides insufficient support for LLM representations.
    """
