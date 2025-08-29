import z3
import sympy as sp
import numpy as np
from utils import create_tensor, equalize_tensors, print_check, z3_tactic_check_unsat
from operators.z3_attn import self_attention


def test_sympy_sumsquare():
    print("🌿 SymPy Sum Square")

    x1, x2 = sp.symbols("x1 x2", real=True)

    y1 = (x1 + x2) ** 2
    y2 = x1**2 + x2**2 + 2 * x1 * x2

    # 1) Prove y1 == y2 symbolically in several independent ways
    # a) Simplify the difference to 0
    diff = sp.simplify(y1 - y2)
    print_check(diff == 0, True, "simplify(y1 - y2) == 0")

    # b) Structural equality after expansion
    eq_expand = sp.expand(y1) == sp.expand(y2)
    print_check(eq_expand, True, "expand equality")

    # c) Polynomial identity check
    poly_zero = sp.Poly(y1 - y2, x1, x2).is_zero
    print_check(poly_zero, True, "polynomial identity (is_zero)")

    # d) Eq + simplify to True
    eq_true = sp.simplify(sp.Eq(y1, y2)) is True
    print_check(eq_true, True, "Eq(...) simplifies to True")

    # 2) “Negation” of equivalence: show there’s no counterexample
    # In SymPy terms, that means the difference is the zero polynomial.
    # If it's identically zero, the set { (x1,x2) | y1!=y2 } is empty.
    # We assert this again via several routes:

    # a) Factor and check it’s 0
    factored = sp.factor(y1 - y2)
    print_check(factored == 0, True, "factor(y1 - y2) == 0")

    # b) Together/cancel won’t change a zero identity either
    together_zero = sp.together(y1 - y2)
    print_check(together_zero == 0, True, "together(y1 - y2) == 0")

    # c) Ask for the solution set of y1 - y2 != 0 (should be empty)
    # SymPy doesn’t directly “solve” general multivariate inequalities into a neat set,
    # but the polynomial-identity check above already certifies there is no solution.
    has_counterexample = not poly_zero
    print_check(has_counterexample, False, "counterexample exists?")


def test_sympy_squarediv():
    print("🌿 SymPy Square Div")

    x1, x2 = sp.symbols("x1 x2", real=True)

    y1 = (x1**2 - x2**2) / (x1 - x2)
    y2 = x1 + x2

    # 1) Prove y1 == y2 symbolically in several independent ways
    # a) Simplify the difference to 0
    diff = sp.simplify(y1 - y2)
    print_check(diff == 0, True, "simplify(y1 - y2) == 0")

    # b) Structural equality after expansion
    eq_expand = sp.expand(y1) == sp.expand(y2)
    print_check(eq_expand, True, "expand equality")

    # c) Polynomial identity check
    try:
        poly_zero = sp.Poly(y1 - y2, x1, x2).is_zero
    except:
        poly_zero = False
    print_check(poly_zero, True, "polynomial identity (is_zero)")

    # d) Eq + simplify to True
    eq_true = sp.simplify(sp.Eq(y1, y2)) is True
    print_check(eq_true, True, "Eq(...) simplifies to True")

    # 2) “Negation” of equivalence: show there’s no counterexample
    # In SymPy terms, that means the difference is the zero polynomial.
    # If it's identically zero, the set { (x1,x2) | y1!=y2 } is empty.
    # We assert this again via several routes:

    # a) Factor and check it’s 0
    factored = sp.factor(y1 - y2)
    print_check(factored == 0, True, "factor(y1 - y2) == 0")

    # b) Together/cancel won’t change a zero identity either
    together_zero = sp.together(y1 - y2)
    print_check(together_zero == 0, True, "together(y1 - y2) == 0")

    # c) Ask for the solution set of y1 - y2 != 0 (should be empty)
    # SymPy doesn’t directly “solve” general multivariate inequalities into a neat set,
    # but the polynomial-identity check above already certifies there is no solution.
    has_counterexample = not poly_zero
    print_check(has_counterexample, False, "counterexample exists?")


def test_sympy_abscomp_unknown():
    print("🌿 SymPy Abs Comparison Unknown")
    x = sp.symbols("x")
    A = sp.sqrt(x**2)
    B = sp.Abs(x)

    expr = A - B
    equal = sp.simplify(expr) == 0
    print_check(equal, True, "A==B")


def test_sympy_abscomp_known():
    print("🌿 SymPy Abs Comparison Known")
    x = sp.symbols("x", real=True)
    A = sp.sqrt(x**2)
    B = sp.Abs(x)

    expr = A - B
    equal = sp.simplify(expr) == 0
    print_check(equal, True, "A==B")


def test_sympy_log():
    print("🌿 SymPy Log")
    x, y = sp.symbols("x y", real=True, positive=True)

    expr = sp.log(x * y) - sp.log(x) - sp.log(y)

    is_zero = sp.simplify(expr) == 0
    print_check(is_zero, True, "log(xy)/log(x)/log(y)=log(1)=0")


def test_sympy_divsqrt():
    print("🌿 SymPy Div Sqrt")
    x1, x2 = sp.symbols("x1 x2", positive=True)  # assume >0 so sqrt well-defined

    expr = 1 / sp.sqrt(x1) - 1 / sp.sqrt(x2)
    expr_sub = expr.subs(x2, x1)

    is_zero = sp.simplify(expr_sub) == 0
    print_check(is_zero, True, "1/sqrt(x1) == 1/sqrt(x2) given x1==x2")
    

def test_sympy_max():
    print("🌿 SymPy Max")
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", positive=True) 
    expr = sp.Max(sp.Max(x1, x2), sp.Max(x3, x4)) - sp.Max(x1,x2,x3,x4)
    is_zero = sp.simplify(expr) == 0
    print_check(is_zero, True, "Max")


if __name__ == "__main__":
    test_sympy_sumsquare()
    test_sympy_squarediv()
    test_sympy_abscomp_unknown()
    test_sympy_abscomp_known()
    test_sympy_log()
    test_sympy_divsqrt()
    test_sympy_max()
