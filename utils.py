from typing import List, Any
import numpy as np
import z3


def z3_to_python_number(val: z3.ExprRef):
    if z3.is_int_value(val):
        return val.as_long()
    elif z3.is_rational_value(val):
        num = val.numerator_as_long()
        den = val.denominator_as_long()
        return num / den  # float
    else:
        raise TypeError(f"Unsupported Z3 value type: {val}")


def concrete_z3(zt, model: z3.ModelRef) :
    if type(zt) is z3.ArithRef:
        return model.evaluate(zt, model_completion=True)
    flat_result = [
        z3_to_python_number(model.evaluate(x, model_completion=True))
        for x in zt.ravel()
    ]
    result = np.array(flat_result).reshape(zt.shape)
    return result


def print_check(result, expect, dscp=""):
    icon = "⚠️ " if result == z3.unknown else ("✅" if result == expect else "🚨")
    print(f"   {icon} {dscp} | Result: {result}, Expect: {expect}")


def create_name_tensor(shape, prefix):
    """Recursively create a tensor with the given dynamic shape."""
    if len(shape) == 1:
        return np.array([f"{prefix}[{i}]" for i in range(shape[0])])
    return np.array(
        [
            create_name_tensor(shape[1:], prefix=f"{prefix}[{i}]")
            for i in range(shape[0])
        ]
    )


def create_tensor(shape, prefix: str, dtype: object):
    names = create_name_tensor(shape, prefix)
    return create_tensors_from_names(names, dtype)


def create_tensors_from_names(names, dtype):
    return np.vectorize(lambda x: dtype(x))(names)


def equalize_tensors(ts: List[np.ndarray]) -> List:
    unique([type(t) for t in ts])
    constraints = []
    t0 = ts[0]
    if type(t0) is not np.ndarray:
        constraints = [t0 == t for t in ts[1:]]
    else:
        shape = unique([t.shape for t in ts])
        for t in ts[1:]:
            constraints.extend(t0[i] == t[i] for i in np.ndindex(shape))
    return constraints


def unique(l: List[Any]):
    if len(l) == 1:
        return l[0]
    assert len(set(l)) == 1, f"{set(l)}"
    return l[0]


def z3_tactic_check_unsat(constraints) -> bool:
    g = z3.Goal()
    g.add(*constraints)
    simplify = z3.Tactic("solve-eqs")
    simplified_goal = simplify(g)
    all_unsat = all(str(subgoal) == "[False]" for subgoal in simplified_goal)
    return all_unsat
