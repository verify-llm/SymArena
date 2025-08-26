from typing import List, Any
import numpy as np


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
    

def create_tensor(shape, prefix: str, dtype: object, ctx=None):
    names = create_name_tensor(shape, prefix)
    return create_tensors_from_names(names, dtype, ctx)


def create_tensors_from_names(names, dtype, ctx=None):
    return np.vectorize(lambda x: dtype(x, ctx=ctx))(names)


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
