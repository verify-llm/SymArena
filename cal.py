import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerical stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def standard_attention_np(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Q, K, V: numpy arrays of shape (N, d)
    N: sequence length, d: model dimension
    """
    N, d = Q.shape
    assert K.shape == (N, d) and V.shape == (N, d)

    S = Q @ K.T                # (N, N) scores
    P = softmax(S, axis=-1)    # (N, N) probabilities
    O = P @ V                  # (N, d) output
    return O


Q = np.array([[1,2], [3,4]])
K = np.array([[5,6], [7,8]])
V = np.array([[9,10], [11,12]])
O = standard_attention_np(Q,K,V)
print(O)