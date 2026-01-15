#L2 normalize
import numpy as np

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize vectors along last dimension
    vectors: (N, D)
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D array")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def normalize_inplace(vectors: np.ndarray, eps: float = 1e-12):
    """
    In-place normalization to save memory
    """
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms < eps] = 1.0
    vectors /= norms[:, None]
