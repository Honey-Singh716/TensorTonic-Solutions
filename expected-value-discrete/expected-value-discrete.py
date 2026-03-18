import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x)
    p = np.array(p)

    # Check same shape
    if x.shape != p.shape:
        raise ValueError("x and p must have same shape")

    # Check probabilities sum to 1 (with tolerance)
    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")

    return np.sum(x * p)