import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    m, n = X.shape

    W = np.zeros((n,1), dtype=float)
    b = 0.0

    for i in range(steps):

        z = X @ W + b
        A = _sigmoid(z)

        gd_w = (1/m) * (X.T @ (A - y))
        gd_b = (1/m) * np.sum(A - y)

        W = W - lr * gd_w
        b = b - lr * gd_b

    return W.flatten(), b