import numpy as np

def dropout(x, p=0.5, rng=None):

    x = np.array(x)

    if rng is None:
        rng = np.random.default_rng()

    keep_prob = 1 - p
    scale = 1 / keep_prob

    mask = (rng.random(x.shape) < 1 - p).astype(x.dtype)

    mask = mask * scale

    output = x * mask

    return output, mask