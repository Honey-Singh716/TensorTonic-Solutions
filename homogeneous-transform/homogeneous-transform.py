import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.array(points)

    is_single = (points.ndim == 1)   # store BEFORE reshape

    if is_single:
        points = points.reshape(1, -1)

    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    result_h = (T @ points_h.T).T

    result = result_h[:, :3]

    if is_single:
        return result[0]

    return result