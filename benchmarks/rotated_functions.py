import os
import numpy as np
import continuous_functions
from continuous_functions import _transform_and_check


# helper functions
def _load_rotation_matrix(func, x, rotation_matrix=None):
    x = _transform_and_check(x)
    if rotation_matrix is None:
        if (not hasattr(func, "rotation_matrix")) or (func.rotation_matrix.shape != (x.size, x.size)):
            data_path = os.path.join("po_input_data", "rotation_matrix___" +
                func.__name__ + "_dim_" + str(x.size) + ".txt")
            rotation_matrix = np.loadtxt(data_path)
            func.rotation_matrix = rotation_matrix
        rotation_matrix = func.rotation_matrix
    else:
        rotation_matrix = np.squeeze(rotation_matrix)
    if rotation_matrix.shape != (x.size, x.size):
        raise TypeError("rotation_matrix should have the same size as x " +
            "after squeezing it via numpy.squeeze().")
    if rotation_matrix.size != x.size * x.size:
        raise TypeError("rotation_matrix should have the size: " +
            str(x.size) + " * " + str(x.size) + " .")
    x = np.dot(rotation_matrix, x)
    return x


def generate_rotation_matrix(func, n_dim):
    if hasattr(func, '__call__'):
        func = func.__name__
    data_folder = "po_input_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder,
        "rotation_matrix___" + func + "_dim_" + str(n_dim) + ".txt")
    rotation_matrix = np.random.randn(n_dim, n_dim)
    for i in range(n_dim):
        for j in range(i):
            rotation_matrix[:, i] = rotation_matrix[:, i] - np.dot(rotation_matrix[:, i].transpose(),
                rotation_matrix[:, j]) * rotation_matrix[:, j]
        rotation_matrix[:, i] = rotation_matrix[:, i] / np.linalg.norm(rotation_matrix[:, i])
    np.savetxt(data_path, rotation_matrix)
    return rotation_matrix
