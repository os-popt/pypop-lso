import os
import numpy as np
import continuous_functions
from continuous_functions import _transform_and_check


# helper functions
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
