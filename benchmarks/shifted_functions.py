import os
import numpy as np


def generate_shift_vector(func_name, n_dim, low, high):
    data_folder = 'po_input_data'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder,
        "shift_vector___" + func_name + "_dim_" + str(n_dim) + ".txt")
    shift_vector = np.random.uniform(low, high)
    np.savetxt(data_path, shift_vector)
    return shift_vector
