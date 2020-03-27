import os
import numpy as np
from continuous_functions import _transform_and_check


# helper function
def _load_shift_vector(func, x, shift_vector=None):
    x = _transform_and_check(x)
    if shift_vector is None:
        if (not hasattr(func, 'shift_vector')) or (func.shift_vector.size != x.size):
            data_path = os.path.join("po_input_data", "shift_vector___" +
                func.__name__ + "_dim_" + str(x.size) + ".txt")
            shift_vector = _transform_and_check(np.loadtxt(data_path))
            func.shift_vector = shift_vector
        shift_vector = func.shift_vector
    else:
        shift_vector = _transform_and_check(shift_vector)
    if shift_vector.shape != x.shape:
        raise TypeError("shift_vector should have the same shape as x " +
            "after squeezing them via numpy.squeeze().")
    if shift_vector.size != x.size:
        raise TypeError("shift_vector should have the same size as x.")
    x = x - shift_vector
    return x


def generate_shift_vector(func_name, n_dim, low, high):
    data_folder = "po_input_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder,
        "shift_vector___" + func_name + "_dim_" + str(n_dim) + ".txt")
    shift_vector = np.random.uniform(low, high)
    np.savetxt(data_path, shift_vector)
    return shift_vector
