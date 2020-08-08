import time
import os
import numpy as np

from continuous_functions import _transform_and_check
from shifted_functions import generate_shift_vector
from rotated_functions import generate_rotation_matrix


class TestCases(object):
    def load(n_dim):
        if n_dim == 1:
            X = [[-2],
                [-1],
                [0],
                [1],
                [2]]
        elif n_dim == 2:
            X = [[-2, 2],
                [-1, 1],
                [0, 0],
                [1, 1],
                [2, 2]]
        elif n_dim == 3:
            X = [[-2, 2, 2],
                [-1, 1, 1],
                [0, 0, 0],
                [1, 1, -1],
                [2, 2, -2]]
        elif n_dim == 4:
            X = [[0, 0, 0, 0],
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
                [1, -1, 1, -1],
                [1, 2, 3, 4],
                [1, -2, 3, -4],
                [-4, 3, 2, -1]]
        elif n_dim == 5:
            X = [[0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1],
                [1, 2, 3, 4, 5],
                [1, -2, 3, -4, 5],
                [-5, 4, 3, 2, -1]]
        elif n_dim == 6:
            X = [[0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, 1],
                [1, 2, 3, 4, 5, 6],
                [1, -2, 3, -4, 5, -6],
                [-6, 5, 4, 3, 2, -1]]
        elif n_dim == 7:
            X = [[0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, 1, -1],
                [1, 2, 3, 4, 5, 6, 7],
                [1, -2, 3, -4, 5, -6, 7],
                [-7, 6, 5, 4, 3, 2, -1],
                [0, 1, 2, 3, 4, 5, 6]]
        else:
            raise TypeError("n_dim should >=1 and <= 7.")
        return np.array(X, dtype=np.float64)


# helper functions
def _load_shift_vector(func, x):
    data_path = os.path.join("po_input_data", "shift_vector___" +
        func.__name__ + "_dim_" + str(x.size) + ".txt")
    shift_vector = _transform_and_check(np.loadtxt(data_path))
    return shift_vector

def _search_tolerance(y1, y2, atols=None):
    if atols is None:
        atols = [(10 ** i) for i in range(-9, 2)]
    for atol in atols:
        is_pass = np.allclose(y1, y2, rtol=0, atol=atol)
        if is_pass:
            break
    return is_pass, atol


class BenchmarkTest(object):
    def check_via_sampling(func, y, start_from=1, end_with=7, is_shifted=False):
        start_time = time.time()
        if is_shifted:
            shift_sign = "shifted "
        else:
            shift_sign = ""
        print("check {}'{}' via sampling ->".format(shift_sign, func.__name__))
        for d in range(start_from, end_with + 1):
            X = TestCases.load(d)
            if is_shifted:
                generate_shift_vector(func, d, -10, 10)
            for s in range(X.shape[0]):
                if is_shifted:
                    X[s, :] = X[s, :] + _load_shift_vector(func, X[s, :])
                is_pass, atol = _search_tolerance(func(X[s, :]), y[d - start_from][s])
                if not(is_pass):
                    message = " NOT "
                    break
            else:
                message = " "
            print("    -> can{}pass the {}-dimensional check with tolerance {:.2e}".format(
                message, d, atol))
        end_time = time.time()
        print("    : take {:.2f} seconds.".format(end_time - start_time))

    def check_origin(func, y=0, atols=[1e-9], start_from=1, end_with=10000,
        n_samples=10000, is_shifted=False, is_rotated=False):
        start_time = time.time()
        for s in range(n_samples):
            x = np.zeros(np.random.randint(start_from, end_with + 1))
            if is_shifted:
                generate_shift_vector(func, x.size, -5, 5)
                x = x + _load_shift_vector(func, x)
            if is_rotated:
                generate_rotation_matrix(func, x.size)
            is_pass, atol = _search_tolerance(func(x), y, atols)
            if not(is_pass):
                message = " NOT "
                break
        else:
            message = " "
        end_time = time.time()
        message_sign = ""
        if is_rotated:
            message_sign += "Rotated "
        if is_shifted:
            message_sign += "Shifted "
        print("{}'{}' has{}passed the origin check with tolerance {:.2e}: take {:.2f} seconds.".format(
            message_sign, func.__name__, message, atol, end_time - start_time))
