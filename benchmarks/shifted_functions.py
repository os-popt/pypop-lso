import os
import numpy as np
import continuous_functions
from continuous_functions import _transform_and_check


# helper functions
def _load_shift_vector(func, x, shift_vector=None):
    x = _transform_and_check(x)
    if shift_vector is None:
        if (not hasattr(func, "shift_vector")) or (func.shift_vector.size != x.size):
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


def generate_shift_vector(func, n_dim, low, high):
    low, high = _transform_and_check(low), _transform_and_check(high)
    if low.size == 1:
        low = np.tile(low, n_dim)
    if high.size == 1:
        high = np.tile(high, n_dim)
    if low.size != n_dim:
        raise TypeError("low'size should equal n_dim.")
    if high.size != n_dim:
        raise TypeError("high'size should equal n_dim.")
    if hasattr(func, "__call__"):
        func = func.__name__
    data_folder = "po_input_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder,
        "shift_vector___" + func + "_dim_" + str(n_dim) + ".txt")
    shift_vector = np.random.uniform(low, high)
    np.savetxt(data_path, shift_vector)
    return shift_vector


# unimodal functions
def sphere(x, shift_vector=None):
    x = _load_shift_vector(sphere, x, shift_vector)
    y = continuous_functions.sphere(x)
    return y

def cigar(x, shift_vector=None):
    x = _load_shift_vector(cigar, x, shift_vector)
    y = continuous_functions.cigar(x)
    return y

def discus(x, shift_vector=None):
    x = _load_shift_vector(discus, x, shift_vector)
    y = continuous_functions.discus(x)
    return y

def cigar_discus(x, shift_vector=None):
    x = _load_shift_vector(cigar_discus, x, shift_vector)
    y = continuous_functions.cigar_discus(x)
    return y

def ellipsoid(x, shift_vector=None):
    x = _load_shift_vector(ellipsoid, x, shift_vector)
    y = continuous_functions.ellipsoid(x)
    return y

def different_powers(x, shift_vector=None):
    x = _load_shift_vector(different_powers, x, shift_vector)
    y = continuous_functions.different_powers(x)
    return y

def parabolic_ridge(x, shift_vector=None):
    x = _load_shift_vector(parabolic_ridge, x, shift_vector)
    y = continuous_functions.parabolic_ridge(x)
    return y

def sharp_ridge(x, shift_vector=None):
    x = _load_shift_vector(sharp_ridge, x, shift_vector)
    y = continuous_functions.sharp_ridge(x)
    return y

def schwefel12(x, shift_vector=None):
    x = _load_shift_vector(schwefel12, x, shift_vector)
    y = continuous_functions.schwefel12(x)
    return y

def schwefel221(x, shift_vector=None):
    x = _load_shift_vector(schwefel221, x, shift_vector)
    y = continuous_functions.schwefel221(x)
    return y

def rosenbrock(x, shift_vector=None):
    x = _load_shift_vector(rosenbrock, x, shift_vector)
    y = continuous_functions.rosenbrock(x + 1)
    return y

# multi-modal functions
def griewank(x, shift_vector=None):
    x = _load_shift_vector(griewank, x, shift_vector)
    y = continuous_functions.griewank(x)
    return y

def ackley(x, shift_vector=None):
    x = _load_shift_vector(ackley, x, shift_vector)
    y = continuous_functions.ackley(x)
    return y

def rastrigin(x, shift_vector=None):
    x = _load_shift_vector(rastrigin, x, shift_vector)
    y = continuous_functions.rastrigin(x)
    return y

def scaled_rastrigin(x, shift_vector=None):
    x = _load_shift_vector(scaled_rastrigin, x, shift_vector)
    y = continuous_functions.scaled_rastrigin(x)
    return y

def skew_rastrigin(x, shift_vector=None):
    x = _load_shift_vector(skew_rastrigin, x, shift_vector)
    y = continuous_functions.skew_rastrigin(x)
    return y

def schaffer(x, shift_vector=None):
    x = _load_shift_vector(schaffer, x, shift_vector)
    y = continuous_functions.schaffer(x)
    return y

def schwefel(x, shift_vector=None):
    x = _load_shift_vector(schwefel, x, shift_vector)
    y = continuous_functions.schwefel(x)
    return y

def bohachevsky(x, shift_vector=None):
    x = _load_shift_vector(bohachevsky, x, shift_vector)
    y = continuous_functions.bohachevsky(x)
    return y
