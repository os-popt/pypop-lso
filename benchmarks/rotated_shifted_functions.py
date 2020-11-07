import os
import numpy as np
import continuous_functions
from shifted_functions import _load_shift_vector, generate_shift_vector
from rotated_functions import _load_rotation_matrix, generate_rotation_matrix


# helper functions
def _load_rotation_and_shift(func, x):
    x = _load_shift_vector(func, x)
    x = _load_rotation_matrix(func, x)
    return x


def generate_rotation_shift(func, n_dim, low, high):
    shift_vector = generate_shift_vector(func, n_dim, low, high)
    rotation_matrix = generate_rotation_matrix(func, n_dim)
    return (rotation_matrix, shift_vector)


# unimodal functions
def sphere(x):
    x = _load_rotation_and_shift(sphere, x)
    y = continuous_functions.sphere(x)
    return y

def cigar(x):
    x = _load_rotation_and_shift(cigar, x)
    y = continuous_functions.cigar(x)
    return y

def discus(x):
    x = _load_rotation_and_shift(discus, x)
    y = continuous_functions.discus(x)
    return y

def cigar_discus(x):
    x = _load_rotation_and_shift(cigar_discus, x)
    y = continuous_functions.cigar_discus(x)
    return y

def ellipsoid(x):
    x = _load_rotation_and_shift(ellipsoid, x)
    y = continuous_functions.ellipsoid(x)
    return y

def different_powers(x):
    x = _load_rotation_and_shift(different_powers, x)
    y = continuous_functions.different_powers(x)
    return y

def different_powers_beyer(x):
    x = _load_rotation_and_shift(different_powers_beyer, x)
    y = continuous_functions.different_powers_beyer(x)
    return y

def parabolic_ridge(x):
    x = _load_rotation_and_shift(parabolic_ridge, x)
    y = continuous_functions.parabolic_ridge(x)
    return y

def sharp_ridge(x):
    x = _load_rotation_and_shift(sharp_ridge, x)
    y = continuous_functions.sharp_ridge(x)
    return y

def schwefel12(x):
    x = _load_rotation_and_shift(schwefel12, x)
    y = continuous_functions.schwefel12(x)
    return y

def schwefel221(x):
    x = _load_rotation_and_shift(schwefel221, x)
    y = continuous_functions.schwefel221(x)
    return y

def rosenbrock(x):
    x = _load_rotation_and_shift(rosenbrock, np.array(x) + 1)
    y = continuous_functions.rosenbrock(x)
    return y

# multi-modal functions
def griewank(x):
    x = _load_rotation_and_shift(griewank, x)
    y = continuous_functions.griewank(x)
    return y

def ackley(x):
    x = _load_rotation_and_shift(ackley, x)
    y = continuous_functions.ackley(x)
    return y

def rastrigin(x):
    x = _load_rotation_and_shift(rastrigin, x)
    y = continuous_functions.rastrigin(x)
    return y

def scaled_rastrigin(x):
    x = _load_rotation_and_shift(scaled_rastrigin, x)
    y = continuous_functions.scaled_rastrigin(x)
    return y

def skew_rastrigin(x):
    x = _load_rotation_and_shift(skew_rastrigin, x)
    y = continuous_functions.skew_rastrigin(x)
    return y

def schaffer(x):
    x = _load_rotation_and_shift(schaffer, x)
    y = continuous_functions.schaffer(x)
    return y

def schwefel(x):
    x = _load_rotation_and_shift(schwefel, x)
    y = continuous_functions.schwefel(x)
    return y

def bohachevsky(x):
    x = _load_rotation_and_shift(bohachevsky, x)
    y = continuous_functions.bohachevsky(x)
    return y
