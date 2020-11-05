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
    if rotation_matrix.ndim == 0:
        rotation_matrix = np.reshape(rotation_matrix, [1, 1])
    if rotation_matrix.shape != (x.size, x.size):
        raise TypeError("rotation_matrix should have the same size as x " +
            "after squeezing it via numpy.squeeze().")
    if rotation_matrix.size != x.size * x.size:
        raise TypeError("rotation_matrix should have the size: " +
            str(x.size) + " * " + str(x.size) + " .")
    x = np.dot(rotation_matrix, x)
    return x


def generate_rotation_matrix(func, n_dim):
    if hasattr(func, "__call__"):
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


def check_rotation_matrix(x, error=1e-6):
    x = np.mat(x)
    if x.shape[0] != x.shape[1]:
        raise TypeError("the input 'x' should be a square matrix.")
    conditionA = (np.abs(np.matmul(x, x.transpose()) - np.eye(x.shape[0])) < error)
    conditionB = (np.abs(np.matmul(x.transpose(), x) - np.eye(x.shape[0])) < error)
    conditionC = (np.abs(np.sum(np.power(x, 2), 0) - 1) < error)
    conditionD = (np.abs(np.sum(np.power(x, 2), 1) - 1) < error)
    conditionE = (np.linalg.matrix_rank(x) == x.shape[0])
    return (np.all(conditionA) and np.all(conditionB) and
        np.all(conditionC) and np.all(conditionD) and
        conditionE)


# unimodal functions
def sphere(x, rotation_matrix=None):
    x = _load_rotation_matrix(sphere, x, rotation_matrix)
    y = continuous_functions.sphere(x)
    return y

def cigar(x, rotation_matrix=None):
    x = _load_rotation_matrix(cigar, x, rotation_matrix)
    y = continuous_functions.cigar(x)
    return y

def discus(x, rotation_matrix=None):
    x = _load_rotation_matrix(discus, x, rotation_matrix)
    y = continuous_functions.discus(x)
    return y

def cigar_discus(x, rotation_matrix=None):
    x = _load_rotation_matrix(cigar_discus, x, rotation_matrix)
    y = continuous_functions.cigar_discus(x)
    return y

def ellipsoid(x, rotation_matrix=None):
    x = _load_rotation_matrix(ellipsoid, x, rotation_matrix)
    y = continuous_functions.ellipsoid(x)
    return y

def different_powers(x, rotation_matrix=None):
    x = _load_rotation_matrix(different_powers, x, rotation_matrix)
    y = continuous_functions.different_powers(x)
    return y

def different_powers_beyer(x, rotation_matrix=None):
    x = _load_rotation_matrix(different_powers_beyer, x, rotation_matrix)
    y = continuous_functions.different_powers_beyer(x)
    return y

def parabolic_ridge(x, rotation_matrix=None):
    x = _load_rotation_matrix(parabolic_ridge, x, rotation_matrix)
    y = continuous_functions.parabolic_ridge(x)
    return y

def sharp_ridge(x, rotation_matrix=None):
    x = _load_rotation_matrix(sharp_ridge, x, rotation_matrix)
    y = continuous_functions.sharp_ridge(x)
    return y

def schwefel12(x, rotation_matrix=None):
    x = _load_rotation_matrix(schwefel12, x, rotation_matrix)
    y = continuous_functions.schwefel12(x)
    return y

def schwefel221(x, rotation_matrix=None):
    x = _load_rotation_matrix(schwefel221, x, rotation_matrix)
    y = continuous_functions.schwefel221(x)
    return y

def rosenbrock(x, rotation_matrix=None):
    x = _load_rotation_matrix(rosenbrock, x, rotation_matrix)
    y = continuous_functions.rosenbrock(x)
    return y

# multi-modal functions
def griewank(x, rotation_matrix=None):
    x = _load_rotation_matrix(griewank, x, rotation_matrix)
    y = continuous_functions.griewank(x)
    return y

def ackley(x, rotation_matrix=None):
    x = _load_rotation_matrix(ackley, x, rotation_matrix)
    y = continuous_functions.ackley(x)
    return y

def rastrigin(x, rotation_matrix=None):
    x = _load_rotation_matrix(rastrigin, x, rotation_matrix)
    y = continuous_functions.rastrigin(x)
    return y

def scaled_rastrigin(x, rotation_matrix=None):
    x = _load_rotation_matrix(scaled_rastrigin, x, rotation_matrix)
    y = continuous_functions.scaled_rastrigin(x)
    return y

def skew_rastrigin(x, rotation_matrix=None):
    x = _load_rotation_matrix(skew_rastrigin, x, rotation_matrix)
    y = continuous_functions.skew_rastrigin(x)
    return y

def schaffer(x, rotation_matrix=None):
    x = _load_rotation_matrix(schaffer, x, rotation_matrix)
    y = continuous_functions.schaffer(x)
    return y

def schwefel(x, rotation_matrix=None):
    x = _load_rotation_matrix(schwefel, x, rotation_matrix)
    y = continuous_functions.schwefel(x)
    return y

def bohachevsky(x, rotation_matrix=None):
    x = _load_rotation_matrix(bohachevsky, x, rotation_matrix)
    y = continuous_functions.bohachevsky(x)
    return y
