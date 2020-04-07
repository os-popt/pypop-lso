import numpy as np


# helper functions
def _transform_and_check(x):
    """transform the input into the numpy.ndarray type, if necessary, and then
        check whether or not the number of dimensions <= 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if x.ndim > 1:
        raise TypeError("the number of dimensions should <= 1 " +
            "after squeezing the input via numpy.squeeze().")
    return x

def _transform_and_check_size(x):
    """Check whether or not the size >= 2. If not, raise a TypeError.
    """
    x = _transform_and_check(x)
    if x.size < 2:
        raise TypeError("the size should >= 2 after squeezing the input " +
            "via numpy.squeeze().")
    return x


def sphere(x):
    x = _transform_and_check(x)
    y = np.sum(np.power(x, 2))
    return y

def cigar(x):
    x = _transform_and_check_size(x)
    x = np.power(x, 2)
    y = x[0] + (10 ** 6) * np.sum(x[1:])
    return y

def discus(x):
    x = _transform_and_check_size(x)
    x = np.power(x, 2)
    y = (10 ** 6) * x[0] + np.sum(x[1:])
    return y

def ellipsoid(x):
    x = _transform_and_check_size(x)
    weights = np.power(10, 6 * np.linspace(0, 1, num=x.size))
    y = np.dot(weights, np.power(x, 2))
    return y

def parabolic_ridge(x):
    x = _transform_and_check_size(x)
    y = -x[0] + 100 * np.sum(np.power(x[1:], 2))
    return y

def sharp_ridge(x):
    x = _transform_and_check_size(x)
    y = -x[0] + 100 * np.sqrt(np.sum(np.power(x[1:], 2)))
    return y

def rosenbrock(x):
    x = _transform_and_check_size(x)
    y = 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + \
        np.sum(np.power(x[:-1] - 1, 2))
    return y

# multi-modal functions
def griewank(x):
    x = _transform_and_check(x)
    y = np.sum(np.power(x, 2)) / 4000 - np.prod(np.cos(
        x / np.sqrt(np.arange(1, x.size+1)))) + 1
    return y

def ackley(x):
    x = _transform_and_check(x)
    y = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.power(x, 2)) / x.size)) - \
        np.exp(np.sum(np.cos(2 * np.pi * x)) / x.size) + \
        20 + np.exp(1)
    return y

def rastrigin(x):
    x = _transform_and_check(x)
    y = 10 * x.size + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))
    return y
