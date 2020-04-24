import numpy as np


# helper functions
def _transform_and_check(x, from_2_dim=False):
    """transform the input into the numpy.ndarray type, if necessary, and then
        check whether or not the number of dimensions <= 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if x.ndim > 1:
        raise TypeError("the number of dimensions should <= 1 " +
            "after squeezing the input via numpy.squeeze().")
    if from_2_dim and (x.size < 2):
        raise TypeError("the size should >= 2 after squeezing the input " +
            "via numpy.squeeze().")
    return x


# unimodal functions
def sphere(x):
    x = _transform_and_check(x)
    y = np.sum(np.power(x, 2))
    return y

def cigar(x):
    x = _transform_and_check(x, True)
    x = np.power(x, 2)
    y = x[0] + (10 ** 6) * np.sum(x[1:])
    return y

def discus(x):
    x = _transform_and_check(x, True)
    x = np.power(x, 2)
    y = (10 ** 6) * x[0] + np.sum(x[1:])
    return y

def ellipsoid(x):
    x = _transform_and_check(x, True)
    weights = np.power(10, 6 * np.linspace(0, 1, num=x.size))
    y = np.dot(weights, np.power(x, 2))
    return y

def parabolic_ridge(x):
    x = _transform_and_check(x, True)
    y = -x[0] + 100 * np.sum(np.power(x[1:], 2))
    return y

def sharp_ridge(x):
    x = _transform_and_check(x, True)
    y = -x[0] + 100 * np.sqrt(np.sum(np.power(x[1:], 2)))
    return y

def schwefel12(x):
    x = _transform_and_check(x, True)
    x = [np.sum(x[:i + 1]) for i in range(x.size)]
    y = np.sum(np.power(x, 2))
    return y

def rosenbrock(x):
    x = _transform_and_check(x, True)
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

def scaled_rastrigin(x):
    x = _transform_and_check(x, True)
    x = (10 ** np.linspace(0, 1, num=x.size)) * x
    y = rastrigin(x)
    return y

def skew_rastrigin(x):
    x = _transform_and_check(x)
    x[x > 0] = 10 * x[x > 0]
    y = rastrigin(x)
    return y

def schaffer(x):
    x = _transform_and_check(x, True)
    x = np.power(x, 2)
    x = x[:-1] + x[1:]
    y = np.sum(np.power(x, 0.25) * (np.power(np.sin(
        50 * np.power(x, 0.1)), 2) + 1.0))
    return y

def schwefel(x):
    x = _transform_and_check(x)
    y = 418.9828872724339 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return y

def bohachevsky(x):
    x = _transform_and_check(x, True)
    xx = np.power(x, 2)
    y = np.sum(xx[:-1] + 2 * xx[1:] - 0.3 * np.cos(3 * np.pi * x[:-1]) -
        0.4 * np.cos(4 * np.pi * x[1:]) + 0.7)
    return y
