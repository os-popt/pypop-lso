import numpy as np


# helper function
def _transform_and_check(x):
    """transform the input into the numpy.ndarray type, if necessary, and then
        check whether or not the number of dimensions is 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("the number of dimensions should equal 1.")
    return x


def sphere(x):
    x = _transform_and_check(x)
    y = np.sum(np.power(x, 2))
    return y
