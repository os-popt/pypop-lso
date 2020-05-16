import time
import numpy as np


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
        return np.array(X)


def _search_tolerance(y1, y2, atols=None):
    if atols is None:
        atols = [10 ** i for i in range(-9, 2)]
    is_pass = False
    for atol in atols:
        is_pass = np.allclose(y1, y2, rtol=0, atol=atol)
        if is_pass:
            break
    return is_pass, atol


class BenchmarkTest(object):
    def check_via_sampling(func, y, start_from=1, end_with=7):
        start_time = time.time()
        print("check '{}' via sampling ->".format(func.__name__))
        for d in range(start_from, end_with + 1):
            message = " "
            X = TestCases.load(d)
            for s in range(X.shape[0]):
                is_pass, atol = _search_tolerance(func(X[s, :]), y[d - start_from][s])
                if not(is_pass):
                    message = " NOT "
                    break
            print("    -> can{}pass the {}-dimensional check with tolerance {:.2e}".format(
                message, d, atol))
        end_time = time.time()
        print("    : take {:.2f} seconds.".format(end_time - start_time))

    def check_origin(func, y=0, atols=[1e-9], start_from=1, end_with=10000, n_samples=10000):
        start_time = time.time()
        is_pass, message = True, " "
        for s in range(n_samples):
            x = np.zeros(np.random.randint(start_from, end_with + 1))
            is_pass, atol = _search_tolerance(func(x), y, atols)
            if not(is_pass):
                message = " NOT "
                break
        end_time = time.time()
        print("'{}' has{}passed the origin check with tolerance {:.2e}: take {:.2f} seconds.".format(
            func.__name__, message, atol, end_time - start_time))
