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


class BenchmarkTest(object):
    def check_via_sampling(func, y, rtol=1e-09, atol=1e-09, start_from=1, end_with=7):
        is_pass = True
        for d in range(start_from, end_with + 1):
            X = TestCases.load(d)
            for s in range(X.shape[0]):
                is_pass = np.allclose(func(X[s, :]), y[d][s], rtol, atol)
                if not(is_pass):
                    break
            if not(is_pass):
                break
        return is_pass, d, s
