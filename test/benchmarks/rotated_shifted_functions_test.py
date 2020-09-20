import numpy as np

from rotated_shifted_functions import *
from test_cases import BenchmarkTest, _load_shift_vector

# rotated shifted sphere
BenchmarkTest.check_origin(sphere,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted cigar
BenchmarkTest.check_origin(cigar,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted discus
BenchmarkTest.check_origin(discus,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted cigar_discus
BenchmarkTest.check_origin(cigar_discus,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted ellipsoid
BenchmarkTest.check_origin(ellipsoid,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted different_powers
BenchmarkTest.check_origin(different_powers,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted parabolic_ridge
BenchmarkTest.check_origin(parabolic_ridge,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted sharp_ridge
BenchmarkTest.check_origin(sharp_ridge,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted schwefel12
BenchmarkTest.check_origin(schwefel12,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted schwefel221
BenchmarkTest.check_origin(schwefel221,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted rosenbrock
for i in [2, 10, 100, 1000]:
    x = np.zeros((i,))
    generate_shift_vector(rosenbrock, x.size, -5, 5)
    generate_rotation_matrix(rosenbrock, i)
    print(rosenbrock(x + _load_shift_vector(rosenbrock, x)))

# rotated shifted griewank
BenchmarkTest.check_origin(griewank,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted ackley
BenchmarkTest.check_origin(ackley,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted rastrigin
BenchmarkTest.check_origin(rastrigin,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted scaled_rastrigin
BenchmarkTest.check_origin(scaled_rastrigin,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted skew_rastrigin
BenchmarkTest.check_origin(skew_rastrigin,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted schaffer
BenchmarkTest.check_origin(schaffer,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")

# rotated shifted bohachevsky
BenchmarkTest.check_origin(bohachevsky,
    start_from=2, end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")
