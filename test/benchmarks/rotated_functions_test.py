from rotated_functions import *
from test_cases import BenchmarkTest

# rotated sphere
BenchmarkTest.check_origin(sphere,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated cigar
BenchmarkTest.check_origin(cigar,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated discus
BenchmarkTest.check_origin(discus,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated cigar_discus
BenchmarkTest.check_origin(cigar_discus,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated ellipsoid
BenchmarkTest.check_origin(ellipsoid,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated different_powers
BenchmarkTest.check_origin(different_powers,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated parabolic_ridge
BenchmarkTest.check_origin(parabolic_ridge,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated sharp_ridge
BenchmarkTest.check_origin(sharp_ridge,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated schwefel12
BenchmarkTest.check_origin(schwefel12,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated schwefel221
BenchmarkTest.check_origin(schwefel221,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated griewank
BenchmarkTest.check_origin(griewank,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated ackley
BenchmarkTest.check_origin(ackley,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated rastrigin
BenchmarkTest.check_origin(rastrigin,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated scaled_rastrigin
BenchmarkTest.check_origin(scaled_rastrigin,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated skew_rastrigin
BenchmarkTest.check_origin(skew_rastrigin,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated schaffer
BenchmarkTest.check_origin(schaffer,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated bohachevsky
BenchmarkTest.check_origin(bohachevsky,
    start_from=2, end_with=1000, n_samples=10, is_rotated=True)
print("")
