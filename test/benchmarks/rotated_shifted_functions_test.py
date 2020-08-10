from rotated_shifted_functions import *
from test_cases import BenchmarkTest

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
