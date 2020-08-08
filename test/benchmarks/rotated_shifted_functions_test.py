from rotated_shifted_functions import *
from test_cases import BenchmarkTest

# rotated shifted sphere
BenchmarkTest.check_origin(sphere,
    end_with=1000, n_samples=10, is_shifted=True, is_rotated=True)
print("")
