from rotated_functions import *
from test_cases import BenchmarkTest

# rotated sphere
BenchmarkTest.check_origin(sphere,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated cigar
BenchmarkTest.check_origin(cigar,
    end_with=1000, n_samples=10, is_rotated=True)
print("")

# rotated discus
BenchmarkTest.check_origin(discus,
    end_with=1000, n_samples=10, is_rotated=True)
print("")
