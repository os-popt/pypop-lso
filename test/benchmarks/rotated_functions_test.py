from rotated_functions import *
from test_cases import BenchmarkTest

# sphere
BenchmarkTest.check_origin(sphere, 
    end_with=1000, n_samples=10, is_rotated=True)
print("")
