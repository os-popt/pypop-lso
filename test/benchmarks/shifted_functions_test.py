from shifted_functions import *
from test_cases import BenchmarkTest

# shifted sphere
y = [[4, 1, 0, 1, 4],
    [8, 2, 0, 2, 8],
    [12, 3, 0, 3, 12],
    [0, 4, 4, 4, 30, 30, 30],
    [0, 5, 5, 5, 55, 55, 55],
    [0, 6, 6, 6, 91, 91, 91],
    [0, 7, 7, 7, 140, 140, 140, 91]]
BenchmarkTest.check_via_sampling(sphere, y, is_shifted=True)
BenchmarkTest.check_origin(sphere, n_samples=100, is_shifted=True)
print("")

# shifted cigar
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [8000004, 2000001, 0, 2000001, 8000004],
    [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016],
    [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025],
    [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036],
    [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]]
BenchmarkTest.check_via_sampling(cigar, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(cigar, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted discus
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4000008, 1000002, 0, 1000002, 4000008],
    [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014],
    [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030],
    [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055],
    [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]]
BenchmarkTest.check_via_sampling(discus, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(discus, start_from=2, n_samples=100, is_shifted=True)
print("")
