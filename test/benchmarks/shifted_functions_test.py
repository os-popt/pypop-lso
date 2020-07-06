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

# shifted cigar_discus
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4040004, 1010001, 0, 1010001, 4040004],
    [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016],
    [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025],
    [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036],
    [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]]
BenchmarkTest.check_via_sampling(cigar_discus, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(cigar_discus, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted ellipsoid
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4004004, 1001001, 0, 1001001, 4004004],
    [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916],
    [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022],
    [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664],
    [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]]
BenchmarkTest.check_via_sampling(ellipsoid, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(ellipsoid, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted different_powers
y = [[68, 2, 0, 2, 68],
    [84, 3, 0, 3, 84],
    [0, 4, 4, 4, 4275.6, 4275.6, 81.3],
    [0, 5, 5, 5, 16739, 16739, 203],
    [0, 6, 6, 6, 51473.5, 51473.5, 437.1],
    [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]]
BenchmarkTest.check_via_sampling(different_powers, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(different_powers, start_from=2, n_samples=100, is_shifted=True)
print("")
