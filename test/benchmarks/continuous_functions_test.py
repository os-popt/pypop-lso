from continuous_functions import *
from test_cases import BenchmarkTest

# sphere
y = [[4, 1, 0, 1, 4],
    [8, 2, 0, 2, 8],
    [12, 3, 0, 3, 12],
    [0, 4, 4, 4, 30, 30, 30],
    [0, 5, 5, 5, 55, 55, 55],
    [0, 6, 6, 6, 91, 91, 91],
    [0, 7, 7, 7, 140, 140, 140, 91]]
BenchmarkTest.check_via_sampling(sphere, y)
BenchmarkTest.check_origin(sphere)
print("")

# cigar
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [8000004, 2000001, 0, 2000001, 8000004],
    [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016],
    [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025],
    [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036],
    [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]]
BenchmarkTest.check_via_sampling(cigar, y, start_from=2)
BenchmarkTest.check_origin(cigar, start_from=2)
print("")

# discus
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4000008, 1000002, 0, 1000002, 4000008],
    [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014],
    [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030],
    [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055],
    [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]]
BenchmarkTest.check_via_sampling(discus, y, start_from=2)
BenchmarkTest.check_origin(discus, start_from=2)
print("")

# cigar_discus
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4040004, 1010001, 0, 1010001, 4040004],
    [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016],
    [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025],
    [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036],
    [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]]
BenchmarkTest.check_via_sampling(cigar_discus, y, start_from=2)
BenchmarkTest.check_origin(cigar_discus, start_from=2)
print("")

# ellipsoid
y = [[4000004, 1000001, 0, 1000001, 4000004],
    [4004004, 1001001, 0, 1001001, 4004004],
    [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916],
    [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022],
    [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664],
    [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]]
BenchmarkTest.check_via_sampling(ellipsoid, y, start_from=2)
BenchmarkTest.check_origin(ellipsoid, start_from=2)
print("")

# different_powers
y = [[68, 2, 0, 2, 68],
    [84, 3, 0, 3, 84],
    [0, 4, 4, 4, 4275.6, 4275.6, 81.3],
    [0, 5, 5, 5, 16739, 16739, 203],
    [0, 6, 6, 6, 51473.5, 51473.5, 437.1],
    [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]]
BenchmarkTest.check_via_sampling(different_powers, y, start_from=2)
BenchmarkTest.check_origin(different_powers, start_from=2)
print("")

# parabolic_ridge
y = [[402, 101, 0, 99, 398],
    [802, 201, 0, 199, 798],
    [0, 299, 301, 299, 2899, 2899, 1404],
    [0, 399, 401, 399, 5399, 5399, 3005],
    [0, 499, 501, 499, 8999, 8999, 5506],
    [0, 599, 601, 599, 13899, 13899, 9107, 9100]]
BenchmarkTest.check_via_sampling(parabolic_ridge, y, start_from=2)
BenchmarkTest.check_origin(parabolic_ridge, start_from=2)
print("")

# schwefel12
y = [[4, 1, 0, 5, 20],
    [8, 2, 0, 6, 24],
    [0, 30, 30, 2, 146, 10, 18],
    [0, 55, 55, 3, 371, 19, 55],
    [0, 91, 91, 7, 812, 28, 195],
    [0, 140, 140, 8, 1596, 44, 564, 812]]
BenchmarkTest.check_via_sampling(schwefel12, y, start_from=2)
# BenchmarkTest.check_origin(schwefel12, start_from=2) # time-consuming
print("")

# schwefel221
y = [[2, 1, 0, 1, 2],
    [2, 1, 0, 1, 2],
    [2, 1, 0, 1, 2],
    [0, 1, 1, 1, 4, 4, 4],
    [0, 1, 1, 1, 5, 5, 5],
    [0, 1, 1, 1, 6, 6, 6],
    [0, 1, 1, 1, 7, 7, 7, 6]]
BenchmarkTest.check_via_sampling(schwefel221, y)
BenchmarkTest.check_origin(schwefel221)
print("")

# rosenbrock
y = [[409, 4, 1, 0, 401],
    [810, 4, 2, 400, 4002],
    [3, 0, 1212, 804, 2705, 17913, 24330],
    [4, 0, 1616, 808, 14814, 30038, 68450],
    [5, 0, 2020, 808, 50930, 126154, 164579],
    [6, 0, 2424, 1208, 135055, 210303, 349519, 51031]]
BenchmarkTest.check_via_sampling(rosenbrock, y, start_from=2)
print("")

# griewank
y = [[1.066895, 0.589738, 0, 0.589738, 1.066895],
    [1.029230, 0.656567, 0, 0.656567, 1.029230],
    [0, 0.698951, 0.698951, 0.698951, 1.001870, 1.001870, 0.886208],
    [0, 0.728906, 0.728906, 0.728906, 1.017225, 1.017225, 0.992641],
    [0, 0.751538, 0.751538, 0.751538, 1.020074, 1.020074, 0.998490],
    [0, 0.769431, 0.769431, 0.769431, 1.037353, 1.037353, 1.054868, 1.024118]]
BenchmarkTest.check_via_sampling(griewank, y, start_from=2)
BenchmarkTest.check_origin(griewank, start_from=2)
print("")

# ackley
y = [[6.593599, 3.625384, 0, 3.625384, 6.593599],
    [6.593599, 3.625384, 0, 3.625384, 6.593599],
    [0, 3.625384, 3.625384, 3.625384, 8.434694, 8.434694, 8.434694],
    [0, 3.625384, 3.625384, 3.625384, 9.697286, 9.697286, 9.697286],
    [0, 3.625384, 3.625384, 3.625384, 10.821680, 10.821680, 10.821680],
    [0, 3.625384, 3.625384, 3.625384, 11.823165, 11.823165, 11.823165, 10.275757]]
BenchmarkTest.check_via_sampling(ackley, y, start_from=2)
BenchmarkTest.check_origin(ackley, start_from=2)
print("")

# rastrigin
y = [[8, 2, 0, 2, 8],
    [12, 3, 0, 3, 12],
    [0, 4, 4, 4, 30, 30, 30],
    [0, 5, 5, 5, 55, 55, 55],
    [0, 6, 6, 6, 91, 91, 91],
    [0, 7, 7, 7, 140, 140, 140, 91]]
BenchmarkTest.check_via_sampling(rastrigin, y, start_from=2)
BenchmarkTest.check_origin(rastrigin, start_from=2)
print("")
