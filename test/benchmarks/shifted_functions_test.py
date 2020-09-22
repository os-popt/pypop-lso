from shifted_functions import *
from test_cases import BenchmarkTest, _load_shift_vector

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

# shifted parabolic_ridge
y = [[402, 101, 0, 99, 398],
    [802, 201, 0, 199, 798],
    [0, 299, 301, 299, 2899, 2899, 1404],
    [0, 399, 401, 399, 5399, 5399, 3005],
    [0, 499, 501, 499, 8999, 8999, 5506],
    [0, 599, 601, 599, 13899, 13899, 9107, 9100]]
BenchmarkTest.check_via_sampling(parabolic_ridge, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(parabolic_ridge, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted sharp_ridge
y = [[202, 101, 0, 99, 198],
    [284.8427, 142.4214, 0, 140.4214, 280.8427],
    [0, 172.2051, 174.2051, 172.2051, 537.5165, 537.5165, 378.1657],
    [0, 199.0000, 201.0000, 199.0000, 733.8469, 733.8469, 552.7226],
    [0, 222.6068, 224.6068, 222.6068, 947.6833, 947.6833, 747.6198],
    [0, 243.9490, 245.94897, 243.9490, 1177.9826, 1177.9826, 960.9392, 953.9392]]
BenchmarkTest.check_via_sampling(sharp_ridge, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(sharp_ridge, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted schwefel12
y = [[4, 1, 0, 5, 20],
    [8, 2, 0, 6, 24],
    [0, 30, 30, 2, 146, 10, 18],
    [0, 55, 55, 3, 371, 19, 55],
    [0, 91, 91, 7, 812, 28, 195],
    [0, 140, 140, 8, 1596, 44, 564, 812]]
BenchmarkTest.check_via_sampling(schwefel12, y, start_from=2, is_shifted=True)
# BenchmarkTest.check_origin(schwefel12, start_from=2, n_samples=1, is_shifted=True)
print("")

# shifted schwefel221
y = [[2, 1, 0, 1, 2],
    [2, 1, 0, 1, 2],
    [2, 1, 0, 1, 2],
    [0, 1, 1, 1, 4, 4, 4],
    [0, 1, 1, 1, 5, 5, 5],
    [0, 1, 1, 1, 6, 6, 6],
    [0, 1, 1, 1, 7, 7, 7, 6]]
BenchmarkTest.check_via_sampling(schwefel221, y, is_shifted=True)
BenchmarkTest.check_origin(schwefel221, n_samples=100, is_shifted=True)
print("")

# shifted rosenbrock
for i in [2, 10, 100, 1000]:
    x = np.zeros((i,)) - 1
    generate_shift_vector(rosenbrock, x.size, -5, 5)
    print(rosenbrock(x + _load_shift_vector(rosenbrock, x)))
# 1.0
# 9.0
# 99.0
# 999.0

# shifted griewank
y = [[1.066895, 0.589738, 0, 0.589738, 1.066895],
    [1.029230, 0.656567, 0, 0.656567, 1.029230],
    [0, 0.698951, 0.698951, 0.698951, 1.001870, 1.001870, 0.886208],
    [0, 0.728906, 0.728906, 0.728906, 1.017225, 1.017225, 0.992641],
    [0, 0.751538, 0.751538, 0.751538, 1.020074, 1.020074, 0.998490],
    [0, 0.769431, 0.769431, 0.769431, 1.037353, 1.037353, 1.054868, 1.024118]]
BenchmarkTest.check_via_sampling(griewank, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(griewank, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted ackley
y = [[6.593599, 3.625384, 0, 3.625384, 6.593599],
    [6.593599, 3.625384, 0, 3.625384, 6.593599],
    [0, 3.625384, 3.625384, 3.625384, 8.434694, 8.434694, 8.434694],
    [0, 3.625384, 3.625384, 3.625384, 9.697286, 9.697286, 9.697286],
    [0, 3.625384, 3.625384, 3.625384, 10.821680, 10.821680, 10.821680],
    [0, 3.625384, 3.625384, 3.625384, 11.823165, 11.823165, 11.823165, 10.275757]]
BenchmarkTest.check_via_sampling(ackley, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(ackley, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted rastrigin
y = [[8, 2, 0, 2, 8],
    [12, 3, 0, 3, 12],
    [0, 4, 4, 4, 30, 30, 30],
    [0, 5, 5, 5, 55, 55, 55],
    [0, 6, 6, 6, 91, 91, 91],
    [0, 7, 7, 7, 140, 140, 140, 91]]
BenchmarkTest.check_via_sampling(rastrigin, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(rastrigin, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted scaled_rastrigin
y = [[404, 101, 0, 101, 404],
    [458.5150, 115.7631, 0, 115.7631, 458.5150],
    [0, 147.8328, 147.8328, 147.8328, 1828.1772, 1828.1772, 275.7566],
    [0, 175.92186, 175.9219, 175.9219, 3168.9466, 3168.9466, 424.2752],
    [0, 217.7910, 217.7910, 217.7910, 4962.2668, 4962.2668, 621.1378],
    [0, 237.1110, 237.1110, 237.1110, 7367.6445, 7367.6445, 931.3332, 5289.1605]]
BenchmarkTest.check_via_sampling(scaled_rastrigin, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(scaled_rastrigin, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted skew_rastrigin
y = [[404, 101, 0, 200, 800],
    [804, 201, 0, 201, 804],
    [0, 400, 4, 202, 3000, 1020, 1317],
    [0, 500, 5, 302, 5500, 3520, 2926],
    [0, 600, 6, 402, 9100, 3556, 5437],
    [0, 700, 7, 403, 14000, 8456, 9050, 9100]]
BenchmarkTest.check_via_sampling(skew_rastrigin, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(skew_rastrigin, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted schaffer
y = [[3.2203, 1.2280, 0, 1.2280, 3.2203],
    [6.4407, 2.4560, 0, 2.4560, 6.4407],
    [0, 3.6840, 3.6840, 3.6840, 8.4804, 8.4804, 8.4804],
    [0, 4.9120, 4.9120, 4.9120, 11.1405, 11.1405, 11.1405],
    [0, 6.1400, 6.1400, 6.1400, 13.9369, 13.9369, 13.9369],
    [0, 7.3680, 7.3680, 7.3680, 17.8647, 17.8647, 17.8647, 15.0058]]
BenchmarkTest.check_via_sampling(schaffer, y, start_from=2, is_shifted=True)
BenchmarkTest.check_origin(schaffer, start_from=2, n_samples=100, is_shifted=True)
print("")

# shifted schwefel
y = [[420.9584, 419.8244, 418.9829, 418.1414, 417.0074],
    [837.9658, 837.9658, 837.9658, 836.2828, 834.0147],
    [1254.973130, 1256.107191, 1256.948662, 1256.107191, 1254.973130],
    [1675.931549, 1672.565665, 1679.297433, 1675.931549, 1666.516277, 1677.741720, 1675.473598],
    [2094.914436, 2090.707081, 2099.121791, 2094.072965, 2081.565418, 2092.790861, 2091.115851],
    [2513.897324, 2508.848498, 2518.946150, 2512.214382, 2496.719360, 2515.602694, 2506.060193],
    [2932.880211, 2926.989914, 2938.770508, 2932.038740, 2912.371844, 2931.255179, 2920.715592, 2915.702247]]
BenchmarkTest.check_via_sampling(schwefel, y, is_shifted=True)
print("")

# shifted bohachevsky
y = [[12, 3.6, 0, 3.6, 12],
    [24, 7.2, 0, 7.2, 24],
    [0, 10.8, 10.8, 10.8, 73.2, 73.2, 57.6],
    [0, 14.4, 14.4, 14.4, 139.2, 139.2, 115.2],
    [0, 18, 18, 18, 236.8, 236.8, 201.2],
    [0, 21.6,21.6, 21.6, 370.8, 370.8, 322.8, 238.8]]
BenchmarkTest.check_via_sampling(bohachevsky, y, start_from=2, is_shifted=True)
print("")
