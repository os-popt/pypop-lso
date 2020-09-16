import time
from gym_continuous_control import grid_search_boundary
from rrm import RestartRm

start_time = time.time()
grid_search_boundary(RestartRm)
print("$$$$$ total train time: {:.2e}.".format(
    time.time() - start_time))

start_time = time.time()
grid_search_boundary(RestartRm, is_test=True)
print("$$$$$ total test time: {:.2e}.".format(
    time.time() - start_time))
