import time
from gym_continuous_control import grid_search_boundary
from rechenberg import Rechenberg

start_time = time.time()
grid_search_boundary(Rechenberg)
print("$$$$$ total train time: {:.2e}.".format(
    time.time() - start_time))

start_time = time.time()
grid_search_boundary(Rechenberg, is_test=True)
print("$$$$$ total test time: {:.2e}.".format(
    time.time() - start_time))
