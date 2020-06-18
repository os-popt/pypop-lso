import time
from gym_continuous_control import grid_search_boundary
from prs import PureRandomSearch

start_time = time.time()
grid_search_boundary(PureRandomSearch, is_test=True)
print("$$$$$ total grid_search time: {:.2e}.".format(
    time.time() - start_time))
