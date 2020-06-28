import time
import copy
import numpy as np

import continuous_functions as cf
from r1 import RankOne as R1ES


message = "  iteration {}: best_so_far_y {:.2e}, n_evaluations {:d}, step_size: {:.2e}, runtime: {:.2e}"

functions = [cf.ellipsoid, cf.discus, cf.rosenbrock]
ndim_problem = 1000
for f in functions:
    # set parameters of problem
    print("** set parameters of problem: {}".format(f.__name__))
    problem = {"ndim_problem": ndim_problem,
        "lower_boundary": -10 * np.ones((ndim_problem,)),
        "upper_boundary": 10 * np.ones((ndim_problem,))}
    
    # set options of optimizer
    print("* set options of optimizer: {}".format("R1-ES"))
    options = {"max_evaluations": int(1e8),
        "threshold_fitness": 1e-8,
        "step_size": 20 / 3}
    
    # solve
    start_time = time.time()
    for t in range(31):
        options_t = copy.deepcopy(options)
        options_t["seed"] = t
        solver = R1ES(problem, options_t)
        results = solver.optimize(f)
        print(message.format(t + 1,
            results["best_so_far_y"], results["n_evaluations"],
            results["step_size"], results["runtime"]))
        np.savetxt("{}_{:02d}_fitness_data.txt".format(
            f.__name__, t + 1), results["fitness_data"])
    print("$ total runtime: {:.2e}.".format(time.time() - start_time))
