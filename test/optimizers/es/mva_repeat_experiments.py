import time
import copy
import numpy as np
import rotated_shifted_functions as cf
from mva import MainVectorAdaptation
import matplotlib.pyplot as plt

functions = [cf.sphere, cf.schwefel12]
ndim_problem = 20
lower_boundary = -5 * np.ones((ndim_problem,)) # undefined in the original paper
upper_boundary = 5 * np.ones((ndim_problem,))

threshold_fitness = [1e-10, 1e-10]
d_sigma = [1, 10] # undefined in the original paper

for i, f in enumerate(functions):
    # set parameters of problem
    print("** set parameters of problem: {}".format(f.__name__))
    # in the original paper, the shift operation may be not used
    cf.generate_rotation_shift(f, ndim_problem, lower_boundary + 1, upper_boundary - 1)
    problem = {"ndim_problem": ndim_problem,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary}
    
    # set options of optimizer
    print("* set options of optimizer: {}".format("MVA-ES"))
    # a (1, 10) variant without recombination
    options = {"n_individuals": 10,
        "n_parents": 1,
        "max_evaluations": int(ndim_problem * 1e5),
        "threshold_fitness": threshold_fitness[i],
        "step_size": 0.5, # undefined in the original paper
        "d_sigma": d_sigma[i]}
    
    # solve
    start_time = time.time()
    figure = plt.figure()
    for t in range(70):
        solver = MainVectorAdaptation(problem, copy.deepcopy(options))
        results = solver.optimize(f)
        print("  iteration {}: best_so_far_y {:.2e}, n_evaluations {:d}".format(
            t + 1, results["best_so_far_y"], results["n_evaluations"]))
        
        fitness_data = results["fitness_data"]
        if t == 0:
            plt.plot(fitness_data[:, 0], fitness_data[:, 1], "k-",
                label="\u03BC=1, \u03BB=10, no recombination")
        else:
            plt.plot(fitness_data[:, 0], fitness_data[:, 1], "k-")
    
    end_time = time.time()
    print("$ total runtime: {:.2e}.".format(end_time - start_time))

    # plot fitness curve
    plt.title("f{:d} ({})".format(i + 1, f.__name__))
    plt.xlabel("function evaluations")
    plt.ylabel("fitness")
    if i == 0:
        plt.xticks(np.linspace(0, 6000, 7))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 1:
        plt.xticks(np.linspace(0, 2e4, 5))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    plt.legend(loc=1)
    plt.savefig(f.__name__ + ".png")
    # plt.show()
