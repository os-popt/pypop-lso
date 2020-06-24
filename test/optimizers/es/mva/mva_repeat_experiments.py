import time
import copy
import numpy as np
import rotated_shifted_functions as cf
from mva import MainVectorAdaptation as MVA
import matplotlib.pyplot as plt

# schwefel12 -> Schwefel, discus -> Tablet
functions = [cf.sphere, cf.schwefel12, cf.cigar, cf.discus, cf.ellipsoid,
    cf.parabolic_ridge, cf.sharp_ridge, cf.rosenbrock, cf.different_powers]
ndim_problem = 20
lower_boundary = -5 * np.ones((ndim_problem,)) # undefined in the original paper
upper_boundary = 5 * np.ones((ndim_problem,))

max_evaluations = [6000, 2e4, 5e4, 3.5e5, 3.5e5, 6000, 3.5e4, 1e5, 1e5]
max_ylim = [1e4, 1e4, 1e4, 1e5, 1e4, 400, 400, 1e4, 1]
threshold_fitness = [1e-10, 1e-10, 1e-10, 1e-10, 1e-10, -800, -800, 1e-10, 1e-10]
d_sigma = [1, 10, 1, 10, 10, 2, 50, 1, 1] # undefined in the original paper

for i, f in enumerate(functions):
    # set parameters of problem
    print("** set parameters of problem: {}".format(f.__name__))
    # in the original paper, the shift operation may be not used
    np.random.seed(20200620 + i) # for repeatability
    cf.generate_rotation_shift(f, ndim_problem, lower_boundary + 1, upper_boundary - 1)
    problem = {"ndim_problem": ndim_problem,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary}
    
    # set options of optimizer
    print("* set options of optimizer: {}".format("MVA-ES"))
    # a (1, 10) variant without recombination
    options = {"n_individuals": 10,
        "n_parents": 1,
        "max_evaluations": int(max_evaluations[i]),
        "threshold_fitness": threshold_fitness[i],
        "step_size": 0.3, # undefined in the original paper
        "d_sigma": d_sigma[i]}
    
    # solve
    start_time = time.time()
    figure = plt.figure()
    for t in range(70):
        options_t = copy.deepcopy(options)
        options_t["seed"] = t
        solver = MVA(problem, options_t)
        results = solver.optimize(f)
        message = "  iteration {}: best_so_far_y {:.2e}, n_evaluations {:d}, " +\
            "step_size: {:.2e}, runtime: {:.2e}"
        print(message.format(t + 1, results["best_so_far_y"], results["n_evaluations"],
            results["step_size"], results["runtime"]))
        
        fitness_data = results["fitness_data"]
        fitness_data[fitness_data[:, 1] > max_ylim[i], 1] = max_ylim[i] # just for plot
        if t == 0:
            plt.plot(fitness_data[:, 0], fitness_data[:, 1], "k-",
                label="\u03BC=1, \u03BB=10, no recombination")
        else:
            plt.plot(fitness_data[:, 0], fitness_data[:, 1], "k-")
    
    print("$ total runtime: {:.2e}.".format(time.time() - start_time))

    # plot fitness curve
    plt.title("f{:d} ({})".format(i + 1, f.__name__.title()))
    plt.xlabel("function evaluations")
    plt.ylabel("fitness")
    if i == 0:
        plt.xticks(np.linspace(0, max_evaluations[i], 7))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 1:
        plt.xticks(np.linspace(0, max_evaluations[i], 5))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 2:
        plt.xticks(np.linspace(0, max_evaluations[i], 6))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 3:
        plt.xticks(np.linspace(0, max_evaluations[i], 8))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 5, 4))
    elif i == 4:
        plt.xticks(np.linspace(0, max_evaluations[i], 8))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 5:
        plt.xticks(np.linspace(0, max_evaluations[i], 7))
        plt.yticks(np.linspace(-800, 400, 7))
    elif i == 6:
        plt.xticks(np.linspace(0, max_evaluations[i], 8))
        plt.yticks(np.linspace(-800, 400, 7))
    elif i == 7:
        plt.xticks(np.linspace(0, max_evaluations[i], 6))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 4, 8))
    elif i == 8:
        plt.xticks(np.linspace(0, max_evaluations[i], 6))
        plt.yscale("log")
        plt.yticks(np.logspace(-10, 0, 6))
    plt.legend(loc=1)
    plt.savefig("{}_{}.png".format(i + 1, f.__name__))
    # plt.show()

##
print("$$ plot the time consumption of MVA-ES on Sphere:")
ndim_problem = [2, 5, 10, 20, 50, 100, 200, 400, 800]
n_trials = 70
best_so_far_y = np.empty((len(ndim_problem), n_trials))
runtime = np.empty((len(ndim_problem), n_trials))
# set options of optimizer: a (1, 10) variant without recombination
options = {"n_individuals": 10,
    "n_parents": 1,
    "max_evaluations": 6000,
    "step_size": 0.3}
for i, d in enumerate(ndim_problem):
    # set parameters of problem
    np.random.seed(20200624 + d) # for repeatability
    lower_boundary, upper_boundary = -5 * np.ones((d,)), 5 * np.ones((d,))
    cf.generate_rotation_shift(cf.sphere, d, lower_boundary + 1, upper_boundary - 1)
    problem = {"ndim_problem": d,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary}
    # solve
    for t in range(n_trials):
        options_t = copy.deepcopy(options)
        options_t["seed"] = t
        solver = MVA(problem, options_t)
        results = solver.optimize(cf.sphere)
        best_so_far_y[i, t] = results["best_so_far_y"]
        runtime[i, t] = results["runtime"]
    message = "  dimension {}: best_so_far_y {:.2e}, runtime: {:.2e}"
    print(message.format(d, np.mean(best_so_far_y[i, :]), np.mean(runtime[i, :])))
figure = plt.figure()
plt.plot(ndim_problem, np.mean(runtime, 1), "k-", label="MVA-ES")
plt.title("time per generation with function f1 (Sphere)")
plt.xlabel("dimension (N)")
plt.ylabel("time [sec]")
plt.xticks(np.linspace(0, 800, 9))
plt.yticks(np.linspace(0, 1, 11))
plt.legend(loc=2)
plt.savefig("time_consumption.png")
# plt.show()
