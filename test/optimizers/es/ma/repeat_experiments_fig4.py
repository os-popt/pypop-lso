"""Repeat experiments of Fig. 4 from the following paper:
    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115

    $ python repeat_experiments_fig4.py -bf=sphere -ndp=3 -tf=1e-14 -mg=140
    $ python repeat_experiments_fig4.py -bf=sphere -ndp=30 -tf=1e-14 -mg=600
"""
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

from continuous_functions import *
from ma import MA


if __name__ == "__main__":
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    
    # set benchmark function parameters
    parser.add_argument('--benchmark_function', '-bf', type=str)
    parser.add_argument('--ndim_problem', '-ndp', type=int)

    # set optimizer options
    parser.add_argument('--max_generations', '-mg', type=int)
    parser.add_argument('--optimizer_seed', '-os', type=int, default=20201030)
    parser.add_argument('--threshold_fitness', '-tf', type=float)

    # set experiment parameters
    parser.add_argument('--n_trials', '-nt', type=int, default=20)
    parser.add_argument('--is_plot', '-ip', type=bool, default=False)
    
    # set plot parameters
    parser.add_argument('--xticks_upper', '-xu', type=int, default=0)
    parser.add_argument('--xticks_n', '-xn', type=int, default=0)
    parser.add_argument('--yticks_lower', '-yl', type=int, default=0)
    parser.add_argument('--yticks_upper', '-yu', type=int, default=0)
    parser.add_argument('--yticks_n', '-yn', type=int, default=0)
    
    # parse all parameters
    args = parser.parse_args()
    params = vars(args)
    if params["xticks_upper"] <= 0: params["xticks_upper"] = params["max_generations"]
    
    data_dir = "Fig4"
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    n_individuals = int(4 + np.floor(3 * np.log(params["ndim_problem"])))
    problem = {"fitness_function": params["benchmark_function"],
        "ndim_problem": params["ndim_problem"],
        "lower_boundary": -5.0 * np.ones((params["ndim_problem"],)), # undefined in the paper
        "upper_boundary": 5.0 * np.ones((params["ndim_problem"],))} # undefined in the paper
    min_evaluations, min_step_size_data = np.inf, np.inf
    for i in range(params["n_trials"]):
        if params["benchmark_function"] == "rosenbrock": # undefined in the paper
            initial_guess = -1.0 * np.ones((params["ndim_problem"],))
        else:
            initial_guess = np.ones((params["ndim_problem"],))
        if params["benchmark_function"] in ["parabolic_ridge", "sharp_ridge"]:
            threshold_fitness = -1e10
        else:
            threshold_fitness = params["threshold_fitness"]
        optimizer_options = {"max_evaluations": params["max_generations"] * n_individuals + 1,
            "seed": params["optimizer_seed"] + i, # undefined in the paper
            "step_size": 1.0,
            "initial_guess": initial_guess,
            "threshold_fitness": threshold_fitness,
            "len_fitness_data": 0,
            "save_step_size_data": True}
        results_file = os.path.join(data_dir, "{:s}-Dim-{:d}-Trial-{:d}.pickle".format(
                params["benchmark_function"], params["ndim_problem"], i + 1))
        if not(params["is_plot"]):
            solver = MA(problem, optimizer_options)
            results = solver.optimize(eval(params["benchmark_function"]))
            with open(results_file, "wb") as results_handle:
                pickle.dump(results, results_handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(results_file, 'rb') as results_handle:
                results = pickle.load(results_handle)
            if len(results["fitness_data"]) < min_evaluations:
                min_evaluations = len(results["fitness_data"])
            if len(results["step_size_data"]) < min_step_size_data:
                min_step_size_data = len(results["step_size_data"])
    
    if params["is_plot"]:
        fitness_data = np.zeros((min_evaluations, 2))
        step_size_data = np.zeros((min_step_size_data))
        n_stat = 0
        for i in range(params["n_trials"]):
            results_file = os.path.join(data_dir, "{:s}-Dim-{:d}-Trial-{:d}.pickle".format(
                params["benchmark_function"], params["ndim_problem"], i + 1))
            with open(results_file, "rb") as results_handle:
                results = pickle.load(results_handle)
            if params["benchmark_function"] == "rosenbrock":
                fitness_data_tmp = np.array(results["fitness_data"])
                if np.min(fitness_data_tmp[:min_evaluations, 1]) > 1e-7:
                    continue
            if params["benchmark_function"] in ["parabolic_ridge", "sharp_ridge"]:
                fitness_data += np.abs(np.array(results["fitness_data"][:min_evaluations, :]))
                step_size_data += np.abs(np.array(results["step_size_data"][:min_step_size_data]))
            else:
                fitness_data += np.array(results["fitness_data"][:min_evaluations, :])
                step_size_data += np.array(results["step_size_data"][:min_step_size_data])
            n_stat += 1
        fitness_data /= n_stat
        step_size_data /= n_stat
        plt.figure()
        fitness_data = fitness_data[np.arange(0, len(fitness_data), n_individuals), 1]
        plt.plot(range(len(fitness_data)), fitness_data, color="red")
        plt.plot(range(len(step_size_data)), step_size_data, color="magenta")
        plt.title("{:s} N = {:d}".format(params["benchmark_function"], params["ndim_problem"]))
        plt.xlabel("g")
        xticks = np.linspace(0, params["xticks_upper"], params["xticks_n"])
        plt.xticks(xticks)
        plt.ylabel("step-size and best-so-far fitness")
        plt.yscale("log")
        plt.yticks(np.logspace(params["yticks_lower"], params["yticks_upper"], params["yticks_n"]))
        plt.savefig(os.path.join(data_dir, "{:s}-Dim-{:d}.png".format(
            params["benchmark_function"], params["ndim_problem"])))

    print("$$$$$$$ Total Runtime: {:.2e}.".format(time.time() - start_time))