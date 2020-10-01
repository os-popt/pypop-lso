import numpy as np
import matplotlib.pyplot as plt

from continuous_functions import _transform_and_check, schwefel12 as schwefel
from continuous_functions import rosenbrock
from ostermeier import Ostermeier


# set ES's hyper-parameters
n_parents, n_individuals = 1, 10

# Figure 1 (note that only the bottom sub-figure is plotted here)
def hyper_ellipsoid(x):
    x = _transform_and_check(x)
    w = np.arange(1, x.size + 1)
    return np.sum(np.power(w * x, 2))

ndim_problem = 30
problem = {"ndim_problem": ndim_problem,
    "lower_boundary": -10 * np.ones((ndim_problem,)), # not given
    "upper_boundary": 10 * np.ones((ndim_problem,))} # not given
options = {"max_evaluations": 2.5e4,
    "n_parents": n_parents,
    "n_individuals": n_individuals,
    "initial_guess": np.ones((ndim_problem,)),
    "step_size": 0.3, # not given
    "threshold_fitness": 1e-10,
    "seed": 20200929} # not given
solver = Ostermeier(problem, options)
results = solver.optimize(hyper_ellipsoid)
print(results)
fitness_data = results["fitness_data"]
plt.figure(1)
plt.plot(fitness_data[:, 0], fitness_data[:, 1])
plt.title("Ostermeier's ES on Hyper-Ellipsoid")
plt.xlabel("function evaluations")
xticks = np.linspace(0, options["max_evaluations"], 6)
xticks_label = ("{:.1E}".format(x) for x in xticks)
plt.xticks(xticks, xticks_label)
plt.ylabel("best function value")
plt.yscale("log")
plt.yticks(np.logspace(-10, 4, 15))
plt.savefig("Ostermeier-Figure-1.png")

# Figure 3
ndim_problem = 20
problem = {"ndim_problem": ndim_problem,
    "lower_boundary": -65 * np.ones((ndim_problem,)),
    "upper_boundary": 65 * np.ones((ndim_problem,))}
options = {"max_evaluations": 6e4,
    "n_parents": n_parents,
    "n_individuals": n_individuals,
    "step_size": 0.3, # not given
    "threshold_fitness": 1e-3,
    "seed": 20200930} # not given
solver = Ostermeier(problem, options)
results = solver.optimize(schwefel)
print(results)
fitness_data = results["fitness_data"]
plt.figure(3)
plt.plot(fitness_data[:, 0], fitness_data[:, 1])
plt.title("Ostermeier's ES on Schwefel's Problem")
plt.xlabel("function evaluations")
xticks = np.linspace(0, options["max_evaluations"], 7)
xticks_label = ("{:.1E}".format(x) for x in xticks)
plt.xticks(xticks, xticks_label)
plt.ylabel("best function value")
plt.yscale("log")
plt.yticks(np.logspace(-3, 5, 9))
plt.savefig("Ostermeier-Figure-3.png")

# Figure 4
ndim_problem = 30
problem = {"ndim_problem": ndim_problem,
    "lower_boundary": -1 * np.ones((ndim_problem,)), # not given
    "upper_boundary": 1 * np.ones((ndim_problem,))} # not given
options = {"max_evaluations": 2.4e6,
    "n_parents": n_parents,
    "n_individuals": n_individuals,
    "initial_guess": np.zeros((ndim_problem,)),
    "step_size": 0.3, # not given
    "threshold_fitness": 1e-6,
    "seed": 20201001} # not given
solver = Ostermeier(problem, options)
results = solver.optimize(rosenbrock)
print(results)
fitness_data = results["fitness_data"]
plt.figure(4)
plt.plot(fitness_data[:, 0], fitness_data[:, 1])
plt.title("Ostermeier's ES on Generalized Rosenbrock")
plt.xlabel("function evaluations")
xticks = np.linspace(0, options["max_evaluations"], 7)
xticks_label = ("{:.1E}".format(x) for x in xticks)
plt.xticks(xticks, xticks_label)
plt.ylabel("best function value")
plt.yscale("log")
plt.yticks(np.logspace(-6, 1, 8))
plt.savefig("Ostermeier-Figure-4.png")
