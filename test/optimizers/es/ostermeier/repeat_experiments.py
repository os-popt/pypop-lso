import numpy as np
import matplotlib.pyplot as plt

from continuous_functions import _transform_and_check
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
