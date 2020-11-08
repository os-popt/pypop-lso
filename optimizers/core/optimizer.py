import time
import numpy as np


def compress_fitness_data(fitness_data, len_fitness_data=2000):
    # converge in non-increasing order
    fitness_data = np.array(fitness_data)
    for index in range(len(fitness_data) - 1):
        if fitness_data[index] < fitness_data[index + 1]:
            fitness_data[index + 1] = fitness_data[index]
    
    if len_fitness_data <= 0:
        return np.stack((np.arange(1, len(fitness_data) + 1), fitness_data), 1)
    
    # compress for space saving
    frequency = int(np.ceil(len(fitness_data) / len_fitness_data))
    frequency = max(100, round(frequency, -(len(str(frequency)) - 1)))
    index = np.append(np.arange(0, len(fitness_data) - 1, frequency), len(fitness_data) - 1)
    fitness_data = fitness_data[index]
    index[0], index[len(index) - 1] = index[0] + 1, index[len(index) - 1] + 1 # 1-based index
    fitness_data = np.stack((index, fitness_data), 1)
    return fitness_data

class Optimizer(object):
    """The base class of all optimizers for real-valued, black-box minimization."""
    def __init__(self, problem, options):
        self.problem = problem
        self.options = options

        # problem-related
        self.ndim_problem = problem["ndim_problem"]
        self.upper_boundary = problem["upper_boundary"]
        self.lower_boundary = problem["lower_boundary"]
        self.initial_upper_boundary = problem.get("initial_upper_boundary", problem["upper_boundary"])
        self.initial_lower_boundary = problem.get("initial_lower_boundary", problem["lower_boundary"])
        self.fitness_function = problem.get("fitness_function")
        self.problem_name = problem.get("problem_name")
        if (self.problem_name is None) and hasattr(self.fitness_function, "__name__"):
            self.problem_name = self.fitness_function.__name__

        # optimizer-related
        self.max_evaluations = options.get("max_evaluations", np.inf)
        self.max_runtime = options.get("max_runtime", np.inf)
        self.threshold_fitness = options.get("threshold_fitness", -np.inf)
        self.seed = options.get("seed")
        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed)
        self.initial_guess = options.get("initial_guess")
        self.seed_initial_guess = options.get("seed_initial_guess")
        if self.seed_initial_guess is None:
            self.seed_initial_guess = self.rng.integers(np.iinfo(np.int64).max)
        self.n_individuals = options.get("n_individuals", 1)
        self.save_fitness_data = options.get("save_fitness_data", True)
        self.len_fitness_data = options.get("len_fitness_data", 2000)
        self.save_best_so_far_x = options.get("save_best_so_far_x", False)
        self.freq_best_so_far_x = options.get("freq_best_so_far_x", 1000)
        self.txt_best_so_far_x = options.get("txt_best_so_far_x", "history_best_so_far_x.txt")

    def optimize(self, fitness_function=None):
        pass
    
    def _check_n_individuals(self, options, class_name, n_individuals=1):
        """Check the number of individuals, if necessary."""
        if options.get("n_individuals") not in [None, n_individuals]:
            print("For {}, 'n_individuals' is reset to {:d} by default (from {:d}).".format(
                class_name, n_individuals, options.get("n_individuals")))
            options["n_individuals"] = n_individuals
    
    def _check_terminations(self, n_evaluations, runtime, best_so_far_y):
        if n_evaluations >= self.max_evaluations: return True, "max_evaluations"
        if runtime >= self.max_runtime: return True, "max_runtime"
        if best_so_far_y <= self.threshold_fitness: return True, "threshold_fitness"
        return False, "no_termination"
    
    def _save_data(self, history_x, fitness_data):
        if self.save_best_so_far_x: np.savetxt(self.txt_best_so_far_x, history_x)

        if self.save_fitness_data:
            start_compression = time.time()
            fitness_data = compress_fitness_data(fitness_data, self.len_fitness_data)
            time_compression = time.time() - start_compression
        else:
            fitness_data, time_compression = None, None
        return fitness_data, time_compression

    def __repr__(self):
        tip = "NOTE that the optimizer'name to be printed is not set. " +\
            "Set 'optimizer_name' for the dict object 'options'."
        optimizer_name = self.options.get("optimizer_name", tip)
        return optimizer_name

class PopulationOptimizer(Optimizer):
    """The base class of all population-based optimizers for continuous, black-box minimization."""
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals == 1:
            self._X_size = (self.ndim_problem,)
        else:
            self._X_size = (self.n_individuals, self.ndim_problem)
        
        # initialize population (i.e., _X)
        if self.initial_guess is None:
            self._X = np.random.default_rng(self.seed_initial_guess).uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                self._X_size)
        else:
            self._X = self.initial_guess