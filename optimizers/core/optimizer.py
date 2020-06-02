import numpy as np


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

    def optimize(self, fitness_function):
        pass
    
    def __repr__(self):
        tip = "NOTE that the optimizer'name to be printed is not set. " +\
            "Set the field 'optimizer_name' for the dict object 'options'."
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
