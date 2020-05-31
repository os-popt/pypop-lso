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
        self.initial_guess = options.get("initial_guess")
        self.seed_initial_guess = options.get("seed_initial_guess")
        self.n_individuals = options.get("n_individuals", 1)

    def optimizer(self, fitness_function):
        pass
    
    def __repr__(self):
        tip = "NOTE that the optimizer'name to be printed is not set. " +\
            "Set the field 'optimizer_name' for the dict object 'options'."
        optimizer_name = self.options.get("optimizer_name", tip)
        return optimizer_name
