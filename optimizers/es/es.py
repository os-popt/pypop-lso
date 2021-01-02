import numpy as np

from optimizer import PopulationOptimizer


class EvolutionStrategy(PopulationOptimizer):
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "EvolutionStrategy (ES)")
        if options.get("step_size") is None:
            raise ValueError("the option 'step_size' is a hyperparameter which should be set or tuned in advance.")
        self.step_size = options.get("step_size")
        self.threshold_step_size = options.get("threshold_step_size", 1e-15)
        PopulationOptimizer.__init__(self, problem, options)
        # lambda -> n_individuals
        self.n_individuals = int(options.get("n_individuals", 4 + np.floor(3 * np.log(problem["ndim_problem"]))))
        # mu -> n_parents
        self.n_parents = int(options.get("n_parents", np.floor(self.n_individuals / 2)))
        self.save_step_size_data = options.get("save_step_size_data", False)
    
    def _get_m(self):
        if self._X.ndim == 1: m = np.copy(self._X)
        else: m = np.copy(self._X[0, :]) # discard all other individuals
        self._X = None # clear
        return m
    
    def _check_terminations(self, n_evaluations, runtime, best_so_far_y, sigma=None):
        is_break, termination = PopulationOptimizer._check_terminations(self,
            n_evaluations, runtime, best_so_far_y)
        if (sigma != None) and (sigma <= self.threshold_step_size):
            is_break, termination = True, "threshold_step_size (lower)"
        return is_break, termination

class OnePlusOne(EvolutionStrategy):
    def __init__(self, problem, options):
        options["n_individuals"] = 1
        options["n_parents"] = 1
        EvolutionStrategy.__init__(self, problem, options)

class MuCommaLambda(EvolutionStrategy):
    def __init__(self, problem, options):
        EvolutionStrategy.__init__(self, problem, options)
        if self.n_individuals < 4:
            self.n_individuals = int(4 + np.floor(3 * np.log(problem["ndim_problem"])))
            print("the option 'n_individuals' should >= 4, " +\
                "and it has been reset to {:d} " +
                "(a commonly suggested value).".format(self.n_individuals))
        if self.n_parents < 1:
            self.n_parents = 1
            print("the option 'n_parents' should >= 1, and it has been reset to 1.")
        if self.n_parents > self.n_individuals:
            raise ValueError("the option 'n_parents' should <= the option 'n_individuals'.")