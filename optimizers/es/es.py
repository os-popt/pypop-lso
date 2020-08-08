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
        self.n_individuals = int(options.get("n_individuals",
            4 + np.floor(3 * np.log(problem["ndim_problem"]))))
        # mu -> n_parents
        self.n_parents = int(options.get("n_parents", np.floor(self.n_individuals / 2)))

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
