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
