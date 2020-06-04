from optimizer import PopulationOptimizer


class EvolutionStrategy(PopulationOptimizer):
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "EvolutionStrategy (ES)")
        if options.get("step_size") is None:
            raise ValueError("the option 'step_size' is a hyperparameter needed to be set or tuned in advance.")
        self.step_size = options.get("step_size")
        PopulationOptimizer.__init__(self, problem, options)
