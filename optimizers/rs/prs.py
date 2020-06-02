import numpy as np

from optimizer import PopulationOptimizer


class PureRandomSearch(PopulationOptimizer):
    """Pure Random Search (PRS) sampling uniformly on the box-constrained search space."""
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "PureRandomSearch (PRS)")
        n_individuals = options.get("n_individuals")
        if (n_individuals != None) and (n_individuals != 1):
            options["n_individuals"] = 1
            print("For PureRandomSearch, only one individual is used to represent the population, " +\
                "and the option 'n_individuals' has been reset to 1 (not {:d}).".format(n_individuals))
        PopulationOptimizer.__init__(self, problem, options)
    
    def optimize(self, fitness_function=None):
        if (fitness_function is None) and (self.fitness_function != None):
                fitness_function = self.fitness_function
        
        # initialize
        x = self._X # population with one individual
        y = fitness_function(x) # evaluate fitness of population
        n_evaluations = 1 # counter of fitness evaluations
        best_so_far_x = np.copy(x) # best-so-far solution
        best_so_far_y = np.copy(y) # best-so-far fitness

        # iterate
        while n_evaluations < self.max_evaluations:
            # sample uniformly and then evaluate
            x = self.rng.uniform(self.lower_boundary,
                self.upper_boundary,
                self._X_size)
            y = fitness_function(x)
            n_evaluations += self.n_individuals
            
            # update best-so-far x and y
            if best_so_far_y > y:
                best_so_far_x = np.copy(x)
                best_so_far_y = np.copy(y)
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations}
        self._X = None # clear
        return results
