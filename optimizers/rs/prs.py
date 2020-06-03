import time
import numpy as np

from optimizer import PopulationOptimizer, compress_fitness_data


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
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
                fitness_function = self.fitness_function
        
        # initialize
        x = self._X # population with one individual
        start_evaluation = time.time()
        y = fitness_function(x) # evaluate fitness of population
        time_evaluations = time.time() - start_evaluation # time used for fitness evaluations
        n_evaluations = 1 # counter of fitness evaluations
        best_so_far_x = np.copy(x) # best-so-far solution
        best_so_far_y = np.copy(y) # best-so-far fitness

        if self.save_fitness_data:
            fitness_data = [y]
        
        # iterate
        termination = "max_evaluations" # by default
        while n_evaluations < self.max_evaluations:
            # sample uniformly and then evaluate
            x = self.rng.uniform(self.lower_boundary,
                self.upper_boundary,
                self._X_size)
            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += self.n_individuals
            
            if self.save_fitness_data:
                fitness_data.append(y)
            
            # update best-so-far x and y
            if best_so_far_y > y:
                best_so_far_x = np.copy(x)
                best_so_far_y = np.copy(y)
            
            # check two termination criteria
            runtime = time.time() - start_optimization
            if runtime >= self.max_runtime:
                termination = "max_runtime"
                break
            if best_so_far_y <= self.threshold_fitness:
                termination = "threshold_fitness"
                break
        
        if self.save_fitness_data:
            start_compression = time.time()
            fitness_data = compress_fitness_data(fitness_data, self.len_fitness_data)
            time_compression = time.time() - start_compression
        else:
            fitness_data = None
            time_compression = None
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression}
        self._X = None # clear
        return results
