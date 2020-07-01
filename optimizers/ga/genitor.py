import time
import numpy as np

from optimizer import PopulationOptimizer, compress_fitness_data


class GENITOR(PopulationOptimizer):
    """Genetic reinforcement learning (GENITOR)."""
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "GENITOR")
        options.setdefault("n_individuals", 50)
        PopulationOptimizer.__init__(self, problem, options)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
                fitness_function = self.fitness_function
        
        # initialize
        n_individuals = self.n_individuals
        X = np.copy(self._X) # population
        self._X = None # clear
        Y = np.empty((n_individuals,))
        time_evaluations = 0
        n_evaluations = 0 # counter of fitness evaluations
        best_so_far_y = np.inf # best-so-far fitness

        if self.save_fitness_data:
            fitness_data = []
        
        # iterate
        termination = "max_evaluations" # by default
        while n_evaluations < self.max_evaluations:
            # select one parent (no crossover is implemented)
            if n_evaluations < n_individuals:
                s = n_evaluations # index for selected parent
                x = X[s, :]
            else:
                index = np.argsort(Y)
                X, Y = X[index, :], Y[index]
                selection_prob = np.arange(n_individuals, 0, -1)
                selection_prob = selection_prob / np.sum(selection_prob)
                s = self.rng.choice(n_individuals, 1, p=selection_prob)
                # mutate (adding a random value with range +-10.0 is not implemented)
                X[s, :] += 0.03 * self.rng.uniform(
                    self.lower_boundary - self.upper_boundary,
                    self.upper_boundary - self.lower_boundary)
                x = X[s, :]
            
            # evaluate
            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1
            Y[s] = y
            
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
        return results
