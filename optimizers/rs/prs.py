import time
import numpy as np

from optimizer import PopulationOptimizer, compress_fitness_data


class PureRandomSearch(PopulationOptimizer):
    """Pure Random Search (PRS) sampling uniformly on the box-constrained search space.
    
    Reference
    ---------
    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "PureRandomSearch (PRS)")
        PopulationOptimizer._check_n_individuals(self, options, self.__class__.__name__)
        PopulationOptimizer.__init__(self, problem, options)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
                fitness_function = self.fitness_function
        
        # initialize
        x = np.copy(self._X) # population with one individual
        self._X = None # to save memory space
        start_evaluation = time.time()
        y = fitness_function(x) # evaluate fitness of population
        time_evaluations = time.time() - start_evaluation # time used for fitness evaluations
        n_evaluations = 1 # counter of fitness evaluations
        best_so_far_x = np.copy(x) # best-so-far solution
        best_so_far_y = np.copy(y) # best-so-far fitness
        history_x = np.hstack((n_evaluations, best_so_far_x))

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
            if self.save_best_so_far_x:
                if not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x,
                        np.hstack((n_evaluations, best_so_far_x))))
            
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
        
        if self.save_best_so_far_x:
            np.savetxt(self.txt_best_so_far_x, history_x)
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression}
        return results
