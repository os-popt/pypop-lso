import time
import numpy as np

from optimizer import compress_fitness_data
from es import OnePlusOne


class Rechenberg(OnePlusOne):
    """Rechenberg's (1+1)-Evolution Strategy (Rechenberg's (1+1)-ES).
    
    Reference
    ---------
    Back, T., Hoffmeister, F. and Schwefel, H.P., 1991, July.
    A survey of evolution strategies.
    In Proceedings of the Fourth International Conference on Genetic Algorithms (Vol. 2, No. 9).
    Morgan Kaufmann Publishers, San Mateo, CA.
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "Rechenberg (Rechenberg's (1+1)-ES)")
        OnePlusOne._check_n_individuals(self, options, self.__class__.__name__)
        OnePlusOne.__init__(self, problem, options)
    
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

        if self.save_fitness_data:
            fitness_data = [y]
        
        # iterate
        termination = "max_evaluations" # by default
        while n_evaluations < self.max_evaluations:
            # sample from normal distribution and then evaluate
            x = best_so_far_x + self.step_size *\
                self.rng.standard_normal((self.ndim_problem,))
            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1
            
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
