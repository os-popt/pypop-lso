import time
import numpy as np

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
        x = OnePlusOne._get_m(self) # population with one individual
        start_evaluation = time.time()
        y = fitness_function(x) # evaluate fitness of population
        n_evaluations, time_evaluations = 1, time.time() - start_evaluation
        best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
        
        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None
        
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
            
            if self.save_fitness_data: fitness_data.append(y)
            
            # update best-so-far x and y
            if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
            if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
            
            # check three termination criteria
            runtime = time.time() - start_optimization
            is_break, termination = OnePlusOne._check_terminations(
                    self, n_evaluations, runtime, best_so_far_y)
            if is_break: break
        
        fitness_data, time_compression = OnePlusOne._save_data(self, history_x, fitness_data)
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression}
        return results