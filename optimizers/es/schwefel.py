import time
import numpy as np

from es import OnePlusOne


class Schwefel(OnePlusOne):
    """Schwefel's (1+1)-Evolution Strategy (Schwefel's (1+1)-ES).
    
    Reference
    ---------
    Back, T., Hoffmeister, F. and Schwefel, H.P., 1991, July.
    A survey of evolution strategies.
    In Proceedings of the Fourth International Conference on Genetic Algorithms (Vol. 2, No. 9).
    Morgan Kaufmann Publishers, San Mateo, CA.
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "Schwefel (Schwefel's (1+1)-ES)")
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
        n_evaluations_restart = 1 # counter of fitness evaluations for each re-start
        best_so_far_x, best_so_far_y = np.copy(x), np.copy(y) # best-so-far solution and fitness
        parent_x, parent_y = np.copy(x), np.copy(y) # for each re-start

        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # initilize parameters for step_size adjustment
        increase_ratio, decrease_ratio = 1 / 0.82, 0.82 # ratio to increase/decrease step_size
        n_success = 0 # number of successful mutations
        step_size = self.step_size

        # iterate
        termination = "max_evaluations"  # by default
        is_restart, n_restart = True, 0
        while n_evaluations < self.max_evaluations:    
            if is_restart:
                if n_restart > 0:
                    x = self.rng.uniform(self.initial_lower_boundary,
                        self.initial_upper_boundary)
                    start_evaluation = time.time()
                    y = fitness_function(x)  # evaluate fitness of population
                    time_evaluations += (time.time() - start_evaluation)
                    n_evaluations += 1
                    n_evaluations_restart = 1
                    parent_x, parent_y = np.copy(x), np.copy(y)
                    n_success = 0
                    step_size = self.step_size
                    if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
                    if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                        history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                    if self.save_fitness_data: fitness_data.append(y)
                is_restart = False
            else:
                # adjust step_size every `ndim_problem` mutations (via the 1/5 success rule)
                if (n_evaluations_restart > 1) and (((n_evaluations_restart - 1) %\
                        self.ndim_problem) == 0):
                    if n_success > 0.2 * self.ndim_problem:
                        step_size *= increase_ratio
                    elif n_success < 0.2 * self.ndim_problem:
                        step_size *= decrease_ratio
                    n_success = 0
            
            # sample from normal distribution and then evaluate
            x = parent_x + step_size * self.rng.standard_normal((self.ndim_problem,))

            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1
            n_evaluations_restart += 1

            if self.save_fitness_data: fitness_data.append(y)

            if parent_y > y:
                parent_x, parent_y = np.copy(x), np.copy(y)
                n_success += 1
            
            # update best-so-far x and y
            if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
            if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
            
            # check three termination criteria
            runtime = time.time() - start_optimization
            is_break, termination = OnePlusOne._check_terminations(
                    self, n_evaluations, runtime, best_so_far_y)
            if is_break: break

            # check re-start condition
            if step_size <= self.threshold_step_size:
                is_restart, n_restart = True, n_restart + 1

        fitness_data, time_compression = OnePlusOne._save_data(self, history_x, fitness_data)
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression,
            "step_size": step_size,
            "n_restart": n_restart}
        return results