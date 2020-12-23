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
        OnePlusOne.__init__(self, problem, options)

    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize population with one parent and one offspring
        x = OnePlusOne._get_m(self) # one offspring
        start_evaluation = time.time()
        y = fitness_function(x)
        time_evaluations, n_evaluations = time.time() - start_evaluation, 1
        best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # for each re-start
        n_evaluations_restart, parent_x, parent_y = 1, np.copy(x), np.copy(y)
        
        # initilize parameters for (global) step-size adjustment
        step_size = self.step_size # mutation strength (global step-size)
        # ratio to increase/decrease (global) step-size
        increase_ratio, decrease_ratio = 1 / 0.82, 0.82
        n_success = 0 # number of successful mutations

        # iterate / evolve
        termination = "max_evaluations" # by default
        is_restart, n_restart = True, 0
        while n_evaluations < self.max_evaluations:    
            if is_restart:
                is_restart = False
                if n_restart > 0:
                    x = self.rng.uniform(self.lower_boundary, self.upper_boundary, (self.ndim_problem,))
                    start_evaluation = time.time()
                    y = fitness_function(x)
                    time_evaluations += (time.time() - start_evaluation)
                    n_evaluations += 1
                    n_evaluations_restart, parent_x, parent_y = 1, np.copy(x), np.copy(y)
                    step_size = self.step_size
                    n_success = 0
                    if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
                    if self.save_fitness_data: fitness_data.append(y)
                    if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                        history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
            else:
                # adjust step-size every `ndim_problem` mutations (via the 1/5 success rule)
                if (n_evaluations_restart > 1) and (((n_evaluations_restart - 1) % self.ndim_problem) == 0):
                    if n_success > 0.2 * self.ndim_problem:
                        step_size *= increase_ratio
                    elif n_success < 0.2 * self.ndim_problem:
                        step_size *= decrease_ratio
                    n_success = 0
            
            # sample one offspring from normal distribution and then evaluate
            x = parent_x + step_size * self.rng.standard_normal((self.ndim_problem,))

            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1
            n_evaluations_restart += 1

            if parent_y > y:
                parent_x, parent_y = np.copy(x), np.copy(y)
                n_success += 1
            
            # update best-so-far x and y
            if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
            if self.save_fitness_data: fitness_data.append(y)
            if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
            
            # check four termination criteria
            runtime = time.time() - start_optimization
            is_break, termination = OnePlusOne._check_terminations(
                self, n_evaluations, runtime, best_so_far_y, step_size)
            if termination == "threshold_step_size (lower)":
                is_restart, n_restart = True, n_restart + 1
            elif is_break: break
        
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