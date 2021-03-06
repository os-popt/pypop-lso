import time
import numpy as np

from es import MuCommaLambda


class Ostermeier(MuCommaLambda):
    """Restart-based Ostermeier's (1,lambda)-Evolution Strategy (Ostermeier's (1,lambda)-ES).
    
    The concept of relatively large mutations within one generation,
    but passing only smaller variations to the next generation,
    is applicable successfully to parameter optimization
    superimposed with Gaussian noise.

    Reference
    ---------
    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994.
    A derandomized approach to self-adaptation of evolution strategies.
    Evolutionary Computation, 2(4), pp.369-380.
    https://www.mitpressjournals.org/doi/10.1162/evco.1994.2.4.369
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "Ostermeier (Ostermeier's (1,lambda)-ES)")
        MuCommaLambda.__init__(self, problem, options)
        self.beta_scal = options.get("beta_scal", 1 / self.ndim_problem)

    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize
        parent = MuCommaLambda._get_m(self)
        start_evaluation = time.time()
        y = fitness_function(parent)
        n_evaluations, time_evaluations = 1, time.time() - start_evaluation
        best_so_far_x, best_so_far_y = np.copy(parent), np.copy(y)

        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # initialize variables involved for all individuals
        beta = np.sqrt(1 / self.ndim_problem) # adaptation vs precision
        beta_scal = self.beta_scal # in the range (0, 1)
        # beta_scal: small values facilitate a precise but time-consuming adaptaton

        # iterate
        termination = "max_evaluations"
        is_restart, n_restart = True, 0
        while n_evaluations < self.max_evaluations:
            if is_restart:
                if n_restart > 0:
                    parent = self.rng.uniform(
                        self.lower_boundary, self.upper_boundary, (self.ndim_problem, ))
                    start_evaluation = time.time()
                    y = fitness_function(parent)
                    time_evaluations += (time.time() - start_evaluation)
                    n_evaluations += 1
                    if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(parent), np.copy(y)
                    if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                        history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                    if self.save_fitness_data: fitness_data.append(np.copy(y))
                    
                    self.n_individuals *= 2
                    if self.n_individuals >= 1000: self.n_individuals = 1000
                
                Y = np.tile(y, (self.n_individuals,))  # fitness of population
                X = np.empty((self.n_individuals, self.ndim_problem))  # population
                delta = np.ones((self.ndim_problem,))  # individual step-sizes
                xi = np.empty((self.n_individuals,))  # global step-size
                Z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noises
                is_restart = False

            # 1. creation of lambda offspring
            for k in range(self.n_individuals):
                if self.rng.uniform(0, 1, 1) > 0.5: xi[k] = 1.4
                else: xi[k] = 1 / 1.4
                Z[k, :] = self.rng.standard_normal((self.ndim_problem,))
                X[k, :] = parent + xi[k] * delta * Z[k, :]

                start_evaluation = time.time()
                Y[k] = fitness_function(X[k, :])
                time_evaluations += time.time() - start_evaluation
                n_evaluations += 1

                if self.save_fitness_data: fitness_data.append(Y[k])
                
                # update best-so-far x and y
                if best_so_far_y > Y[k]: best_so_far_x, best_so_far_y = np.copy(X[k, :]), np.copy(Y[k])
                if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))

                # check three termination criteria
                is_break, termination = MuCommaLambda._check_terminations(
                    self, n_evaluations, time.time() - start_optimization, best_so_far_y)
                if is_break: break
            
            # 2. selection / adaptation
            sel = np.argmin(Y)
            parent = np.copy(X[sel, :])
            delta *= (np.power(xi[sel], beta) *
                np.power(np.exp(np.abs(Z[sel, :]) - np.sqrt(2 / np.pi)), beta_scal))
            
            # check termination criteria
            runtime = time.time() - start_optimization
            is_break, termination = MuCommaLambda._check_terminations(
                    self, n_evaluations, runtime, best_so_far_y)
            if is_break: break
            if np.min(delta) <= self.threshold_step_size:
                is_restart, n_restart = True, n_restart + 1
            if np.max(delta) >= 3 * np.min(self.upper_boundary - self.lower_boundary):
                is_restart, n_restart = True, n_restart + 1

        fitness_data, time_compression = MuCommaLambda._save_data(self, history_x, fitness_data)

        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression,
            "delta": delta,
            "n_restart": n_restart}
        return results