import time
import numpy as np

from optimizer import compress_fitness_data
from es import MuCommaLambda


class Ostermeier(MuCommaLambda):
    """Ostermeier's (1,lambda)-Evolution Strategy (Ostermeier's (1,lambda)-ES).
    
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

    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize
        if self._X.ndim == 1:
            parent = np.copy(self._X)
        else:
            parent = np.copy(self._X[0, :]) # discard all other individuals
        self._X = None # clear
        start_evaluation = time.time()
        y = fitness_function(parent)
        time_evaluations = time.time() - start_evaluation
        n_evaluations = 1
        best_so_far_x, best_so_far_y = np.copy(parent), np.copy(y)
        Y = np.tile(y, (self.n_individuals,)) # fitness of population
        
        if self.save_fitness_data:
            fitness_data = [y]

        # initialize variables involved for all individuals
        X = np.empty((self.n_individuals, self.ndim_problem)) # population
        delta = np.ones((self.ndim_problem,)) # individual step-sizes
        xi = np.empty((self.n_individuals,)) # global step-size
        Z = np.empty((self.n_individuals, self.ndim_problem)) # Gaussian noises
        beta = np.sqrt(1 / self.ndim_problem) # adaptation vs precision
        beta_scal = 1 / self.ndim_problem # in the range (0, 1)
        # beta_scal: small values facilitate a precise but time-consuming adaptaton

        # iterate
        termination = "max_evaluations"
        while n_evaluations < self.max_evaluations:
            # 1. creation of lambda offspring
            for k in range(self.n_individuals):
                if self.rng.uniform(0, 1, 1) > 0.5:
                    xi[k] = 1.4
                else:
                    xi[k] = 1 / 1.4
                Z[k, :] = self.rng.standard_normal((self.ndim_problem,))
                X[k, :] = parent + xi[k] * delta * Z[k, :]

                start_evaluation = time.time()
                Y[k] = fitness_function(X[k, :])
                time_evaluations += time.time() - start_evaluation
                n_evaluations += 1

                if self.save_fitness_data:
                    fitness_data.append(Y[k])
                
                # update best-so-far x and y
                if best_so_far_y > Y[k]:
                    best_so_far_x, best_so_far_y = np.copy(X[k, :]), np.copy(Y[k])
                
                # check three termination criteria
                if n_evaluations >= self.max_evaluations:
                    break
                runtime = time.time() - start_optimization
                if runtime >= self.max_runtime:
                    termination = "max_runtime"
                    break
                if best_so_far_y <= self.threshold_fitness:
                    termination = "threshold_fitness"
                    break
            
            # 2. selection / adaptation
            sel = np.argmin(Y)
            parent = np.copy(X[sel, :])
            delta *= (np.power(xi[sel], beta) *
                np.power(np.exp(np.abs(Z[sel, :]) - np.sqrt(2 / np.pi)), beta_scal))
            
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
            "time_compression": time_compression,
            "delta": delta}
        return results
