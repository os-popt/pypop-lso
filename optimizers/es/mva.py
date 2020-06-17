import time
import math
import numpy as np

from optimizer import compress_fitness_data
from es import EvolutionStrategy


class MainVectorAdaptation(EvolutionStrategy):
    """Main Vector Adaptation Evolution Strategy (MVA-ES) for large-scale, black-box optimization."""
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "MainVectorAdaptation (MVA-ES)")
        EvolutionStrategy.__init__(self, problem, options)
        ndim_problem = problem["ndim_problem"]
        if self.n_individuals < 4:
            self.n_individuals = int(4 + numpy.floor(3 * np.log(ndim_problem)))
            print("For MainVectorAdaptation, the option 'n_individuals' should >= 4, " +\
                "and it has been reset to {:d} " +
                "(a commonly suggested value).".format(self.n_individuals))
        self.w_v = options.get("w_v", 3)
        self.c_sigma = options.get("c_sigma", 4 / (ndim_problem + 4))
        self.chi_N = options.get("chi_N", np.sqrt(ndim_problem - 0.5))
        self.d_sigma = options.get("d_sigma", 1)
        self.c_m = options.get("c_m", 4 / (ndim_problem + 4))
        self.c_v = options.get("c_v", 2 / ((ndim_problem + np.sqrt(2)) ** 2))
        self.n_parents = options.get("n_parents", math.floor(self.n_individuals / 2))
        self.threshold_step_size = options.get("threshold_step_size", 1e-15)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()
        
        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize
        if self._X.ndim == 1:
            x_mean = np.copy(self._X)
        else:
            x_mean = np.copy(self._X[0, :]) # discard all other individuals
        self._X = None # clear
        start_evaluation = time.time()
        y = fitness_function(x_mean)
        time_evaluations = time.time() - start_evaluation
        n_evaluations = 1
        best_so_far_x = np.copy(x_mean)
        best_so_far_y = np.copy(y)
        
        if self.save_fitness_data:
            fitness_data = [y]
        
        # iterate
        termination = "max_evaluations"
        sigma = self.step_size
        c_u_sigma = np.sqrt(self.c_sigma * (2 - self.c_sigma))
        c_u_m = np.sqrt(self.c_m * (2 - self.c_m))
        p_sigma = np.zeros((self.ndim_problem,))
        p_m = np.zeros((self.ndim_problem,))
        v = np.zeros((self.ndim_problem,))
        while n_evaluations < self.max_evaluations:
            X = np.empty((self.n_individuals, self.ndim_problem)) # population
            Y = np.empty((self.n_individuals,))
            # Z is a 2-d arrary, where each row represents a "Z" defined in the original paper
            Z = np.empty((self.n_individuals, self.ndim_problem))
            for i in range(self.n_individuals):
                Z[i, :] = self.rng.standard_normal((self.ndim_problem,))
                Z1 = self.rng.standard_normal()
                X[i, :] = x_mean + sigma * (Z[i, :] + Z1 * self.w_v * v)
                start_evaluation = time.time()
                y = fitness_function(X[i, :])
                time_evaluations += (time.time() - start_evaluation)
                n_evaluations += 1
                Y[i] = y

                if self.save_fitness_data:
                    fitness_data.append(np.copy(y))
                
                # update best-so-far x and y
                if best_so_far_y > y:
                    best_so_far_x = np.copy(X[i, :])
                    best_so_far_y = np.copy(y)
                
                # check three termination criteria
                if n_evaluations >= self.max_evaluations:
                    break
                if (time.time() - start_optimization) >= self.max_runtime:
                    termination = "max_runtime"
                    break
                if best_so_far_y <= self.threshold_fitness:
                    termination = "threshold_fitness"
                    break
            
            # check four termination criteria
            runtime = time.time() - start_optimization
            if n_evaluations >= self.max_evaluations:
                break
            if runtime >= self.max_runtime:
                termination = "max_runtime"
                break
            if best_so_far_y <= self.threshold_fitness:
                termination = "threshold_fitness"
                break
            if sigma <= self.threshold_step_size:
                termination = "threshold_step_size (lower)"
                break
            if sigma >= 3 * np.min(self.upper_boundary - self.lower_boundary):
                termination = "threshold_step_size (upper)"
                break
            
            # update x_mean
            index = np.argsort(Y)
            X, Y, Z = X[index, :], Y[index], Z[index, :]
            x_mean_bak = x_mean
            x_mean = np.mean(X[0 : self.n_parents, :], 0)
            
            # update main vector
            # ((x_mean - x_mean_bak) / sigma) == mean of (Z + Z1 * w_v * v)
            p_m_bak = p_m
            p_m = (1 - self.c_m) * p_m + c_u_m * ((x_mean - x_mean_bak) / sigma)
            v = (1 - self.c_v) * math.copysign(1, np.dot(v, p_m_bak)) * v + self.c_v * p_m
            
            # update step-size
            p_sigma = (1 - self.c_sigma) * p_sigma + c_u_sigma * np.mean(Z[0 : self.n_parents, :], 0)
            sigma = sigma * np.exp((np.linalg.norm(p_sigma) - self.chi_N) / (self.d_sigma * self.chi_N))
        
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
            "x_mean": x_mean,
            "v": v,
            "step_size": sigma}
        return results
