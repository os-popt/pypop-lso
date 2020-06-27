import time
import numpy as np

from optimizer import compress_fitness_data
from es import MuCommaLambda


class RestartRankOne(MuCommaLambda):
    """Restart-based Rank-One Evolution Strategy (R-R1-ES) for large-scale, black-box optimization."""
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "RestartRankOne (R-R1-ES)")
        MuCommaLambda.__init__(self, problem, options)
        self.threshold_step_size = 1e-10
        ndim_problem = problem["ndim_problem"]
        self.c_cov = 1 / (3 * np.sqrt(ndim_problem) + 5)
        self.c_c = 2 / (ndim_problem + 7)
        self.q_star = 0.3
        self.c_s = 0.3
        self.d_sigma = 1
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()
        
        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize
        if self._X.ndim == 1:
            m = np.copy(self._X)
        else:
            m = np.copy(self._X[0, :]) # discard all other individuals
        self._X = None # clear
        start_evaluation = time.time()
        y = fitness_function(m)
        time_evaluations = time.time() - start_evaluation
        n_evaluations = 1
        best_so_far_x = np.copy(m)
        best_so_far_y = np.copy(y)
        
        if self.save_fitness_data:
            fitness_data = [y]
        
        # iterate
        termination = "max_evaluations"
        m_c1 = np.sqrt(1 - self.c_cov)
        m_c2 = np.sqrt(self.c_cov)
        p_c1 = 1 - self.c_c
        is_restart, n_restart = True, 0
        while n_evaluations < self.max_evaluations:
            if is_restart:
                if n_restart > 0:
                    m = self.rng.uniform(self.lower_boundary,
                        self.upper_boundary, (self.ndim_problem,))
                    start_evaluation = time.time()
                    y = fitness_function(m)
                    time_evaluations += (time.time() - start_evaluation)
                    n_evaluations += 1
                    if best_so_far_y > y:
                        best_so_far_x = np.copy(m)
                        best_so_far_y = np.copy(y)
        
                    if self.save_fitness_data:
                        fitness_data.append(np.copy(y))

                    self.n_individuals = self.n_individuals * 2
                    self.n_parents = int(np.floor(self.n_individuals / 2))
                    self.d_sigma = self.d_sigma * 2
                                             
                # set weights for parents
                w = np.log(np.arange(1, self.n_parents + 1))
                w = (np.log(self.n_parents + 1) - w) / (
                    self.n_parents * np.log(self.n_parents + 1) - np.sum(w))
                mu_eff = 1 / np.sum(np.power(w, 2))
                W = np.tile(w[:, np.newaxis], (1, self.ndim_problem))
                
                p = np.zeros((self.ndim_problem,))
                Y = np.tile(y, (self.n_individuals,))
                RR = np.arange(1, self.n_parents * 2 + 1) # ranks
                p_c2 = np.sqrt(self.c_c * (2 - self.c_c) * mu_eff)
                s = 0
                sigma = self.step_size
                
                is_restart = False
            
            X = np.empty((self.n_individuals, self.ndim_problem)) # population
            Y_bak = np.copy(Y)
            for i in range(self.n_individuals):
                z = self.rng.standard_normal((self.ndim_problem,))
                r = self.rng.standard_normal()
                X[i, :] = m + sigma * (m_c1 * z + m_c2 * r * p)
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
                is_restart, n_restart = True, n_restart + 1
            
            # update distribution mean
            index = np.argsort(Y)
            X, Y = X[index, :], Y[index]
            m_bak = m
            m = np.sum(X[:self.n_parents, :] * W, 0)

            # update evolution path
            p = p_c1 * p + p_c2 * ((m - m_bak) / sigma)
            
            # update step-size
            F = np.hstack((Y_bak[:self.n_parents], Y[:self.n_parents]))
            R = np.argsort(F)
            q = np.sum(w * (RR[R < self.n_parents] -
                RR[R >= self.n_parents])) / self.n_parents
            s = (1 - self.c_s) * s + self.c_s * (q - self.q_star)
            sigma = sigma * np.exp(s / self.d_sigma)
        
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
            "m": m,
            "p": p,
            "step_size": sigma,
            "n_individuals": self.n_individuals,
            "n_parents": self.n_parents,
            "d_sigma": self.d_sigma,
            "n_restart": n_restart}
        return results
