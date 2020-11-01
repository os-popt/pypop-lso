import time
import numpy as np
import scipy.stats as st

from optimizer import compress_fitness_data
from es import MuCommaLambda


class SDA(MuCommaLambda):
    """Search Direction Adaptation Evolution Strategy (SDA-ES) for large-scale, black-box optimization.
    
    Reference
    ---------
    He, X., Zhou, Y., Chen, Z., Zhang, J. and Chen, W.N., 2019.
    Large-scale evolution strategy based on search direction adaptation.
    IEEE Transactions on Cybernetics. (Early Access)
    https://ieeexplore.ieee.org/abstract/document/8781905
    
    The Matlab source code of SDA-ES is provided by the orginal author, freely available at
    https://github.com/hxyokokok/SDAES
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "SDA (SDA-ES)")
        MuCommaLambda.__init__(self, problem, options)
        # m -> number of search directions (evolution paths)
        self.n_evolution_paths = options.setdefault("n_evolution_paths", 10)
        self.c_cov = 0.4 / np.sqrt(problem["ndim_problem"])
        self.c_c = 0.25 / np.sqrt(problem["ndim_problem"])
        self.c_s = 0.3
        self.d_sigma = 1
        self.p_star = 0.05
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize distribution mean
        m = MuCommaLambda._get_m(self) # distribution mean
        start_evaluation = time.time()
        y = fitness_function(m)
        n_evaluations, time_evaluations = 1, time.time() - start_evaluation
        best_so_far_x, best_so_far_y = np.copy(m), np.copy(y)
        Y = np.tile(y, (self.n_individuals,)) # fitness of population

        if self.save_fitness_data:
            fitness_data = [y]
        if self.save_best_so_far_x:
            history_x = np.hstack((n_evaluations, best_so_far_x))
        
        # set weights for parents
        w = np.log(np.arange(1, self.n_parents + 1))
        w = (np.log(self.n_parents + 1) - w) / (
            self.n_parents * np.log(self.n_parents + 1) - np.sum(w))
        mu_eff = 1 / np.sum(np.power(w, 2))
        W = np.tile(w[:, np.newaxis], (1, self.ndim_problem))

        # initialize m search directions
        n_evolution_paths = self.n_evolution_paths
        # note that in the original paper Q is a n*m matrix (v.s. a m*n matrix here) 
        Q = np.empty((n_evolution_paths, self.ndim_problem))
        for i in range(n_evolution_paths):
            Q[i, :] = 1e-10 * self.rng.standard_normal((self.ndim_problem,))
        
        # iterate
        termination = "max_evaluations"
        sigma, step_size_data = self.step_size, [self.step_size if self.save_step_size_data else None]
        x_z1, x_z2 = np.sqrt(1 - self.c_cov), np.sqrt(self.c_cov) # Line 9 of Algorithm 1
        q_1, q_2 = 1 - self.c_c, np.sqrt(self.c_c * (2 - self.c_c)) # Line 15 of Algorithm 1
        RR = np.arange(1, self.n_individuals * 2 + 1)
        Z = 0
        Z_1, Z_2 = 1 - self.c_s, np.sqrt(self.c_s * (2 - self.c_s)) # Line 22 of Algorithm 1
        U_mean = (self.n_individuals ** 2) / 2
        U_var = np.sqrt((self.n_individuals ** 2) * (2 * self.n_individuals + 1) / 12)
        d_sigma, p_star = self.d_sigma, self.p_star
        X = np.empty((self.n_individuals, self.ndim_problem)) # population
        while n_evaluations < self.max_evaluations:
            Y_bak = np.copy(Y)
            for i in range(self.n_individuals):
                z1 = self.rng.standard_normal((self.ndim_problem,))
                z2 = self.rng.standard_normal((n_evolution_paths,))
                X[i, :] = m + sigma * (x_z1 * z1 + x_z2 * np.dot(z2, Q))
                start_evaluation = time.time()
                y = fitness_function(X[i, :])
                time_evaluations += (time.time() - start_evaluation)
                n_evaluations += 1
                Y[i] = y

                if self.save_fitness_data:
                    fitness_data.append(np.copy(y))

                # update best-so-far x and y
                if best_so_far_y > y:
                    best_so_far_x, best_so_far_y = np.copy(X[i, :]), np.copy(y)
                if self.save_best_so_far_x:
                    if not(n_evaluations % self.freq_best_so_far_x):
                        history_x = np.vstack((history_x,
                            np.hstack((n_evaluations, best_so_far_x))))

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

            # update distribution mean
            index = np.argsort(Y)
            X, Y = X[index, :], Y[index]
            m_bak = m
            m = np.sum(X[:self.n_parents, :] * W, 0)

            # update search directions
            z = np.sqrt(mu_eff) * (m - m_bak) / sigma
            for i in range(n_evolution_paths):
                Q[i, :] = q_1 * Q[i, :] + q_2 * z
                t = (np.sum(z * Q[i, :])) / (np.sum(Q[i, :] * Q[i, :]))
                z = 1 / np.sqrt(1 + t ** 2) * (z - t * Q[i, :])
            
            # update step-size
            F = np.hstack((Y, Y_bak))
            R1 = np.sum(RR[np.argsort(F) >= self.n_individuals])
            U = R1 - self.n_individuals * (self.n_individuals + 1) / 2
            Z = Z_1 * Z + Z_2 * (U - U_mean) / U_var
            sigma = sigma * np.exp((st.norm.cdf(Z) / (1 - p_star) - 1) / d_sigma)
            if self.save_step_size_data: step_size_data.append(sigma)
        
        if self.save_fitness_data:
            start_compression = time.time()
            fitness_data = compress_fitness_data(fitness_data, self.len_fitness_data)
            time_compression = time.time() - start_compression
        else:
            fitness_data, time_compression = None, None

        if self.save_best_so_far_x:
            np.savetxt(self.txt_best_so_far_x, history_x)
        
        results = {"best_so_far_x": best_so_far_x,
            "best_so_far_y": best_so_far_y,
            "n_evaluations": n_evaluations,
            "runtime": runtime,
            "fitness_data": fitness_data,
            "termination": termination,
            "time_evaluations": time_evaluations,
            "time_compression": time_compression,
            "m": m,
            "step_size": sigma,
            "step_size_data": step_size_data}
        return results