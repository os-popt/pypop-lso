import time
import numpy as np

from es import MuCommaLambda


class RankOne(MuCommaLambda):
    """Rank-One Evolution Strategy (R1-ES) for large-scale, black-box optimization.
    
    Reference
    ---------
    Li, Z. and Zhang, Q., 2017.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "RankOne (R1-ES)")
        MuCommaLambda.__init__(self, problem, options)
        # learning (changing) rate of covariance matrix
        self.c_cov = options.get("c_cov", 1 / (3 * np.sqrt(problem["ndim_problem"]) + 5))
        # learning (changing) rate of principal search direction (evolution path)
        self.c = options.get("c", 2 / (problem["ndim_problem"] + 7))
        # learning (changing) rate of cumulative rank rate
        self.c_s = options.get("c_s", 0.3)
        # target ratio for mutation strength adaptation
        self.q_star = options.get("q_star", 0.3)
        # damping factor for mutation strength adaptation
        self.d_sigma = options.get("d_sigma", 1)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()
        
        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize distribution mean
        m = MuCommaLambda._get_m(self) # distribution mean
        start_evaluation = time.time()
        y = fitness_function(m)
        time_evaluations, n_evaluations = time.time() - start_evaluation, 1
        best_so_far_x, best_so_far_y = np.copy(m), np.copy(y)
        if self.save_fitness_data: fitness_data = [y]
        else: fitness_data = None
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # set weights for parents selection
        w, w_1 = np.log(np.arange(1, self.n_parents + 1)), np.log(self.n_parents + 1)
        w = (w_1 - w) / (self.n_parents * w_1 - np.sum(w))
        # normalization factor for principal search direction adaptation
        mu_eff = 1 / np.sum(np.power(w, 2))
        
        # iterate / evolve
        termination = "max_evaluations"
        sigma = self.step_size # mutation strength (global step-size)
        p = np.zeros((self.ndim_problem,)) # principal search direction (evolution path)
        s = 0 # cumulative rank rate
        # Here introduce 5 new constant symbols to simplify code
        # constant symbols for Line 5 of Algorithm 1
        m_1, m_2 = np.sqrt(1 - self.c_cov), np.sqrt(self.c_cov)
        # constant symbols for Line 12 of Algorithm 1
        p_1, p_2 = 1 - self.c, np.sqrt(self.c * (2 - self.c) * mu_eff)
        # constant symbols for Line 13 of Algorithm 1
        RR = np.arange(1, self.n_parents * 2 + 1) # ranks for R_t, R_(t+1)
        
        X = np.empty((self.n_individuals, self.ndim_problem)) # population
        Y = np.tile(y, (self.n_individuals,)) # fitness of population
        while n_evaluations < self.max_evaluations:
            Y_bak = np.copy(Y)
            for i in range(self.n_individuals): # one generation
                z = self.rng.standard_normal((self.ndim_problem,))
                r = self.rng.standard_normal()
                # sample (Line 5 of Algorithm 1)
                X[i, :] = m + sigma * (m_1 * z + m_2 * r * p)
                start_evaluation = time.time()
                y = fitness_function(X[i, :])
                time_evaluations += (time.time() - start_evaluation)
                n_evaluations += 1
                Y[i] = y

                # update best-so-far x and y
                if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(X[i, :]), np.copy(y)
                if self.save_fitness_data: fitness_data.append(np.copy(y))
                if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                
                # check three termination criteria
                runtime = time.time() - start_optimization
                is_break, termination = MuCommaLambda._check_terminations(
                    self, n_evaluations, runtime, best_so_far_y)
                if is_break: break
            
            # check four termination criteria
            is_break, termination = MuCommaLambda._check_terminations(
                self, n_evaluations, runtime, best_so_far_y, sigma)
            if is_break: break

            # update distribution mean
            index = np.argsort(Y)
            X, Y = X[index, :], Y[index] # Line 10 of Algorithm 1
            m_bak = m
            # Line 11 of Algorithm 1
            m = np.zeros((self.ndim_problem,))
            for j in range(self.n_parents):
                m += w[j] * X[j, :]
            
            # update principal search direction (Line 12 of Algorithm 1)
            p = p_1 * p + p_2 * ((m - m_bak) / sigma)
            
            # adapt mutation strength (rank-based success rule, RSR)
            F = np.hstack((Y_bak[:self.n_parents], Y[:self.n_parents]))
            R = np.argsort(F)
            # Line 13 of Algorithm 1
            R_t, R_t1 = RR[R < self.n_parents], RR[R >= self.n_parents]
            q = np.sum(w * (R_t - R_t1)) / self.n_parents
            s = (1 - self.c_s) * s + self.c_s * (q - self.q_star)
            sigma *= np.exp(s / self.d_sigma)
        
        fitness_data, time_compression = MuCommaLambda._save_data(self, history_x, fitness_data)
        
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
            "s": s,
            "step_size": sigma}
        return results