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
        self.c_cov = 1 / (3 * np.sqrt(problem["ndim_problem"]) + 5)
        self.c_c = 2 / (problem["ndim_problem"] + 7)
        self.q_star = 0.3
        self.c_s = 0.3
        self.d_sigma = 1
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()
        
        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize
        m = MuCommaLambda._get_m(self)
        start_evaluation = time.time()
        y = fitness_function(m)
        n_evaluations, time_evaluations = 1, time.time() - start_evaluation
        best_so_far_x, best_so_far_y = np.copy(m), np.copy(y)
        Y = np.tile(y, (self.n_individuals,)) # fitness of population
        
        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # set weights for parents
        w = np.log(np.arange(1, self.n_parents + 1))
        w = (np.log(self.n_parents + 1) - w) / (
            self.n_parents * np.log(self.n_parents + 1) - np.sum(w))
        mu_eff = 1 / np.sum(np.power(w, 2))
        W = np.tile(w[:, np.newaxis], (1, self.ndim_problem))
        
        # iterate
        termination = "max_evaluations"
        m_c1, m_c2 = np.sqrt(1 - self.c_cov), np.sqrt(self.c_cov)
        p = np.zeros((self.ndim_problem,))
        p_c1, p_c2 = 1 - self.c_c, np.sqrt(self.c_c * (2 - self.c_c) * mu_eff)
        RR = np.arange(1, self.n_parents * 2 + 1) # ranks
        s = 0
        sigma = self.step_size
        X = np.empty((self.n_individuals, self.ndim_problem)) # population
        while n_evaluations < self.max_evaluations:
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

                if self.save_fitness_data: fitness_data.append(np.copy(y))
                
                # update best-so-far x and y
                if best_so_far_y > y: best_so_far_x, best_so_far_y = np.copy(X[i, :]), np.copy(y)
                if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                
                # check three termination criteria
                is_break, termination = MuCommaLambda._check_terminations(
                    self, n_evaluations, time.time() - start_optimization, best_so_far_y)
                if is_break: break
            
            # check four termination criteria
            runtime = time.time() - start_optimization
            is_break, termination = MuCommaLambda._check_terminations(
                    self, n_evaluations, runtime, best_so_far_y)
            if is_break: break
            if sigma <= self.threshold_step_size:
                termination = "threshold_step_size (lower)"
                break
            
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
            "step_size": sigma}
        return results