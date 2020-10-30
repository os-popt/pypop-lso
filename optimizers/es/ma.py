import time
import numpy as np

from optimizer import compress_fitness_data
from es import MuCommaLambda


class MA(MuCommaLambda):
    """Matrix Adaptation Evolution Strategy (MA-ES).

    Reference
    ---------
    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115
    
    http://cma.gforge.inria.fr/purecmaes.m    (for the expectation of chi distribution)
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "MA (MA-ES)")
        MuCommaLambda.__init__(self, problem, options)
        n = self.ndim_problem
        # expectation of chi distribution ||N(0,I)|| # for (M12) in Fig. 3
        self.expectation_chi = (n ** 0.5) * (1 - 1 / (4 * n) + 1 / (21 * (n ** 2)))
        self.alpha_cov = 2

    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize distribution mean
        m = MuCommaLambda._get_m(self) # y in Fig. 3
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
        w = (np.log((self.n_individuals + 1) / 2) - w) / (
            self.n_parents * np.log((self.n_individuals + 1) / 2) - np.sum(w))
        mu_eff = 1 / np.sum(np.power(w, 2))

        # initialize transformation matrix
        s = np.zeros((self.ndim_problem,))
        M = np.diag(np.ones((self.ndim_problem,)))

        c_s = (mu_eff + 2) / (mu_eff + self.ndim_problem + 5)
        s_1, s_2 = 1 - c_s, np.sqrt(mu_eff * c_s * (2 - c_s))
        c_1 = self.alpha_cov / (np.power(self.ndim_problem + 1.3, 2) + mu_eff)
        c_w = min(1 - c_1, self.alpha_cov * (mu_eff + 1 / mu_eff - 2) / (
            np.power(self.ndim_problem + 2, 2) + self.alpha_cov * mu_eff / 2))
        d_sigma = 1 + c_s + 2 * max(0, np.sqrt((mu_eff - 1) / (self.ndim_problem + 1)) - 1)

        # iterate
        termination = "max_evaluations"
        sigma = self.step_size
        step_size_data = [sigma if self.save_step_size_data else None]
        Z = np.empty((self.n_individuals, self.ndim_problem)) # Guassian noise for mutation
        D = np.empty((self.n_individuals, self.ndim_problem)) # search directions
        X = np.empty((self.n_individuals, self.ndim_problem)) # population
        I = np.diag(np.ones((self.ndim_problem,)))
        while n_evaluations < self.max_evaluations:
            for i in range(self.n_individuals): # l in Fig. 3
                Z[i, :] = self.rng.standard_normal((self.ndim_problem,)) # z_l in Fig. 3
                D[i, :] = np.transpose(np.dot(M, Z[i, :][:, np.newaxis])) # d_l in Fig. 3
                X[i, :] = m + sigma * D[i, :]
                start_evaluation = time.time()
                y = fitness_function(X[i, :]) # f_l in Fig. 3
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
            
            
            index = np.argsort(Y)
            Z, D, Y = Z[index, :], D[index, :], Y[index]
            d_w = np.zeros((self.ndim_problem,)) # for (M9) in Fig. 3
            z_w = np.zeros((self.ndim_problem,)) # for (M10) in Fig. 3
            zzt_w = np.zeros((self.ndim_problem, self.ndim_problem)) # for (M11) in Fig. 3
            for j in range(self.n_parents):
                d_w += (w[j] * D[j, :])
                z_w += (w[j] * Z[j, :])
                zzt_w += (w[j] * np.dot(Z[j, :][:, np.newaxis], Z[j, :][np.newaxis, :]))
            
            # update distribution mean
            m += (sigma * d_w)

            # update transformation matrix
            s = s_1 * s + s_2 * z_w
            M1 = (c_1 / 2) * (np.dot(s[:, np.newaxis], s[np.newaxis, :]) - I)
            M2 = (c_w / 2) * (zzt_w - I)
            M = np.dot(M, I + M1 + M2)

            # update step-size
            sigma *= np.exp(c_s / d_sigma * (np.linalg.norm(s) / self.expectation_chi - 1))
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