import time
import numpy as np

from es import MuCommaLambda


class CEM(MuCommaLambda):
    """Cross-Entropy Method (CEM) for direct policy search.

    Reference
    ---------
    Duan, Y., Chen, X., Houthooft, R., Schulman, J. and Abbeel, P., 2016, June.
    Benchmarking deep reinforcement learning for continuous control.
    In International Conference on Machine Learning (pp. 1329-1338).
    http://proceedings.mlr.press/v48/duan16.pdf

    Source Code
    -----------
    https://github.com/rll/rllab    (not updated)
    https://github.com/rlworkgroup/garage    (actively updated)
    https://github.com/rlworkgroup/garage/blob/master/src/garage/np/algos/cem.py    (source code)
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "CEM (CEM)")
        MuCommaLambda.__init__(self, problem, options)
        # fraction of the best individuals for updating mean + std of sampling distribution
        self.best_frac = options.get("best_frac", 0.05)
        # number of the best individuals for updating mean + std of sampling distribution
        self.n_best = max(1, int(self.n_individuals * self.best_frac))
        # initial std for sampling distribution
        self.init_std = options.get("init_std", 1.0)
        # std decayed for updating std of sampling distribution
        self.extra_std = options.get("extra_std", 1.0)
        # number of epochs taken to decay std for updating std of sampling distribution
        self.extra_decay_time = options.get("extra_decay_time", 100)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function

        # initialize distribution mean
        m = MuCommaLambda._get_m(self)  # distribution mean
        cur_std = self.init_std  # distribution std
        start_evaluation = time.time()
        y = fitness_function(m)
        time_evaluations, n_evaluations = time.time() - start_evaluation, 1
        best_so_far_x, best_so_far_y = np.copy(m), np.copy(y)
        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # iterate / evolve
        termination = "max_evaluations"
        n_epoch = 0

        X = np.empty((self.n_individuals, self.ndim_problem))  # population
        Y = np.tile(y, (self.n_individuals,))  # fitness of population
        while n_evaluations < self.max_evaluations:
            for i in range(self.n_individuals): # one generation / epoch
                # sample
                extra_var_mult = max(1.0 - n_epoch / self.extra_decay_time, 0)
                sample_std = np.sqrt(np.square(cur_std) + np.square(self.extra_std) * extra_var_mult)
                z = self.rng.standard_normal((self.ndim_problem,))
                X[i, :] = m + sample_std * z
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

            # check three termination criteria
            is_break, termination = MuCommaLambda._check_terminations(
                self, n_evaluations, runtime, best_so_far_y)
            if is_break: break

            # update distribution mean + std
            index = np.argsort(Y)
            X = X[index, :]
            m = np.mean(X[:self.n_best, :], axis=0)
            cur_std = np.std(X[:self.n_best, :], axis=0) # Here it is a vector rather than a scalar
            n_epoch += 1

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
                   "cur_std": cur_std,
                   "n_epoch": n_epoch}
        return results