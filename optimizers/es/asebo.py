import time
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError

from es import MuCommaLambda


class ASEBO(MuCommaLambda):
    """Adaptive Sample-Efficient Blackbox Optimization (ASEBO) for direct policy search.

    Reference
    ---------
    Choromanski, K.M., Pacchiano, A., Parker-Holder, J., Tang, Y. and Sindhwani, V., 2019.
    From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization.
    In Advances in Neural Information Processing Systems, pp.10299-10309.
    https://proceedings.neurips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html

    Source Code
    -----------
    https://github.com/jparkerholder/ASEBO
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "ASEBO (ASEBO)")
        MuCommaLambda.__init__(self, problem, options)
        # number of samples/individuals for each iteration
        self.n_t = options.get("n_t", 100) # num_sensings in train.py
        # minimum of samples
        self.min_n_t = options.get("min_n_t", 10) # min in train.py
        # number of iterations of full sampling
        if options.get("iota") is None:
            raise ValueError("the option 'iota' is a hyperparameter which should be set or tuned in advance.")
        self.iota = min(options.get("iota") - 1, self.ndim_problem) # k in train.py
        # PCA threshold
        self.epsilon = options.get("epsilon", 0.995) # threshold in train.py
        # smoothing parameter
        self.sigma = options.get("sigma", 0.02)
        # decay rate of covariance matrix adaptation (lambda)
        self.gamma = options.get("gamma", 0.995) # decay in train.py
        # step-size (learning rate of Adam optimizer)
        self.eta = options.get("learning_rate", 0.02) # learning_rate in train.py
        # probability of sampling from isotropic Gaussian distribution
        self.alpha = options.get("alpha", 1)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize distribution mean
        x = MuCommaLambda._get_m(self) # distribution mean
        start_evaluation = time.time()
        y = fitness_function(x)
        time_evaluations, n_evaluations = time.time() - start_evaluation, 1
        best_so_far_x, best_so_far_y = np.copy(x), np.copy(y)
        if self.save_fitness_data: fitness_data = [y]
        if self.save_best_so_far_x: history_x = np.hstack((n_evaluations, best_so_far_x))
        else: history_x = None

        # iterate / evolve
        n_iterations = 1 # n_iter in train.py
        n_failure_cholesky = 0 # number of failure of cholesky decomposition
        G = [] # all gradients obtained during optmization
        m, v = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,)) # for Adam optimizer
        while n_evaluations < self.max_evaluations:
            if n_iterations < self.iota: # do full sampling before iota iterations
                UUT = np.zeros([self.ndim_problem, self.ndim_problem]) # covariance matrix
                n_t, alpha = self.n_t, self.alpha # n_samples in es.py
            else: # use PCA decomposition to obtain subspace
                pca = PCA()
                pca_fit = pca.fit(G) # SVD
                # take top n_t directions of maximum variance to construct covariance matrix
                n_t = max(np.argmax(np.cumsum(pca_fit.explained_variance_ratio_) > self.epsilon) + 1, self.min_n_t)
                U, U_ort = pca_fit.components_[:n_t], pca_fit.components_[n_t:]
                UUT, UUT_ort = np.matmul(U.T, U), np.matmul(U_ort.T, U_ort)
                if n_iterations == self.iota: n_t = self.n_t
            
            # sample from hybrid Gaussian distribution
            cov = ((alpha / self.ndim_problem) * np.eye(self.ndim_problem) + ((1 - alpha) / n_t) * UUT) * self.sigma
            A = np.zeros((n_t, self.ndim_problem)) # search directions (perturbation vectors)
            try:
                search_directions = cholesky(cov, check_finite=False, overwrite_a=True) # l in es.py
                for i in range(n_t):
                    try: A[i] = search_directions.dot(self.rng.standard_normal((self.ndim_problem,)))
                    except LinAlgError: A[i] = self.rng.standard_normal((self.ndim_problem,))
            except LinAlgError:
                n_failure_cholesky += 1
                for i in range(n_t): A[i] = self.rng.standard_normal((self.ndim_problem,))
            # renormalize
            A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]

            # estimate gradient via antithetic sampling
            antithetic_fitness = np.zeros((n_t, 2)) # all_rollouts in es.py
            for i in range(n_t):
                up_x = x + A[i, :]
                start_evaluation = time.time()
                up_y = fitness_function(up_x)
                time_evaluations += (time.time() - start_evaluation)
                n_evaluations += 1
                # update best-so-far x and y
                if best_so_far_y > up_y: best_so_far_x, best_so_far_y = np.copy(up_x), np.copy(up_y)
                if self.save_fitness_data: fitness_data.append(np.copy(up_y))
                if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                if n_evaluations >= self.max_evaluations: break

                down_x = x - A[i, :]
                start_evaluation = time.time()
                down_y = fitness_function(down_x)
                time_evaluations += (time.time() - start_evaluation)
                n_evaluations += 1
                # update best-so-far x and y
                if best_so_far_y > down_y: best_so_far_x, best_so_far_y = np.copy(down_x), np.copy(down_y)
                if self.save_fitness_data: fitness_data.append(np.copy(down_y))
                if self.save_best_so_far_x and not(n_evaluations % self.freq_best_so_far_x):
                    history_x = np.vstack((history_x, np.hstack((n_evaluations, best_so_far_x))))
                if n_evaluations >= self.max_evaluations: break

                antithetic_fitness[i, :] = np.array([up_y, down_y])

            if n_evaluations >= self.max_evaluations: break
            antithetic_fitness = (antithetic_fitness - np.mean(antithetic_fitness)) / (np.std(antithetic_fitness) + 1e-8)
            fitness_diff = antithetic_fitness[:, 0] - antithetic_fitness[:, 1] # m in es.py
            gradient = np.zeros((self.ndim_problem,)) # g in es.py
            for i in range(n_t):
                gradient += (A[i, :] * fitness_diff[i])
            gradient /= (2 * self.sigma)

            # adaptive exploration mechanism
            if n_iterations >= self.iota:
                alpha = np.linalg.norm(np.dot(gradient, UUT_ort)) / np.linalg.norm(np.dot(gradient, UUT))
            
            # add current gradient to G
            if n_iterations == 1: G = np.copy(gradient)
            else: G = np.vstack([self.gamma * G, gradient])
            gradient /= (np.linalg.norm(gradient) / self.ndim_problem + 1e-8)

            # update gradient via Adam optimizer
            m = 0.9 * m + (1 - 0.9) * gradient
            mt = m / (1 - 0.9 ** n_iterations)
            v = 0.999 * v + (1 - 0.999) * (gradient ** 2)
            vt = v / (1 - 0.999 ** n_iterations)
            x += (self.eta * mt / (np.sqrt(vt) + 1e-8))
            n_iterations += 1
        
        fitness_data, time_compression = MuCommaLambda._save_data(self, history_x, fitness_data)
        
        results = {"best_so_far_x": best_so_far_x,
                   "best_so_far_y": best_so_far_y,
                   "n_evaluations": n_evaluations,
                   "runtime": time.time() - start_optimization,
                   "fitness_data": fitness_data,
                   "termination": "max_evaluations",
                   "time_evaluations": time_evaluations,
                   "time_compression": time_compression,
                   "n_iterations": n_iterations,
                   "x": x,
                   "gradient": gradient,
                   "n_t": n_t,
                   "alpha": alpha,
                   "n_failure_cholesky": n_failure_cholesky,
                   "shape_G": G.shape}
        return results