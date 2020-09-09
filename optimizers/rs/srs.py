import time
import numpy as np

from optimizer import PopulationOptimizer, compress_fitness_data


class SimpleRandomSearch(PopulationOptimizer):
    """Simple Random Search (SRS) for direct policy search.
    
    Reference
    ---------
    Rosenstein, M.T. and Barto, A.G., 2001, August.
    Robot weightlifting by direct policy search.
    In International Joint Conference on Artificial Intelligence (Vol. 17, No. 1, pp. 839-846).
    https://www.ijcai.org/Proceedings/01/IJCAI-2001-k.pdf    (pp.22-27)
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "SimpleRandomSearch (SRS)")
        PopulationOptimizer._check_n_individuals(self, options, self.__class__.__name__)
        PopulationOptimizer.__init__(self, problem, options)

        self.alpha = options.get("alpha", 0.3) # step size
        self.beta = options.get("beta", 0) # search strategy
        if options.get("step_size") is None: # sigma
            raise ValueError("the option 'step_size' is a hyperparameter which should be set or tuned in advance.")
        self.step_size = options.get("step_size")
        self.gamma = options.get("gamma", 0.99) # search decay factor 
        self.threshold_step_size = options.get("threshold_step_size", 0.01) # minimum search size 

    def optimize(self, fitness_function=None):
        start_optimization = time.time()
        
        if (fitness_function is None) and (self.fitness_function != None):
            fitness_function = self.fitness_function
        
        # initialize
        x = np.copy(self._X) # population with one individual
        self._X = None # to save memory space
        start_evaluation = time.time()
        y = fitness_function(x) # evaluate fitness of population
        time_evaluations = time.time() - start_evaluation # time used for fitness evaluations
        n_evaluations = 1 # counter of fitness evaluations
        best_so_far_x = np.copy(x) # best-so-far solution
        best_so_far_y = np.copy(y) # best-so-far fitness

        if self.save_fitness_data:
            fitness_data = [y]

        # iterate
        alpha = self.alpha
        beta = self.beta
        step_size = self.step_size
        gamma = self.gamma
        threshold_step_size = self.threshold_step_size
        termination = "max_evaluations"
        while n_evaluations < self.max_evaluations:
            # sample normally and then evaluate
            delta_x = step_size * self.rng.standard_normal((self.ndim_problem,))
            xx = x + delta_x
            start_evaluation = time.time()
            y = fitness_function(xx)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1

            if self.save_fitness_data:
                fitness_data.append(y)

            # update best-so-far x and y
            if best_so_far_y > y:
                best_so_far_x = np.copy(xx)
                best_so_far_y = np.copy(y)

            # update individual and step-size
            if self.rng.uniform(0, 1, 1) < beta:
                x = x + alpha * delta_x
            else:
                x = x + alpha * (best_so_far_x - x)

            step_size = max(gamma * step_size, threshold_step_size)
            
            # check two termination criteria
            runtime = time.time() - start_optimization
            if runtime >= self.max_runtime:
                termination = "max_runtime"
                break
            if best_so_far_y <= self.threshold_fitness:
                termination = "threshold_fitness"
                break
        
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
            "time_compression": time_compression}
        return results
