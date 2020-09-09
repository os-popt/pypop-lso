import time
import numpy as np

from optimizer import PopulationOptimizer, compress_fitness_data


class GENITOR(PopulationOptimizer):
    """Genetic Reinforcement Learning (GENITOR).

    GENITOR is a steady-state genetic algorithm with
        a real-valued representation,
        a very high mutation rate (for diversity and exploration),
        and unusually small populations.
    
    Historically, GAs often involve crossover,
        but for simplicity this code did not include it.

    Reference
    ---------
    Whitley, D., Dominic, S., Das, R. and Anderson, C.W., 1993.
    Genetic reinforcement learning for neurocontrol problems.
    Machine Learning, 13(2-3), pp.259-284.
    https://link.springer.com/content/pdf/10.1007/BF00993045.pdf

    Moriarty, D.E., Schultz, A.C. and Grefenstette, J.J., 1999.
    Evolutionary algorithms for reinforcement learning.
    Journal of Artificial Intelligence Research, 11, pp.241-276.
    https://www.jair.org/index.php/jair/article/view/10240/24373

    Such, F.P., Madhavan, V., Conti, E., Lehman, J., Stanley, K.O. and Clune, J., 2017.
    Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning.
    arXiv preprint arXiv:1712.06567.
    https://arxiv.org/pdf/1712.06567.pdf
    """
    def __init__(self, problem, options):
        options.setdefault("optimizer_name", "GENITOR")
        options.setdefault("n_individuals", 50) # number of individuals
        PopulationOptimizer.__init__(self, problem, options)
    
    def optimize(self, fitness_function=None):
        start_optimization = time.time()

        if (fitness_function is None) and (self.fitness_function != None):
                fitness_function = self.fitness_function
        
        # initialize
        n_individuals = self.n_individuals
        X = np.copy(self._X) # population, each row denotes an individual
        self._X = None # to save memory space
        Y = np.inf * np.ones((n_individuals,)) # fitness of population
        time_evaluations = 0 # time of fitness evaluations
        n_evaluations = 0 # number of fitness evaluations
        best_so_far_y = np.inf # best-so-far fitness

        if self.save_fitness_data:
            fitness_data = []
        
        # iterate
        termination = "max_evaluations" # by default
        search_range = self.upper_boundary - self.lower_boundary
        while n_evaluations < self.max_evaluations:
            # select one parent (no crossover is implemented)
            if n_evaluations < n_individuals: # only for the first generation
                s = n_evaluations # index for selected parent
                x = X[s, :]
            else:
                index = np.argsort(Y)
                X, Y = X[index, :], Y[index]
                # update selection probability of each individual
                select_prob = np.arange(n_individuals, 0, -1)
                select_prob = select_prob / np.sum(select_prob)
                s = self.rng.choice(n_individuals, 1, p=select_prob)[0]
                # mutate (adding a random value with range +-10.0 is not implemented)
                x = X[s, :] + 0.03 * self.rng.uniform(-search_range, search_range)
            
            # evaluate
            start_evaluation = time.time()
            y = fitness_function(x)
            time_evaluations += (time.time() - start_evaluation)
            n_evaluations += 1

            if y < Y[s]:
                Y[s] = y
                X[s, :] = x
            
            if self.save_fitness_data:
                fitness_data.append(y)
            
            # update best-so-far individual (x) and fitness (y)
            if best_so_far_y > y:
                best_so_far_x = np.copy(x)
                best_so_far_y = np.copy(y)
            
            # check two termination criteria
            runtime = time.time() - start_optimization
            if runtime >= self.max_runtime:
                termination = "max_runtime"
                break
            if best_so_far_y <= self.threshold_fitness:
                termination = "threshold_fitness"
                break
        
        if self.save_fitness_data: # to save storage space
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
