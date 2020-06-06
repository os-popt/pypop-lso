import time
from datetime import datetime
import copy
import numpy as np
import gym


def _linear_policy(observation, ndim_actions, weights):
    weights = np.asarray(weights)
    # reshape 'weights' to make each column represent 
    # a mapping from 'observation' to an action
    weights = np.reshape(weights, (-1, ndim_actions))
    actions = [np.dot(observation, weights[:, a]) for a in range(ndim_actions)]
    return np.array(actions)

def _fitness_function(weights, env, ndim_actions, episode_length):
    observation = env.reset()
    accumulated_reward = 0
    for i in range(episode_length):
        actions = _linear_policy(observation, ndim_actions, weights)
        observation, reward, done, _ = env.step(actions)
        accumulated_reward += reward
        if done:
            break
    return -accumulated_reward # negate for minimization

class ContinuousControl(object):
    def __init__(self, optimizer,
        env_names=None, env_seed=2020, episode_length=1000, lower_boundary=-100, upper_boundary=100,
        optimizer_seed=None, options={}, seed_initial_guess=20200607):
        # problem-related (based on gym environment)
        if env_names is None:
            env_names = ["Swimmer-v1", "Hopper-v1", "HalfCheetah-v1", 
                "Walker2d-v1", "Ant-v1", "Humanoid-v1"]
        self.env_names = env_names
        self.env_seed = env_seed
        rng = np.random.default_rng(env_seed)
        self.env_seed_for_train = rng.integers(
            0, np.iinfo(np.int32).max, (len(self.env_names),))
        self.env_seed_for_test = rng.integers(
            0, np.iinfo(np.int32).max, (len(self.env_names),))
        self.episode_length = episode_length
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

        # optimizer-related
        self.optimizer = optimizer
        if optimizer_seed is None:
            optimizer_seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        rng = np.random.default_rng(optimizer_seed)
        self.optimizer_seed = rng.integers(
            0, np.iinfo(np.int32).max, (len(self.env_names),))
        self.options = options
        rng = np.random.default_rng(seed_initial_guess)
        self.seed_initial_guess = rng.integers(
            0, np.iinfo(np.int32).max, (len(self.env_names),))
        self.max_runtime = 3600 * np.array([2, 2, 3, 3, 3, 4]) # seconds
    
    def train(self):
        start_train = time.time()
        for i, env_name in enumerate(self.env_names):
            start_time = time.time()

            # set parameters of problem
            env = gym.make(env_name)
            ndim_observation = env.observation_space.shape[0]
            ndim_actions = env.action_space.shape[0]
            ndim_problem = ndim_observation * ndim_actions
            problem = {"ndim_problem": ndim_problem,
                "lower_boundary": self.lower_boundary * np.ones((ndim_problem,)),
                "upper_boundary": self.upper_boundary * np.ones((ndim_problem,))}
            print("** set parameters of {}-d problem: {} with train seed {}".format(
                ndim_problem, env_name, self.env_seed_for_train[i]))
            env_seed_for_train = self.env_seed_for_train[i]
            episode_length = self.episode_length
            
            def fitness_function(weights):
                if not hasattr(fitness_function, "env_rng"):
                    fitness_function.env_rng = np.random.default_rng(env_seed_for_train)
                env.seed(int(fitness_function.env_rng.integers(0, np.iinfo(np.int32).max)))
                return _fitness_function(weights, env, ndim_actions, episode_length)
            
            # set options of optimizer
            print("* set options of optimizer: {} with seed_initial_guess {} and seed {}".format(
                self.optimizer.__name__, self.seed_initial_guess[i], self.optimizer_seed[i]))
            options = copy.deepcopy(self.options)
            options["seed"] = self.optimizer_seed[i]
            options["seed_initial_guess"] = self.seed_initial_guess[i]
            options["max_runtime"] = self.max_runtime[i]
            
            # solve
            solver = self.optimizer(problem, options)
            results = solver.optimize(fitness_function)
            best_so_far_y = results["best_so_far_y"]
            print("    best_so_far_y: {:2e} (max)".format(-best_so_far_y))
            np.savetxt("env_{}_{}_best_so_far_x.txt".format(
                env_name, self.optimizer.__name__), results["best_so_far_x"])
            np.savetxt("env_{}_{}_fitness_data.txt".format(
                env_name, self.optimizer.__name__), results["fitness_data"])
            print("$ runtime: {:.2e}.".format(time.time() - start_time))
        print("$$$ total train time: {:.2e}.".format(time.time() - start_train))
