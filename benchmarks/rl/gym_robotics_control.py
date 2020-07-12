"""
Benchmarking population-based optimizers on the following 8 gym Robotics environments
from https://openai.com/blog/ingredients-for-robotics-research/ :

    1. "FetchPickAndPlace-v1"   : nd_obs: (25,), nd_des: (3,), nd_ach: (3,), nd_act: (4,); nd_search: 100
    2. "FetchPush-v1"           : nd_obs: (25,), nd_des: (3,), nd_ach: (3,), nd_act: (4,); nd_search: 100
    3. "FetchReach-v1"          : nd_obs: (10,), nd_des: (3,), nd_ach: (3,), nd_act: (4,); nd_search: 40
    4. "FetchSlide-v1"          : nd_obs: (25,), nd_des: (3,), nd_ach: (3,), nd_act: (4,); nd_search: 100
    5. "HandManipulateBlock-v0" : nd_obs: (61,), nd_des: (7,), nd_ach: (7,), nd_act: (20,); nd_search: 1220
    6. "HandManipulateEgg-v0"   : nd_obs: (61,), nd_des: (7,), nd_ach: (7,), nd_act: (20,); nd_search: 1220
    7. "HandManipulatePen-v0"   : nd_obs: (61,), nd_des: (7,), nd_ach: (7,), nd_act: (20,); nd_search: 1220
    8. "HandReach-v0"           : nd_obs: (63,), nd_des: (15,), nd_ach: (15,), nd_act: (20,); nd_search: 1260
"""
import time
from datetime import datetime
import copy
import numpy as np
import gym
from pyvirtualdisplay import Display


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
        actions = _linear_policy(observation["observation"], ndim_actions, weights)
        observation, reward, done, _ = env.step(actions)
        accumulated_reward += reward
        if done:
            break
    return -accumulated_reward # negate for minimization

class ContinuousControl(object):
    def __init__(self, optimizer,
        env_names=None, env_seed=20200707, episode_length=1000, lower_boundary=-1, upper_boundary=1,
        optimizer_seed=None, options={}, seed_initial_guess=20200711,
        suffix_txt='', n_test=100):
        # problem-related (based on gym Robotics environment)
        if env_names is None:
            env_names = ["FetchPickAndPlace-v1",
                "FetchPush-v1",
                "FetchReach-v1",
                "FetchSlide-v1",
                "HandManipulateBlock-v0",
                "HandManipulateEgg-v0",
                "HandManipulatePen-v0",
                "HandReach-v0"]
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
        self.max_runtime = 3600 * np.array([3, 3, 2, 3, 6, 6, 6, 6]) # seconds
        self.suffix_txt = suffix_txt

        # test-related
        self.n_test = n_test
    
    def train(self):
        start_train = time.time()
        for i, env_name in enumerate(self.env_names):
            start_time = time.time()

            # set parameters of problem
            env = gym.make(env_name)
            ndim_observation = env.observation_space["observation"].shape[0]
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
            np.savetxt("env_{}_{}_best_so_far_x{}.txt".format(
                env_name, self.optimizer.__name__, self.suffix_txt), results["best_so_far_x"])
            np.savetxt("env_{}_{}_fitness_data{}.txt".format(
                env_name, self.optimizer.__name__, self.suffix_txt), results["fitness_data"])
            print("$ runtime: {:.2e}.".format(time.time() - start_time))
        print("$$$ total train time: {:.2e}.".format(time.time() - start_train))

    def test(self):
        start_test = time.time()
        rewards = np.empty((len(self.env_names), self.n_test))
        for i, env_name in enumerate(self.env_names):
            start_time = time.time()
            rng = np.random.default_rng(self.env_seed_for_test[i])
            env_seed_pool = rng.integers(0, np.iinfo(np.int32).max, (self.n_test,))
            for j in range(self.n_test):
                # set parameters of problem
                env = gym.make(env_name)
                ndim_observation = env.observation_space["observation"].shape[0]
                ndim_actions = env.action_space.shape[0]
                ndim_problem = ndim_observation * ndim_actions
                problem = {"ndim_problem": ndim_problem,
                    "lower_boundary": self.lower_boundary * np.ones((ndim_problem,)),
                    "upper_boundary": self.upper_boundary * np.ones((ndim_problem,))}
                if j == 0:
                    print("** test {}-d problem: {}".format(ndim_problem, env_name))
                elif j == (self.n_test - 1):
                    virtual_display = Display()
                    virtual_display.start()
                    env = gym.wrappers.Monitor(env, "./test_video/env_{}_{}_{}/".format(
                        env_name, self.optimizer.__name__, self.suffix_txt), force=True)
                env_seed_for_test = env_seed_pool[j]
                episode_length = self.episode_length
                weights = np.loadtxt("env_{}_{}_best_so_far_x{}.txt".format(
                    env_name, self.optimizer.__name__, self.suffix_txt))
                
                def fitness_function(weights):
                    if not hasattr(fitness_function, "env_rng"):
                        fitness_function.env_rng = np.random.default_rng(env_seed_for_test)
                    env.seed(int(fitness_function.env_rng.integers(0, np.iinfo(np.int32).max)))
                    return _fitness_function(weights, env, ndim_actions, episode_length)
                
                rewards[i, j] = -1 * fitness_function(weights) # max
                print("    {}: reward {:.2f}".format(j + 1, rewards[i, j]))
            np.savetxt("env_{}_{}_rewards{}.txt".format(
                env_name, self.optimizer.__name__, self.suffix_txt), rewards[i, :])
            print("  $ env {}: rewards # max {:.2e} # mean {:.2e} # min {:.2e} >$> runtime: {:.2e}.".format(
                env_name, np.max(rewards[i, :]), np.mean(rewards[i, :]),
                np.min(rewards[i, :]), time.time() - start_time))
        print("$$$ total test time: {:.2e}.".format(time.time() - start_test))
        return rewards

def grid_search_boundary(optimizer, boundaries=None,
        env_names=None, env_seed=20200708, episode_length=1000,
        optimizer_seed=None, options={}, seed_initial_guess=20200712,
        is_test=False):
    if boundaries is None:
        boundaries = [(-(10 ** i), (10 ** i)) for i in range(-1, 2)]
    for i, boundary in enumerate(boundaries):
        lower_boundary, upper_boundary = boundary
        options.setdefault("step_size", 0.3 * (upper_boundary - lower_boundary))
        cc = ContinuousControl(optimizer,
            env_names, env_seed, episode_length, lower_boundary, upper_boundary,
            optimizer_seed, options, seed_initial_guess,
            "___gsb_{}".format(i))
        if not is_test:
            cc.train()
        else:
            cc.test()
