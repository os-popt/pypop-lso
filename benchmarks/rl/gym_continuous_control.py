import time
from datetime import datetime
import copy
import numpy as np
import gym
import torch
from pyvirtualdisplay import Display


torch.set_default_tensor_type(torch.DoubleTensor)


def _nn_policy(observation, model):
    actions = model(torch.from_numpy(observation))
    return actions.detach().numpy()

def _fitness_function(weights, env, episode_length, model, separators):
    nn_weights = dict()
    for k, v in model.state_dict().items():
        part_nn_weights = weights[separators[k]].reshape(v.shape).copy()
        nn_weights[k] = torch.from_numpy(part_nn_weights)
    model.load_state_dict(nn_weights)

    observation = env.reset()
    accumulated_reward = 0
    for i in range(episode_length):
        actions = _nn_policy(observation, model)
        observation, reward, done, _ = env.step(actions)
        accumulated_reward += reward
        if done:
            break
    return -accumulated_reward # negate for minimization

class ContinuousControl(object):
    def __init__(self, optimizer,
        env_names=None, env_seed=2020, episode_length=1000, lower_boundary=-100, upper_boundary=100,
        optimizer_seed=None, options={}, seed_initial_guess=20200607,
        suffix_txt='', n_test=100):
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
        self.max_runtime = 3600 * 2 * np.array([2, 2, 3, 3, 3, 4]) # seconds
        self.suffix_txt = suffix_txt

        # test-related
        self.n_test = n_test
    
    def train(self):
        start_train = time.time()
        for i, env_name in enumerate(self.env_names):
            start_time = time.time()

            # set parameters of problem
            env = gym.make(env_name)
            ndim_observation = env.observation_space.shape[0]
            ndim_actions = env.action_space.shape[0]
            env_seed_for_train = self.env_seed_for_train[i]
            episode_length = self.episode_length
            
            # set parameters of artificial neural network
            n_inputs, n_hidden_layer, n_outputs =\
                ndim_observation, 2 * ndim_actions, ndim_actions
            model = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, n_hidden_layer),
                torch.nn.Tanh(),
                torch.nn.Linear(n_hidden_layer, n_outputs),
            )
            model_weights = np.array([])
            separators = dict()
            for k, v in model.state_dict().items():
                part_model_weights = v.numpy().flatten().copy()
                separators[k] = np.arange(model_weights.size,
                    model_weights.size + part_model_weights.size)
                model_weights = np.concatenate((model_weights, part_model_weights))
            
            ndim_problem = model_weights.size
            problem = {"ndim_problem": ndim_problem,
                "lower_boundary": self.lower_boundary * np.ones((ndim_problem,)),
                "upper_boundary": self.upper_boundary * np.ones((ndim_problem,))}
            print("** set parameters of {}-d problem: {} with train seed {}".format(
                ndim_problem, env_name, self.env_seed_for_train[i]))
            
            def fitness_function(weights):
                if not hasattr(fitness_function, "env_rng"):
                    fitness_function.env_rng = np.random.default_rng(env_seed_for_train)
                env.seed(int(fitness_function.env_rng.integers(0, np.iinfo(np.int32).max)))
                return _fitness_function(weights, env, episode_length, model, separators)
            
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
                ndim_observation = env.observation_space.shape[0]
                ndim_actions = env.action_space.shape[0]
                env_seed_for_test = env_seed_pool[j]
                episode_length = self.episode_length
                weights = np.loadtxt("env_{}_{}_best_so_far_x{}.txt".format(
                    env_name, self.optimizer.__name__, self.suffix_txt))
                
                # set parameters of artificial neural network
                n_inputs, n_hidden_layer, n_outputs =\
                    ndim_observation, 2 * ndim_actions, ndim_actions
                model = torch.nn.Sequential(
                    torch.nn.Linear(n_inputs, n_hidden_layer),
                    torch.nn.Tanh(),
                    torch.nn.Linear(n_hidden_layer, n_outputs),
                )
                model_weights = np.array([])
                separators = dict()
                for k, v in model.state_dict().items():
                    part_model_weights = v.numpy().flatten().copy()
                    separators[k] = np.arange(model_weights.size,
                        model_weights.size + part_model_weights.size)
                    model_weights = np.concatenate((model_weights, part_model_weights))

                ndim_problem = model_weights.size
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
                
                def fitness_function(weights):
                    if not hasattr(fitness_function, "env_rng"):
                        fitness_function.env_rng = np.random.default_rng(env_seed_for_test)
                    env.seed(int(fitness_function.env_rng.integers(0, np.iinfo(np.int32).max)))
                    return _fitness_function(weights, env, episode_length, model, separators)
                
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
        env_names=None, env_seed=2021, episode_length=1000,
        optimizer_seed=None, options={}, seed_initial_guess=20200608,
        is_test=False):
    if boundaries is None:
        boundaries = [(-(10 ** i), (10 ** i)) for i in range(-1, 3)]
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
