import os
import time
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# NOT show warning information for gym and torch:
#   https://github.com/openai/gym/blob/master/gym/logger.py
gym.logger.setLevel(40) # ERROR = 40
torch.set_default_tensor_type(torch.DoubleTensor)


def _get_policy(n_inputs, n_outputs, is_tanh=True):
    if is_tanh:
        policy = torch.nn.Sequential( # linear policy
            torch.nn.Linear(n_inputs, n_outputs, bias=False),
            torch.nn.Tanh())
    else:
        policy = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_outputs, bias=False))
    w = np.array([]) # flatten policy weights
    separators = dict()
    for k, v in policy.state_dict().items():
        part_w = v.numpy().flatten()
        separators[k] = np.arange(w.size, w.size + part_w.size)
        w = np.concatenate((w, part_w))
    ndim_problem = w.size
    return policy, separators, ndim_problem

def _get_actions(observation, policy):
    actions = policy(torch.from_numpy(observation).double())
    return actions.detach().numpy()

def _reward_func(new_w, env, len_episode, policy, separators):
    # use new flatten weights to update policy weights
    w = dict() # policy weights
    for k, v in policy.state_dict().items():
        part_w = new_w[separators[k]].reshape(v.shape).copy()
        w[k] = torch.from_numpy(part_w)
    policy.load_state_dict(w)

    # get accumulated reward -> rewards
    observation, rewards = env.reset(), 0
    for i in range(len_episode):
        actions = _get_actions(observation, policy)
        observation, reward, done, _ = env.step(actions)
        rewards += reward
        if done: break
    return rewards

class Experiment(object):
    """Experiment to plot linear policy landscape on one environment."""
    def __init__(self, params):
        # set environment parameters
        self.env_name = params["env_name"]
        self.seed = params["seed"]
        self.env_len_episode = params["env_len_episode"]
        self.is_same_env = params["is_same_env"]

        self.env = gym.make(self.env_name)
        self.env_ndim_observation = self.env.observation_space.shape[0]
        self.env_ndim_actions = self.env.action_space.shape[0]

        # set problem (policy) parameters
        self.is_tanh = params["is_tanh"]

        self.policy, self.separators, self.ndim_problem =\
            _get_policy(self.env_ndim_observation, self.env_ndim_actions, self.is_tanh)
        self.problem = {"ndim_problem": self.ndim_problem,
            "lower_boundary": params["problem_lower_boundary"] * np.ones(self.ndim_problem),
            "upper_boundary": params["problem_upper_boundary"] * np.ones(self.ndim_problem)}
        
        # set experiment parameters
        self.n_repeat = params["n_repeat"]
        self.freq_sampling = params["freq_sampling"]

        self.data_dir = "./env_{}".format(self.env_name)
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)

    def plot(self):
        # set the reward function
        env, len_episode, is_same_env = self.env, self.env_len_episode, self.is_same_env
        rng = np.random.default_rng(self.seed)
        env_seed, sampling_seed = rng.integers(0, np.iinfo(np.int32).max, 2)
        policy, separators = self.policy, self.separators
        def _reward(w, is_same_env):
            seed = int(env_seed)
            if not hasattr(_reward, "rng"):
                _reward.rng = np.random.default_rng(seed)
            if not is_same_env:
                seed = int(_reward.rng.integers(0, np.iinfo(np.int32).max))
            env.seed(seed)
            return _reward_func(w, env, len_episode, policy, separators)
        # sample in the randomized 2-d subspace
        sampling_rng = np.random.default_rng(sampling_seed)
        x_dim, y_dim = sampling_rng.choice(self.ndim_problem, 2, replace=False)
        x = np.linspace(self.problem["lower_boundary"][x_dim], self.problem["upper_boundary"][x_dim], self.freq_sampling)
        y = np.linspace(self.problem["lower_boundary"][y_dim], self.problem["upper_boundary"][y_dim], self.freq_sampling)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        w = sampling_rng.uniform(self.problem["lower_boundary"], self.problem["upper_boundary"])
        for r in range(self.n_repeat):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    w[x_dim] = X[i, j]
                    w[y_dim] = Y[i, j]
                    Z[i, j] += _reward(w, is_same_env)
        Z /= self.n_repeat
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, cmap=plt.cm.cool, alpha=0.9, edgecolor=None)
        plt.title("Optimization Landscape (Max) on 2-d Subspace")
        ax.set_xlabel("Dimension {}".format(x_dim + 1))
        ax.set_ylabel("Dimension {}".format(y_dim + 1))
        plt.savefig(os.path.join(self.data_dir, "repeat_{}".format(self.n_repeat)))
        plt.close()

if __name__ == "__main__":
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser()

    # set environment parameters
    parser.add_argument('--env_name', '-en', type=str)
    #   should set it as a nonnegative int
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_len_episode', '-ele', type=int, default=1000)
    parser.add_argument('--is_same_env', '-ise', type=bool, default=True)

    # set problem parameters
    parser.add_argument('--problem_lower_boundary', '-plb', type=float, default=-0.1)
    parser.add_argument('--problem_upper_boundary', '-pub', type=float, default=0.1)
    parser.add_argument('--is_tanh', '-it', type=bool, default=True)
    
    # set experiment parameters
    parser.add_argument('--n_repeat', '-nr', type=int, default=1)
    parser.add_argument('--freq_sampling', '-fs', type=int, default=100)
    
    # parse all parameters
    args = parser.parse_args()
    params = vars(args)
    
    # conduct experiment on one environment
    experiment = Experiment(params)
    print("******* Environment [{}]:".format(experiment.env_name))
    print("  * env_ndim_observation: {},".format(experiment.env_ndim_observation))
    print("  * env_ndim_actions: {},".format(experiment.env_ndim_actions))
    print("  * ndim_problem: {}, ".format(experiment.ndim_problem))
    print("  * seed: {},".format(experiment.seed))

    experiment.plot()

    print("$$$$$$$ Total Runtime: {:.2e}.".format(time.time() - start_time))
