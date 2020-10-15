import os
import time
import numpy as np
import torch
import gym
import pickle


# not show warning information
#   https://github.com/openai/gym/blob/master/gym/logger.py
gym.logger.setLevel(40)
torch.set_default_tensor_type(torch.DoubleTensor)


def _get_policy(n_inputs, n_outputs):
    policy = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, n_outputs),
        torch.nn.Tanh(),
    )
    w = np.array([]) # flatten policy weights
    separators = dict()
    for k, v in policy.state_dict().items():
        part_w = v.numpy().flatten()
        separators[k] = np.arange(w.size, w.size + part_w.size)
        w = np.concatenate((w, part_w))
    ndim_problem = w.size
    return policy, separators, ndim_problem

def _get_actions(observation, policy):
    actions = policy(torch.from_numpy(observation))
    return actions.detach().numpy()

def _cost_func(new_w, env, len_episode, policy, separators):
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
    return -rewards # negate for minimization

class Experiment(object):
    """Experiment to support one optimizer on one environment."""
    def __init__(self, params):
        # set environment parameters
        self.env_name = params["env_name"]
        self.env_seed_train = params["env_seed_train"]
        self.env_seed_test = params["env_seed_test"]
        self.env_len_episode = params["env_len_episode"]

        self.env = gym.make(self.env_name)
        self.env_ndim_observation =\
            self.env.observation_space.shape[0]
        self.env_ndim_actions = self.env.action_space.shape[0]

        # set problem (policy) parameters
        self.lower_boundary = params["problem_lower_boundary"]
        self.upper_boundary = params["problem_upper_boundary"]

        self.policy, self.separators, self.ndim_problem =\
            _get_policy(self.env_ndim_observation,
                self.env_ndim_actions)
        self.problem = {"ndim_problem": self.ndim_problem,
            "lower_boundary":\
                self.lower_boundary * np.ones(self.ndim_problem),
            "upper_boundary":\
                self.upper_boundary * np.ones(self.ndim_problem)}
        
        # set optimizer options
        self.optimizer_options = params["optimizer_options"]
        self.optimizer = params["optimizer"] # class

        # set experiment parameters
        self.n_test = params["n_test"]
        self.datafile_suffix = params["datafile_suffix"]

        self.data_dir = "./env_{}_opt_{}".format( # save results
            self.env_name, self.optimizer.__name__)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.datafile_prefix = "env_{}_opt_{}_maxeval_{}".format(
            self.env_name, self.optimizer.__name__,
            self.optimizer_options["max_evaluations"])
        
        self.optimizer_options["txt_best_so_far_x"] =\
            "{}/{}_history_best_so_far_x_{}.txt".format(
                self.data_dir, self.datafile_prefix,
                self.datafile_suffix)
    
    def _save_train_results(self, results):
        np.savetxt("{}/{}_best_so_far_x_{}.txt".format(
            self.data_dir, self.datafile_prefix,
            self.datafile_suffix), results["best_so_far_x"])
        np.savetxt("{}/{}_fitness_data_{}.txt".format(
            self.data_dir, self.datafile_prefix,
            self.datafile_suffix), results["fitness_data"])
        results_file = "{}/{}_results_{}.pickle".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix)
        with open(results_file, "wb") as results_handle:
            pickle.dump(results, results_handle,
                protocol=pickle.HIGHEST_PROTOCOL)

    def _load_best_so_far_x(self):
        return np.loadtxt("{}/{}_best_so_far_x_{}.txt".format(
            self.data_dir, self.datafile_prefix,
            self.datafile_suffix))
    
    def _save_test_results(self, rewards):
        np.savetxt("{}/{}_rewards_{}.txt".format(
            self.data_dir, self.datafile_prefix,
            self.datafile_suffix), rewards)

    def train(self, is_test=False):
        # set the cost function (to be minimized)
        env, len_episode = self.env, self.env_len_episode
        if not is_test:
            env_seed = self.env_seed_train
        else:
            env_seed = self.env_seed_test
        policy, separators = self.policy, self.separators
        def _cost(w):
            if not hasattr(_cost, "rng"):
                _cost.rng = np.random.default_rng(env_seed)
            int32_max = np.iinfo(np.int32).max
            env.seed(int(_cost.rng.integers(0, int32_max)))
            return _cost_func(w, env, len_episode, policy, separators)
        
        if not is_test:
            # optimize the cost function
            solver = self.optimizer(
                self.problem, self.optimizer_options)
            results = solver.optimize(_cost)
            self._save_train_results(results)
        else:
            best_so_far_x = self._load_best_so_far_x()
            rewards = np.empty((self.n_test,))
            for i in range(self.n_test):
                rewards[i] = -1.0 * _cost(best_so_far_x)
            self._save_test_results(rewards)

    def test(self):
        self.train(is_test=True)


if __name__ == "__main__":
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser()

    # set environment parameters
    parser.add_argument('--env_name', '-en', type=str)
    #  when --env_seed is set, --env_seed_train and --env_seed_test
    #    will be automatically set (for simplicity)
    parser.add_argument('--env_seed', '-es',
        type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_seed_train', '-estr',
        type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_seed_test', '-este',
        type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_len_episode', '-ele',
        type=int, default=1000)

    # set problem parameters
    parser.add_argument('--problem_lower_boundary', '-plb',
        type=float, default=-0.1)
    parser.add_argument('--problem_upper_boundary', '-pub',
        type=float, default=0.1)
    
    # set optimizer options
    parser.add_argument('--optimizer_name', '-on', type=str)

    parser.add_argument('--max_evaluations', '-me',
        type=int, default=50000)
    #  when --env_seed is set, --optimizer_seed
    #    will be automatically set (for simplicity)
    parser.add_argument('--optimizer_seed', '-os',
        type=int, default=-1) # should set it as a nonnegative int
    
    # set experiment parameters
    parser.add_argument('--n_test', '-nt',
        type=int, default=100)
    parser.add_argument('--datafile_suffix', '-ds',
        type=str, default="_1")
    
    # parse all parameters
    args = parser.parse_args()
    params = vars(args)

    optimizer_name = params["optimizer_name"]
    if optimizer_name == 'PureRandomSearch':
        from prs import PureRandomSearch as Optimizer
    elif optimizer_name == 'SimpleRandomSearch':
        from srs import SimpleRandomSearch as Optimizer
    elif optimizer_name == 'GENITOR':
        from genitor import GENITOR as Optimizer
    elif optimizer_name == 'Rechenberg':
        from rechenberg import Rechenberg as Optimizer
    elif optimizer_name == 'Schwefel':
        from schwefel import Schwefel as Optimizer
    elif optimizer_name == 'Ostermeier':
        from ostermeier import Ostermeier as Optimizer
    elif optimizer_name == 'RankOne':
        from r1 import RankOne as Optimizer
    elif optimizer_name == 'Rm':
        from rm import Rm as Optimizer
    elif optimizer_name == 'SDA':
        from sda import SDA as Optimizer
    elif optimizer_name == 'RestartRankOne':
        from rr1 import RestartRankOne as Optimizer
    elif optimizer_name == 'RestartRm':
        from rrm import RestartRm as Optimizer
    elif optimizer_name == 'RestartSDA':
        from rsda import RestartSDA as Optimizer
    else:
        raise ValueError('cannot find {}.'.format(optimizer_name))
    params["optimizer"] = Optimizer
    
    optimizer_options = {
        "max_evaluations": params.pop("max_evaluations"),
        "seed": params.pop("optimizer_seed"),
        "save_best_so_far_x": True,
        "step_size": 0.3 * (params["problem_upper_boundary"] -
            params["problem_lower_boundary"])}
    params["optimizer_options"] = optimizer_options
    
    if params["env_seed"] >= 0: # for simplicity
        rng = np.random.default_rng(params["env_seed"])
        # override three seeds even if they have been set
        params["env_seed_train"], params["env_seed_test"],\
            params["optimizer_options"]["seed"] = rng.integers(
                0, np.iinfo(np.int32).max, (3,))
    
    # conduct experiment for one optimizer on one environment
    experiment = Experiment(params)
    print("******* Environment [{}] + Optimizer [{}]:".format(
        experiment.env_name, experiment.optimizer.__name__))
    print("  * env_ndim_observation: {},".format(
        experiment.env_ndim_observation))
    print("  * env_ndim_actions: {},".format(
        experiment.env_ndim_actions))
    print("  * ndim_problem: {}, ".format(experiment.ndim_problem))
    print("  * env_seed_train: {},".format(experiment.env_seed_train))
    print("  * env_seed_test: {},".format(experiment.env_seed_test))
    print("  * max_evaluations: {},".format(
        experiment.optimizer_options["max_evaluations"]))
    print("  * optimizer_seed: {},".format(
        experiment.optimizer_options["seed"]))

    train_start_time = time.time()
    experiment.train()
    print(" $$$$$ train time: {:.2e},".format(
        time.time() - train_start_time))
    
    test_start_time = time.time()
    experiment.test()
    print(" $$$$$ test time: {:.2e},".format(
        time.time() - test_start_time))
    
    print("$$$$$$$ Total Runtime: {:.2e}.".format(
        time.time() - start_time))
