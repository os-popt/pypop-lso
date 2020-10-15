import os
import time
import numpy as np
from scipy.linalg import toeplitz
import gym
import pickle


# not show warning information
#   https://github.com/openai/gym/blob/master/gym/logger.py
gym.logger.setLevel(40)


class ToeplitzPolicy(object):
    """Toeplitz Policy, a special neural network structure with only two hidden layers.

    Reference
    ---------
    Choromanski, K., Rowland, M., Sindhwani, V., Turner, R. and Weller, A., 2018, July.
    Structured evolution with compact architectures for scalable policy optimization.
    In International Conference on Machine Learning (pp. 970-978).
    http://proceedings.mlr.press/v80/choromanski18a.html
    https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py
    """
    def __init__(self, n_inputs, n_neurons, n_outputs):
        """
        n_inputs  : dimension of observation space
        n_neurons : number of neurons for each hidden layer (same for all hidden layers)
        n_outputs : dimension of action space
        """
        self.ob_dim = n_inputs
        self.h_dim = n_neurons
        self.ac_dim = n_outputs
        
        self.w1 = np.zeros(self.ob_dim + self.h_dim - 1)
        self.w2 = np.zeros(self.h_dim * 2 - 1)
        self.w3 = np.zeros(self.ac_dim + self.h_dim - 1,)
        self.b1 = np.zeros(self.h_dim) # bias
        self.b2 = np.zeros(self.h_dim)
        
        self.W1 = self._build_layer(self.h_dim, self.w1) # [h_dim (1-layer) * ob_dim] matrix
        self.W2 = self._build_layer(self.h_dim, self.w2) # [h_dim (2-layer) * h_dim (1-layer)] matrix
        self.W3 = self._build_layer(self.ac_dim, self.w3) # [ac_dim * h_dim (2-layer)] matrix
        
        self.params = np.concatenate([self.w1, self.b1, self.w2, self.b2, self.w3])
        self.N = len(self.params)
    
    def _build_layer(self, d, v):
        col = v[:d]
        row = v[(d - 1):]
        W = toeplitz(col, row)
        return W
    
    def set_w(self, vec):
        self.params = np.copy(vec)
        
        self.w1 = vec[:len(self.w1)]
        vec = vec[len(self.w1):]
        self.b1 = vec[:len(self.b1)]
        vec = vec[len(self.b1):]
        self.w2 = vec[:len(self.w2)]
        vec = vec[len(self.w2):]
        self.b2 = vec[:len(self.b2)]
        vec = vec[len(self.b2):]
        self.w3 = vec
        
        self.W1 = self._build_layer(self.h_dim, self.w1)
        self.W2 = self._build_layer(self.h_dim, self.w2)
        self.W3 = self._build_layer(self.ac_dim, self.w3)
        
    def evaluate(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        z1 = np.tanh(np.dot(self.W1, X) + self.b1)
        z2 = np.tanh(np.dot(self.W2, z1) + self.b2)
        return np.squeeze(np.tanh(np.dot(self.W3, z2)))

def _get_policy(n_inputs, n_neurons, n_outputs):
    policy = ToeplitzPolicy(n_inputs, n_neurons, n_outputs)
    ndim_problem = policy.N
    return policy, ndim_problem

def _get_actions(observation, policy):
    actions = policy.evaluate(observation))
    return actions

def _cost_func(new_w, env, len_episode, policy):
    policy.set_w(new_w)
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
        self.env_ndim_observation = self.env.observation_space.shape[0]
        self.env_ndim_actions = self.env.action_space.shape[0]

        # set problem (policy) parameters
        self.lower_boundary = params["problem_lower_boundary"]
        self.upper_boundary = params["problem_upper_boundary"]
        self.n_neurons = params["n_neurons"]

        self.policy, self.ndim_problem =\
            _get_policy(self.env_ndim_observation, self.n_neurons, self.env_ndim_actions)
        self.problem = {"ndim_problem": self.ndim_problem,
            "lower_boundary": self.lower_boundary * np.ones(self.ndim_problem),
            "upper_boundary": self.upper_boundary * np.ones(self.ndim_problem)}
        
        # set optimizer options
        self.optimizer_options = params["optimizer_options"]
        self.optimizer = params["optimizer"] # class

        # set experiment parameters
        self.n_test = params["n_test"]
        self.datafile_suffix = params["datafile_suffix"]

        self.data_dir = "./env_{}_opt_{}".format(self.env_name, self.optimizer.__name__) # save results
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.datafile_prefix = "env_{}_opt_{}_maxeval_{}".format(
            self.env_name, self.optimizer.__name__, self.optimizer_options["max_evaluations"])
        
        self.optimizer_options["txt_best_so_far_x"] =\
            "{}/{}_history_best_so_far_x_{}.txt".format(
                self.data_dir, self.datafile_prefix, self.datafile_suffix)
    
    def _save_train_results(self, results):
        np.savetxt("{}/{}_best_so_far_x_{}.txt".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix), results["best_so_far_x"])
        np.savetxt("{}/{}_fitness_data_{}.txt".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix), results["fitness_data"])
        results_file = "{}/{}_results_{}.pickle".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix)
        with open(results_file, "wb") as results_handle:
            pickle.dump(results, results_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_best_so_far_x(self):
        return np.loadtxt("{}/{}_best_so_far_x_{}.txt".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix))
    
    def _save_test_results(self, rewards):
        np.savetxt("{}/{}_rewards_{}.txt".format(
            self.data_dir, self.datafile_prefix, self.datafile_suffix), rewards)

    def train(self, is_test=False):
        # set the cost function (to be minimized)
        env, len_episode = self.env, self.env_len_episode
        if not is_test:
            env_seed = self.env_seed_train
        else:
            env_seed = self.env_seed_test
        policy = self.policy
        def _cost(w):
            if not hasattr(_cost, "rng"):
                _cost.rng = np.random.default_rng(env_seed)
            int32_max = np.iinfo(np.int32).max
            env.seed(int(_cost.rng.integers(0, int32_max)))
            return _cost_func(w, env, len_episode, policy)
        
        if not is_test:
            # optimize the cost function
            solver = self.optimizer(self.problem, self.optimizer_options)
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
    #  when --env_seed is set, --env_seed_train and --env_seed_test will be automatically set (for simplicity)
    parser.add_argument('--env_seed', '-es', type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_seed_train', '-estr', type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_seed_test', '-este', type=int, default=-1) # should set it as a nonnegative int
    parser.add_argument('--env_len_episode', '-ele', type=int, default=1000)

    # set problem parameters
    parser.add_argument('--problem_lower_boundary', '-plb', type=float, default=-0.1)
    parser.add_argument('--problem_upper_boundary', '-pub', type=float, default=0.1)
    parser.add_argument('--n_neurons', '-nn', type=int)
    
    # set optimizer options
    parser.add_argument('--optimizer_name', '-on', type=str)

    parser.add_argument('--max_evaluations', '-me', type=int, default=50000)
    #  when --env_seed is set, --optimizer_seed will be automatically set (for simplicity)
    parser.add_argument('--optimizer_seed', '-os', type=int, default=-1) # should set it as a nonnegative int
    
    # set experiment parameters
    parser.add_argument('--n_test', '-nt', type=int, default=100)
    parser.add_argument('--datafile_suffix', '-ds', type=str)
    
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

    optimizer_options = {"max_evaluations": params.pop("max_evaluations"),
        "seed": params.pop("optimizer_seed"),
        "save_best_so_far_x": True,
        "step_size": 0.3 * (params["problem_upper_boundary"] - params["problem_lower_boundary"])}
    params["optimizer_options"] = optimizer_options
    
    if params["env_seed"] >= 0: # for simplicity
        rng = np.random.default_rng(params["env_seed"])
        # override three seeds even if they have been set
        params["env_seed_train"], params["env_seed_test"],\
            params["optimizer_options"]["seed"] = rng.integers(0, np.iinfo(np.int32).max, (3,))
    
    # conduct experiment for one optimizer on one environment
    experiment = Experiment(params)
    print("******* Environment [{}] + Optimizer [{}]:".format(
        experiment.env_name, experiment.optimizer.__name__))
    print("  * env_ndim_observation: {},".format(experiment.env_ndim_observation))
    print("  * env_ndim_actions: {},".format(experiment.env_ndim_actions))
    print("  * ndim_problem: {}, ".format(experiment.ndim_problem))
    print("  * env_seed_train: {},".format(experiment.env_seed_train))
    print("  * env_seed_test: {},".format(experiment.env_seed_test))
    print("  * max_evaluations: {},".format(experiment.optimizer_options["max_evaluations"]))
    print("  * optimizer_seed: {},".format(experiment.optimizer_options["seed"]))

    train_start_time = time.time()
    experiment.train()
    print(" $$$$$ train time: {:.2e},".format(time.time() - train_start_time))
    
    test_start_time = time.time()
    experiment.test()
    print(" $$$$$ test time: {:.2e},".format(time.time() - test_start_time))
    
    print("$$$$$$$ Total Runtime: {:.2e}.".format(time.time() - start_time))
