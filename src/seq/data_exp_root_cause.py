""" Observational setting with exponentially distributed root cause"""
from data_generation import *

if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "obs"
    opt.exp_dir = "src/seq/data/" + opt.exp_name
    # data
    opt.noise = 'gauss'
    opt.noise_means = (0., 0.)
    opt.noise_variance = (.1, .1)
    opt.edge_weight_range = (1, 1)
    opt.W_true = np.array([[0, 1],
                           [0, 0]])
    opt.n_nodes = len(opt.W_true)
    opt.n_obs = 10000
    opt.random_seed = 0

    # create dataset
    # TODO alter root cause distribution to exponential
    create_dataset(opt, targets=[], nonlinear=lambda x: x**2)

    # make sure everything went right
    inspect(opt.exp_dir)

