""" 
Intervention on root cause 
Only interventional data!!
"""
from data_generation import *

if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "intervB"
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
    create_dataset(opt, obs=False, targets=[1], nonlinear=lambda x: x**2)

    # make sure everything went right
    inspect(opt.exp_dir)


# TODO generate some interventional data on A!
# TODO generate some interventional data on B!