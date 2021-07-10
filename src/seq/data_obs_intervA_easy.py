""" 
Intervention on root cause 
Interventional data only!!
"""
from data_generation import *

if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "obs_intervA_easy"
    opt.out_dir = "src/seq/data/" + opt.exp_name
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
    opt.standardize_individually = True

    # create dataset
    create_dataset(opt, obs=True, targets=[0], nonlinear=lambda x: x**2)

    # make sure everything went right
    inspect(opt.out_dir)


# TODO generate some interventional data on A!
# TODO generate some interventional data on B!