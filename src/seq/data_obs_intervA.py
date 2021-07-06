""" 
Intervention on root cause 
Interventional data only!!
"""
from data_generation import *
import numpy as np

if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "obs_intervA"
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
    create_dataset(opt, obs=True, targets=[0], 
        # nonlinear=lambda x: -0.05*x**4 +x**2
        nonlinear=lambda x: 0.5*np.sin(x-3)
    )

    # make sure everything went right
    inspect(opt.exp_dir)


# TODO generate some interventional data on A!
# TODO generate some interventional data on B!