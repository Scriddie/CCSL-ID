""" Create a 3-chain to see what zombie threshold can do for pruning? """

""" 
Intervention on root cause
Joint standardization attempt
"""
from data_generation import *
import numpy as np


if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "3chain"
    opt.out_dir = "src/seq/data/" + opt.exp_name
    # data
    opt.noise = 'gauss'
    opt.noise_means = (0., 0.)
    opt.noise_variance = (1, 1)
    opt.edge_weight_range = (1, 1)
    # opt.W_true = np.array([[0, 1],
                        #    [0, 0]])
    opt.W_true = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
    opt.n_nodes = len(opt.W_true)
    opt.n_obs = 10000
    opt.random_seed = 0
    opt.standardize_individually = False

    # create dataset
    create_dataset(opt, obs=True, targets=[1], 
        # nonlinear=lambda x: -0.05*x**4 + x**2
        transformation=[
            lambda x: x,
            lambda x: 3*np.sin(x-3),
            lambda x: x**2
        ]
    )

    # make sure everything went right
    inspect(opt.out_dir)

# TODO we don''t seem to have any obs data for the moment?
# TODO go check if this makes sense!
# TODO see how/what zombie threshold actually does