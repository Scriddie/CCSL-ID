""" Idea: Sanity check modeling assumption by comparing structures learnt from multiple indifudually identifiable datasets of the same variables. """
import sys
sys.path.append(sys.path[0]+'/..')
import os
from argparse import Namespace
import custom_utils as utils
import mvp
import BaseMLP
import pickle as pk
import custom_utils.viz as viz
from mvp import analytic
import numpy as np


def get_opt():
    opt = Namespace()
    # plotting
    opt.plot_freq = 500
    # dirs
    opt.data_dir = os.path.join('src', 'seq', 'data', '3chain')
    opt.out_dir = os.path.join(opt.data_dir, 'exp')
    # network
    opt.n_in = 2
    opt.n_layers = 3
    opt.n_hidden = 16
    # training
    opt.foreplay_iter= 0  # only fit, no acyclicity penalty
    opt.ITER = 30000
    opt.REC_FREQ = opt.ITER/30
    opt.LR = 1e-3
    opt.mu_init = 1e-5
    opt.gamma_init = 1e-5
    opt.h_threshold = 1e-2
    opt.omega_gamma = 1e-3  # Precision to declare convergence of subproblems
    opt.omega_mu = 0.8  # Desired reduction in constraint violation
    opt.mu_mult_factor = 2
    opt.stop_crit_win = 50
    # sparsity
    opt.sparsity = 0.  # see if zombie edges provide better regularization
    opt.zombie_threshold = 0.05
    opt.max_adj_entry = 5.
    opt.indicate_missingness = False
    opt.intervention_type = 'perfect' # ['perfect', 'imperfect', 'change']
    # save opt
    utils.snap(opt, fname='exp_options.txt')
    return opt

if __name__ == '__main__':
    print('GETTING STARTED')

    # options
    opt = get_opt()

    # data and model
    dag, data, mask, regimes = mvp.read(opt)

    # # plot data
    # viz.bivariate(opt, data.values)

    # train
    model = BaseMLP.BaseMLP(
        d=opt.n_in,
        num_layers=opt.n_layers, 
        hid_dim=opt.n_hidden, 
        num_params=1,
        zombie_threshold=opt.zombie_threshold,
        intervention=True,
        intervention_type=opt.intervention_type,
        intervention_knowledge='known',
        max_adj_entry=opt.max_adj_entry,
        indicate_missingness=opt.indicate_missingness,
        num_regimes=len(np.unique(regimes))
    )
    # TODO make regimes a nice mask or something?
    log = mvp.train_nll(
        opt, 
        model, 
        data, 
        dag, 
        mask, 
        regimes=regimes, 
        loss_fn=mvp.gauss_nll
    )
    
    # save log
    with open(f'{opt.out_dir}/log.pk', 'wb') as f:
        pk.dump(log, f)
    
    # TODO causal model evaluation on sample
    # gauss_nll(data, dag, lambda x: x**2)
    # # anti-causal model evaluation on sample
    # analytic(data, dag.T, lambda x: 1/x)

# TODO put on the cluster, compare our results to theirs, separate scaling from zombie threshold

# TODO could part of the problem be that we are taking away too many inputs from the network simultaneously?

# TODO is there some connection to spectral biases?

# TODO everything looks perfect. Why are we losing the wrong edge???
# TODO only obs data, fit is not that good... density is kind of off...
# why is the distribution such nonsense when the fit looks perfect???

# TODO currently we can not even learn the simple observational case... what's the matter here?? something about regime etc.??