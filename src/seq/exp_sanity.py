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


def get_opt():
    opt = Namespace()
    # plotting
    opt.plot_freq = 1000
    # dirs
    opt.data_dir = os.path.join('src', 'seq', 'data', 'sachs')
    opt.out_dir = os.path.join(opt.data_dir, 'exp')
    # network
    opt.n_in = 11
    opt.n_layers = 3
    opt.n_hidden = 16
    # training
    opt.ITER = 30000
    opt.REC_FREQ = opt.ITER/30
    opt.LR = 1e-3
    opt.mu_init = 1e-4
    opt.gamma_init = 1e-4
    opt.h_threshold = 1e-2
    opt.omega_gamma = 1e-3  # Precision to declare convergence of subproblems
    opt.omega_mu = 0.8  # Desired reduction in constraint violation
    opt.mu_mult_factor = 2
    opt.stop_crit_win = 30
    # sparsity
    opt.sparsity = 0.  # see if zombie edges provide better regularization
    opt.zombie_threshold = 0.2
    # save opt
    utils.snap(opt, fname='exp_options.txt')
    return opt

if __name__ == '__main__':
    print('GETTING STARTED')

    # options
    opt = get_opt()

    # data and model
    dag, data, mask, regimes = mvp.read(opt)

    # plot data
    viz.bivariate(opt, data.values)

    # train
    model = BaseMLP.BaseMLP(
        d=opt.n_in,
        num_layers=opt.n_layers, 
        hid_dim=opt.n_hidden, 
        num_params=1,
        zombie_threshold=opt.zombie_threshold,
        intervention=True
    )
    log = mvp.train_nll(opt, model, data, dag, mask, loss_fn=mvp.gauss_nll)
    
    # save log
    with open(f'{opt.out_dir}/log.pk', 'wb') as f:
        pk.dump(log, f)
    
    # TODO causal model evaluation on sample
    # gauss_nll(data, dag, lambda x: x**2)
    # # anti-causal model evaluation on sample
    # analytic(data, dag.T, lambda x: 1/x)

# TODO big problem? standardization required for fair comparison of marginals.
# BUT: standardization results in overlap in different value ranges???

# TODO
# Zombie edges might still be killing some real ones through acyclicity constraint?
# 1) Should I remove zombie edges from acyclicity constraint as well?
# 2) Check that we are using the right data in the right way!
# 3) By the end I am losing a lot of true edges. Anything to be done about this?
# are we still increasing the penalties too fast?

# TODO put on the cluster, compare our results to theirs, separate scaling from zombie threshold