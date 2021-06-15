""" Idea: Sanity check modeling assumption by comparing structures learnt from multiple indifudually identifiable datasets of the same variables. """
import sys
sys.path.append(sys.path[0]+'/..')
import os
from argparse import Namespace
import custom_utils as utils
import mvp


if __name__ == '__main__':
    opt = Namespace()
    opt.data_dir = os.path.join('src', 'seq', 'data', 'sanity')
    opt.exp_dir = os.path.join('src', 'seq', 'experiments')
    opt.GMM_NUM_COMPONENTS = 5
    # MDN
    opt.n_in = 2
    opt.n_hidden = 16
    opt.n_gaussians = 3
    # training
    opt.ITER = 10000
    opt.REC_FREQ = opt.ITER/20
    opt.LR = 1e-3
    opt.mu_init = 1e-2
    opt.gamma_init = 1e-2
    opt.h_threshold = 1e-3
    opt.omega_gamma = 1e-3  # Precision to declare convergence of subproblems
    opt.omega_mu = 0.99  # Desired reduction in constraint violation
    opt.mu_mult_factor = 5
    opt.stop_crit_win = 20
    # TODO max_mu, max+gamma??
    # save opt
    utils.snap(opt)

    # data and model
    dag, data, mask, regimes = mvp.read(opt)

    # train X -> Y
    model = mvp.MDN(opt.n_in, opt.n_hidden, opt.n_gaussians)
    log = mvp.train_nll(opt, model, data, dag, mask, loss_fn=mvp.mdn_gauss_nll)
