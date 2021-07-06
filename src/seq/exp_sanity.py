""" Idea: Sanity check modeling assumption by comparing structures learnt from multiple indifudually identifiable datasets of the same variables. """
import sys
from matplotlib.pyplot import get
from networkx.algorithms.link_prediction import within_inter_cluster
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
    opt.plot_freq = 500
    # dirs
    opt.data_dir = os.path.join('src', 'seq', 'data', 'root_cause_exp')
    opt.exp_dir = os.path.join(opt.data_dir, 'exp') # os.path.join('src', 'seq', 'experiments')
    opt.GMM_NUM_COMPONENTS = 5
    # MDN
    opt.n_in = 2
    opt.n_hidden = 16
    opt.n_gaussians = 3
    # training
    opt.ITER = 10000
    opt.REC_FREQ = opt.ITER/20
    opt.LR = 1e-3
    opt.mu_init = 1e-4
    opt.gamma_init = 1e-4
    opt.h_threshold = 1e-2
    opt.omega_gamma = 1e-3  # Precision to declare convergence of subproblems
    opt.omega_mu = 0.9  # Desired reduction in constraint violation
    opt.mu_mult_factor = 2
    opt.stop_crit_win = 50
    # save opt
    utils.snap(opt, fname='exp_options.txt')
    return opt

if __name__ == '__main__':
    # options
    opt = get_opt()

    # data and model
    dag, data, mask, regimes = mvp.read(opt)

    # train
    model = BaseMLP.BaseMLP(
        d=opt.n_in,
        num_layers=2, 
        hid_dim=opt.n_hidden, 
        num_params=1)
    log = mvp.train_nll(opt, model, data, dag, mask, loss_fn=mvp.gauss_nll)
    
    # save log
    with open(f'{opt.exp_dir}/log.pk', 'wb') as f:
        pk.dump(log, f)

    # save learning curves for individual variables
    with open(f'{opt.exp_dir}/log.pk', 'rb') as f:
        log = pk.load(f)
    viz.learning(opt, log)
    
    # TODO show function fit by NN
    # viz.model_fit(model)

    # TODO show fits of marginal/conditional 

    # # causal model evaluation on sample
    # gauss_nll(data, dag, lambda x: x**2)
    # # anti-causal model evaluation on sample
    # analytic(data, dag.T, lambda x: 1/x)

# TODO big problem? standardization required for fair comparison of marginals.
# BUT: standardization forces overlap in different value ranges???