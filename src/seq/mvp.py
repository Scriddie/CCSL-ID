"""
1. Use score equivalence to find appropriate model structure
2. Combine MDNs and DCDI
3. MVP of DCDI-G 
"""

import sys
sys.path.append(sys.path[0]+'/..')
import numpy as np
import pandas as pd
import torch
import os
from argparse import Namespace
from utils.utils import snap
from transfer.mdn import MDN
from transfer.gmm import GaussianMixture


# TODO this needs to become the likelihood of the entire model somehow!!!
# it's just the cnditional at this point...
# I need to predict both ways, just not provide all the inputs 
# plus filter out interventional at the end!
# TODO walk through this once more!
def nll(pi_mu_sigma, y, mask):
    """ Conditional NLL of a single variable """
    pi, mu, sigma = pi_mu_sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma+torch.tensor(1e-6))
    log_prob_y = m.log_prob(y)
    log_prob_pi_y = log_prob_y + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi_y, dim=1)
    loss = loss * mask
    return torch.mean(loss)


def read(opt):
    """ read in data """
    data = pd.read_csv(os.path.join(opt.data_dir, 'data_interv1.csv'))
    interventions = pd.read_csv(os.path.join(opt.data_dir, 'intervention1.csv'), header=None)
    mask = np.ones(data.shape)
    for idx, val in enumerate(interventions):
        mask[idx, val] = 0
    mask = pd.DataFrame(mask, columns=data.columns)
    regimes = pd.read_csv(os.path.join(opt.data_dir, 'regime1.csv'), header=None)
    return data, mask, regimes


def train_nll(opt, model, df, x_name, y_name, mask):
    """ 
    Train model in a given direction.
    For now only bivariate case!
    For now, no interventions yet!
    """
    X = torch.Tensor(df[x_name].values).unsqueeze(1)
    Y = torch.Tensor(df[y_name].values).unsqueeze(1)
    MX = torch.Tensor(mask[x_name].values).unsqueeze(1)
    MY = torch.Tensor(mask[y_name].values).unsqueeze(1)
    # marginal
    marginal = GaussianMixture(opt.GMM_NUM_COMPONENTS)
    marginal.fit(X)
    nll_marg = nll(marginal(X), X, MX).item()
    # conditional
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    log = {'conditional': [], 'marginal': [], 'iter': []}
    for i in range(opt.ITER):
        nll_cond = nll(model(X), Y, MY)
        optim.zero_grad()
        nll_cond.backward()
        optim.step()
        nll_cond = nll_cond.item()
        if (i % opt.REC_FREQ == 0) or (i == (opt.ITER - 1)):
            log['conditional'].append(nll_cond)
            log['marginal'].append(nll_marg)
            log['iter'].append(i)
            print(f'{x_name} -> {y_name}\t NLL conditional: {nll_cond}\t NLL marginal: {nll_marg}\t NLL TOTAL: {nll_cond+nll_marg}')
    print()
    return log


if __name__ == '__main__':
    opt = Namespace()
    opt.data_dir = os.path.join('src', 'dcdi', 'data', 'custom_data', 'sachs')
    opt.exp_dir = os.path.join('src', 'seq', 'experiments')
    opt.GMM_NUM_COMPONENTS = 5
    # MDN
    opt.n_hidden = 16
    opt.n_gaussians = 8
    # training
    opt.ITER = 5000
    opt.REC_FREQ = 1000
    opt.LR = 5e-3
    # save opt
    snap(opt)

    # data and model
    data, mask, regimes = read(opt)
    model = MDN(opt.n_hidden, opt.n_gaussians)

    # train X -> Y
    log = train_nll(opt, model, data, 'PKA', 'RAF', mask)

    # train Y -> X
    log = train_nll(opt, model, data, 'RAF', 'PKA', mask)

