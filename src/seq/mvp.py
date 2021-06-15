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
from transfer.gmm import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MDN(nn.Module):
    def __init__(self, n_in, n_hidden, n_gaussians):
        super(MDN, self).__init__()

        # gumbel matrix needs to be part of params!
        self.mat = GumbelAdjacency(n_in)

        self.z_h = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.d = n_in

    def forward(self, x, var_idx, mask=None):
        """ predict x from variables in parent column """
        # build interventional graph
        if mask is None:
            mask = torch.ones(x.shape[0]).reshape(-1, 1)
        # gumbel
        M = self.mat.forward(x.shape[0]) 
        parents = M[:, :, var_idx]
        x = x * parents * mask
        # prediction
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma




#-----------------------Gumbel START----------------------------------
import torch
import numpy as np
from scipy.linalg import expm

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)
    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)
        # forward: we get a hard sample; backward: differentiate gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                # expm_input = expm_input.cuda()
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output


def compute_dag_constraint(w_adj):
    """
    Compute the DAG constraint of w_adj
    :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
    """
    assert (w_adj >= 0).detach().cpu().numpy().all()
    h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
    return h


def is_acyclic(adjacency):
    """
    Return true if adjacency is a acyclic
    :param np.ndarray adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


class GumbelAdjacency(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()

    def forward(self, bs, tau=1, drawhard=True):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard) * (torch.ones(self.num_vars, self.num_vars) - torch.eye(self.num_vars))
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha) * (torch.ones(self.num_vars, self.num_vars) - torch.eye(self.num_vars))

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 4.)

#--------------------------------Gumbel END-------------------------------


def nll(pi, mu, sigma, y, mask=None):
    """ Conditional NLL of a single variable """
    if pi is None:
        pi = torch.ones(size=mu.shape[0])
    m = torch.distributions.Normal(loc=mu, scale=sigma+torch.tensor(1e-6))
    log_prob_y = m.log_prob(y.reshape(-1, 1))
    log_prob_pi_y = log_prob_y + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi_y, dim=1)
    return torch.sum(loss)


def read(opt):
    """ read in data """
    dag = np.load(os.path.join(opt.data_dir, 'DAG1.npy'))
    data = pd.read_csv(os.path.join(opt.data_dir, 'data_interv1.csv'))
    mask = np.ones(data.shape)
    # TODO check this is right!
    try:  # works if there were any interventions
        interventions = pd.read_csv(os.path.join(opt.data_dir, 'intervention1.csv'), header=None)
        for idx, val in enumerate(interventions.values.reshape(-1)):
            mask[idx, val] = 0
    except:
        print('Warning: No interventions found')
    mask = pd.DataFrame(mask, columns=data.columns)
    regimes = pd.read_csv(os.path.join(opt.data_dir, 'regime1.csv'), header=None)
    return dag, data, mask, regimes

def mdn_gauss_nll(X, model, mask):
    d = X.shape[1]
    losses = torch.zeros(d)
    if mask is not None:
        mask = (mask > 0).values.squeeze()
    for i in range(d):
        pi_mu_sigma = model.forward(X, var_idx=i)
        pi, mu, sigma = pi_mu_sigma
        y = X[:, i]
        if mask is not None:
            var_mask = mask[:, i]
            pi = pi[var_mask]
            mu = mu[var_mask]
            sigma = sigma[var_mask]
            y = y[var_mask]
        losses[i] = nll(pi, mu, sigma, y)
    loss = losses.sum()
    return loss

# TODO original DCDI version
def mlp_gauss_nll(X, model, mask):
    """ likelihood under iid gaussian assumption parametrized by MLP"""
    # TODO mlp prediction a.k.a. DCDI
    pass


def train_nll(opt, model, df, W, mask, loss_fn):
    """ 
    Train model in a given direction.
    For now only bivariate case!
    For now, no interventions yet!
    """
    X = torch.Tensor(df.values)
    W = torch.Tensor(W)

    # # marginal
    # X_marginal = torch.Tensor(df[x_name].values).unsqueeze(1)
    # marginal = GaussianMixture(opt.GMM_NUM_COMPONENTS)
    # marginal.fit(X_marginal)
    # nll_marg = nll(marginal(X_marginal), X_marginal, MX).item()

    # TODO put in augmented lagrangian logic (external function?)
    ###
    best_nll_val = np.inf
    best_lagrangian_val = np.inf

    # initialize stuff for learning loop
    aug_lagrangians = []
    aug_lagrangian_ma = [0.0] * (opt.ITER + 1)
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = [0.0] * (opt.ITER + 1)

    constraint_violation_list = []
    not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
    nlls = []  # NLL on train
    nlls_val = []  # NLL on validation
    delta_mu = np.inf
    w_adj_mode = "gumbel"

    # Augmented Lagrangian stuff
    mu = opt.mu_init
    gamma = opt.gamma_init
    mus = []
    gammas = []
    ###
    with torch.no_grad():
        full_adjacency = torch.ones((model.d, model.d)) - torch.eye(model.d)
        constraint_normalization = compute_dag_constraint(full_adjacency).item()
    delta_lag = -np.inf

    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    log = {'conditional': [], 'marginal': [], 'iter': []}
    for i in range(opt.ITER):
        # acyclicity violation
        w_adj = model.mat.get_proba()
        h = compute_dag_constraint(w_adj) / constraint_normalization
        # compute augmented langrangian
        lagrangian = gamma * h
        augmentation = h ** 2

        #loss differentiation
        optim.zero_grad() 
        nll = loss_fn(X, model, mask)
        aug_lagrangian = nll + lagrangian + 0.5 * mu * augmentation
        aug_lagrangian.backward()
        optim.step()

        # TODO can I not just do the standard ALM thing?
        # TODO need to somehow evaluate delta gamma!
        if h.item() > opt.h_threshold:
            # we have either solved the problem or gone backwards
            # TODO gone backwards part still missing
            if abs(delta_lag) < opt.omega_gamma or delta_lag > 0:
                gamma += mu * h.item()
                print("Updated gamma to {}".format(gamma))

                # Did the constraint improve sufficiently?
                constraint_violation_list.append(h.item())
                if len(constraint_violation_list) >= 2:
                    if constraint_violation_list[-1] > (constraint_violation_list[-2] * opt.omega_mu):
                        mu *= opt.mu_mult_factor
                        print("Updated mu to {}".format(mu))
        aug_lagrangian = aug_lagrangian.item()

        # compute delta for gamma
        if i % opt.stop_crit_win == 0:
            aug_lagrangians_val.append(aug_lagrangian)
        if i >= 2 * opt.stop_crit_win and i % (2 * opt.stop_crit_win) == 0:
            t0     = aug_lagrangians_val[-3]
            t_half = aug_lagrangians_val[-2]
            t1     = aug_lagrangians_val[-1]
            # if the validation loss went up and down, do not update
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_lag = -np.inf
            else:
                delta_lag = (t1 - t0) / opt.stop_crit_win
            # print('delta_lag:', delta_lag)
        else:
            delta_lag = -np.inf  # do not update gamma nor mu

        # logging
        if (i % opt.REC_FREQ == 0) or (i == (opt.ITER - 1)):
            log['conditional'].append(aug_lagrangian)
            # log['marginal'].append(nll_marg)
            log['iter'].append(i)
            print(f'Iter {i}, NLL conditional: {aug_lagrangian}\t'
                # + f'NLL marginal: {nll_marg}\t'
                # + f'NLL TOTAL: {aug_lagrangian+nll_marg}'
                )
            print(model.mat.get_proba().detach().numpy())
    print()
    return log