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
import custom_utils.viz as viz


class MDN(nn.Module):
    def __init__(self, n_in, n_hidden, n_gaussians):
        super(MDN, self).__init__()

        # gumbel matrix needs to be part of params!
        self.gumbel_adjacency = GumbelAdjacency(n_in)

        self.z_h = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.d = n_in

    def forward(self, x, mask=None):
        # TODO this is not working yet due to missing (num_vars, out_dim, in_dim) dimensions for each layer.
        """ predict x from variables in parent column """
        # build interventional graph
        if mask is None:
            mask = torch.ones_like(x)
        # gumbel
        parents = self.gumbel_adjacency.forward(x.shape[0])
        # apply intervention and parent masks
        x = torch.einsum('ij,ij,ijk->ijk', x, mask, parents)
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
        torch.nn.init.constant_(self.log_alpha, -5.)

#--------------------------------Gumbel END-------------------------------


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


def gauss_nll(X, model, mask):
    d = X.shape[1]
    losses = torch.zeros_like(X)
    if mask is not None:
        mask = (mask > 0).values.squeeze()
    density_params = model.forward(X)
    for i in range(d):
        # estimate conditional mu
        if density_params[0].shape[1] == 1:
            mu = density_params[i]
            y = X[:, i]
            if mask is not None:
                var_mask = mask[:, i]
                mu = mu[var_mask]
                y = y[var_mask]
            Dist = torch.distributions.Normal(loc=mu, 
                scale=torch.ones_like(mu))
            log_prob_y = Dist.log_prob(y.reshape(-1, 1))
            loss = -torch.logsumexp(log_prob_y, dim=1)
            losses[:len(loss), i] = loss
        # estimate conditional pi, mu, sigma
        elif density_params[0].shape[1] == 3:
            pi, mu, sigma = density_params[i]
            y = X[:, i]
            if mask is not None:
                var_mask = mask[:, i]
                pi = pi[var_mask]
                mu = mu[var_mask]
                sigma = sigma[var_mask]
                y = y[var_mask]
            Dist = torch.distributions.Normal(loc=mu, 
                scale=sigma+torch.tensor(1e-6))
            log_prob_y = Dist.log_prob(y.reshape(-1, 1))
            log_prob_pi_y = log_prob_y + torch.log(pi)
            loss = -torch.logsumexp(log_prob_pi_y, dim=1)
            losses[:, i] = loss
    return losses


def analytic():
    """ evaluation of a given matrix and function on data """
    nll = 0.
    return nll


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
    delta_gamma = -np.inf

    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    log = {'losses': [], 'iter': [], 'mat': []}

    # TODO add a sparsity penalty
    for i in range(opt.ITER):
        w_adj = model.gumbel_adjacency.get_proba()
        # TODO eliminate zombies using model.adjacency_matrix??
        # acyclicity violation
        sparsity_penalty = opt.sparsity * torch.norm(w_adj)
        h = compute_dag_constraint(w_adj) / constraint_normalization
        # compute augmented langrangian
        lagrangian = gamma * h
        augmentation = h ** 2

        #loss differentiation
        optim.zero_grad() 
        losses = loss_fn(X, model, mask)
        nll = torch.mean(torch.sum(losses, axis=1))
        aug_lagrangian = (nll + lagrangian + 0.5 * mu * augmentation + sparsity_penalty)
        aug_lagrangian.backward()
        optim.step()

        # determine delta_gamma
        if i % opt.stop_crit_win == 0:
            # TODO they compute nll portion on validation set... 
            aug_lagrangians_val.append(aug_lagrangian.item())
    
        if i >= 2 * opt.stop_crit_win and i % (2 * opt.stop_crit_win) == 0:
            t0     = aug_lagrangians_val[-3]
            t_half = aug_lagrangians_val[-2]
            t1     = aug_lagrangians_val[-1]
            # if the validation loss went up and down, do not update
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_gamma = -np.inf
            else:
                delta_gamma = (t1 - t0) / opt.stop_crit_win
        else:
            delta_gamma = -np.inf  # do not update gamma nor mu

        # We do not have an acyclic solution yet
        constraint_violation = h.item()
        if constraint_violation > opt.h_threshold:

            # we have either solved the problem or gone backwards -> penalty up
            if abs(delta_gamma) < opt.omega_gamma or delta_gamma > 0:
                gamma += mu * constraint_violation
                print("Updated gamma to {}".format(gamma))

                # Did the constraint improve sufficiently?
                constraint_violation_list.append(constraint_violation)
                if len(constraint_violation_list) >= 2:
                    if constraint_violation_list[-1] > (constraint_violation_list[-2] * opt.omega_mu):
                        mu *= opt.mu_mult_factor
                        print("Updated mu to {}".format(mu))

        # logging
        aug_lagrangian = aug_lagrangian.item()
        if (i % int(opt.REC_FREQ) == 0) or (i == (opt.ITER - 1)):
            log['losses'].append(np.mean(losses.detach().numpy(), axis=0))
            # log['marginal'].append(nll_marg)
            log['iter'].append(i)
            print(f'Iter {i}, NLL: {aug_lagrangian}\t'
                # + f'NLL marginal: {nll_marg}\t'
                # + f'NLL TOTAL: {aug_lagrangian+nll_marg}'
                )
            mat = model.gumbel_adjacency.get_proba().detach().squeeze(0).numpy()
            print(mat)
            log['mat'].append(mat)

        # viz progress
        if i % opt.plot_freq == 0:
            viz.learning(opt, log, W)
            viz.conditionals(opt, model, X)
            viz.model_fit(opt, model, X)
            viz.mat(opt, log['mat'][-1])
            # TODO save graph as heatmap

    print()
    return log

# TODO I need to use a different set of weights and biases for each conditional!!!