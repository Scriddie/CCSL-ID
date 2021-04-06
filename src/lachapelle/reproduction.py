# ANNs for conditional densities
# DAG adj mat is mask on NN inputs
# Encode interventions as mask; 1 is an intervention target
# For starters, use gaussian distributions as densities
# So this is MDNs all over again!
# MDNs or simple gaussians?
# each node has its own neural network for each interventional distribution!
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


class MDN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        nn.Module.__init__(self)
        # self.z_h = nn.Sequential(
        #     nn.Linear(n_in, n_hidden),
        #     nn.Tanh())
        self.z_pi = nn.Linear(n_hidden, n_out, bias=False)
        self.z_mu = nn.Linear(n_hidden, n_out, bias=False)
        self.z_sigma = nn.Linear(n_hidden, n_out, bias=False)
    
    def forward(self, x):
        # TODO dangerous hack!
        # z_h = self.z_h(x)
        z_h = x

        pi = F.softmax(self.z_pi(z_h), dim=-1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma

    # TODO use latest acyclicity constraint!


def nll(pi_mu_sigma, data):
    pi, mu, sigma = pi_mu_sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(data.view(-1, 1))
    log_prob_pi = log_prob + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi, dim=1)
    return torch.mean(loss)


def joint_nll(x, adj_mat, int_mask, NNs):
    """ 
    Minimize nll of observations as modeled by the obs/int NNs
    Args:
        nll (function): estimate nll of data given dist params
        x ([type]): n x j data
        adj_mat ([type]): j x j adjacency matrix
        int_mask ([type]): k x j intervention mask
        nets (NN): k x j neural networks
    """
    # for all interventions
    d = torch.ones(int_mask.shape[0])
    for k in range(int_mask.shape[0]):
        # for all nodes
        nll_k = 1.
        for j in range(x.shape[1]):
            data_j = x[:, j]
            parents_j = x * adj_mat[:, j].T
            # no intervention
            if int_mask[k, j] == 0:
                d_params_j = NNs[0][j](parents_j)
                d[k] *= nll(d_params_j, data_j)
            # intervention
            else:
                d_params_j = NNs[k][j](parents_j)
                d[k] *= nll(d_params_j, data_j)
    return torch.sum(d)


def train(x, adj_mat, int_mask):
    K = int_mask.shape[0]
    # adj_mat = torch.zeros(opt.d, opt.d)
    NNs = [[MDN(opt.d, opt.mdn_hidden, opt.mdn_out) for _ in range(opt.d)] for _ in range(K)]
    optimizers = [[torch.optim.Adam(NNs[k][d].parameters(), lr=opt.lr) for d in range(opt.d)] for k in range(K)]
    for episode in range(opt.n):
        for i in optimizers:
            for opti in i:
                opti.zero_grad() 
        loss = joint_nll(x, adj_mat, int_mask, NNs)
        loss.backward()
        for i in optimizers:
            for opti in i:
                opti.step()
    return adj_mat, NNs


# TODO how do we optimize adj mat here???
if __name__ == '__main__':
    opt = Namespace()
    opt.n = 1000
    opt.d = 2
    opt.mdn_hidden = 2
    opt.mdn_out = 1
    opt.lr = 1e-3
    opt.n = int(1e3)

    # TODO get some proper data
    x = torch.normal(torch.zeros(opt.n, opt.d), .1*torch.ones(opt.n, opt.d))
    x[:, 1] += 2* x[:, 0]
    int_mask = torch.FloatTensor([
        [0, 0]
    ])
    adj_mat = torch.zeros(opt.d, opt.d)
    adj_mat[0, 1] = 1.

    # TODO first get to notears formulation!
    res_mat, NNs = train(x, adj_mat, int_mask)
    print(res_mat)

    for k in range(int_mask.shape[0]):
        for d in range(opt.d):
            print()
            for key, val in NNs[k][d].state_dict().items():
                print(key, val)