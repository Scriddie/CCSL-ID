import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq.mvp import GumbelAdjacency
import numpy as np


class BaseMLP(nn.Module):
    def __init__(self, 
                 d, 
                 num_layers, 
                 hid_dim, 
                 num_params, 
                 zombie_threshold,
                 num_regimes,
                 intervention_type,
                 intervention_knowledge, 
                 indicate_missingness=False, 
                 nonlin="leaky-relu", 
                 intervention=False, 
                 custom_gumbel_init=False, 
                 max_adj_entry=np.inf,
                 start_adj_entry=5.):
        """
        :param int d: number of variables in the system
        :param int num_layers: number of hidden layers
        :param int hid_dim: number of hidden units per layer
        :param int num_params: number of parameters per conditional *outputted by MLP*
        :param str nonlin: which nonlinearity to use
        :param boolean intervention: if True, use loss that take into account interventions
        :param str intervention_type: type of intervention: perfect or imperfect
        :param str intervention_knowledge: if False, don't use the intervention targets
        :param int num_regimes: total number of regimes
        """
        super(BaseMLP, self).__init__()
        self.d = d
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.gumbel = True
        self.intervention = intervention
        self.intervention_type = intervention_type
        self.intervention_knowledge = intervention_knowledge
        self.num_regimes = num_regimes
        self.zombie_threshold = zombie_threshold
        self.max_adj_entry = torch.tensor(max_adj_entry)
        self.start_adj_entry = torch.tensor(start_adj_entry)
        self.indicate_missingness = indicate_missingness

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        # Those parameter might be learnable, but they do not depend on parents.
        self.extra_params = []

        if not self.intervention:
            print("No intervention")
            self.intervention_type = "perfect"
            self.intervention_knowledge = "known"

        # initialize current adjacency matrix
        self.adjacency = torch.ones((self.d, self.d)) - torch.eye(self.d)
        self.gumbel_adjacency = GumbelAdjacency(self.d, start_adj_entry, custom_gumbel_init)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # generate parameters
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer: inputs and missingness indicators
            if i == 0:
                in_dim = self.d * (2 if self.indicate_missingness else 1)

            # last layer
            if i == self.num_layers:
                out_dim = self.num_params

            if self.intervention_type == 'perfect':
                # generate one MLP per conditional
                self.weights.append(nn.Parameter(torch.zeros(self.d, out_dim, in_dim)))
                self.biases.append(nn.Parameter(torch.zeros(self.d, out_dim)))
                self.numel_weights += self.d * out_dim * in_dim

            # if interv are imperfect or unknown, generate 'num_regimes' MLPs per conditional
            elif self.intervention_type in ['imperfect', 'change']:
                self.weights.append(nn.Parameter(torch.zeros(self.d,
                                                             out_dim, in_dim,
                                                             self.num_regimes)))
                self.biases.append(nn.Parameter(torch.zeros(self.d, out_dim,
                                                            self.num_regimes)))
                self.numel_weights += self.d * out_dim * in_dim * self.num_regimes
            else:
                if self.intervention_type not in ['perfect', 'imperfect', 'change']:
                    raise ValueError(f'{self.intervention_type} is not a valid for intervention type')
        self.reset_params()

    def get_interv_w(self, bs, regime):
        return self.gumbel_interv_w(bs, regime)

    def adjust_params(self):
        """ make sure parameters maintain all specified conditions """
        with torch.no_grad():
            probas = self.gumbel_adjacency.get_proba()
            # prevent zombie edges using adj
            # TODO eliminate dead edges in acyclicity constraint as well!?
            self.adjacency = torch.where(probas < self.zombie_threshold, torch. zeros_like(self.adjacency), self.adjacency)
            # limit maximum entry values
            vals = self.gumbel_adjacency.log_alpha.data
            self.gumbel_adjacency.log_alpha.data = torch.where(vals>self.   max_adj_entry, self.max_adj_entry, vals)


    def forward(self, x, nomask=False, mask=None, regimes=None, mode='train'):
        assert mode in ['train', 'pretrain'], 'Unknown training mode'
        # TODO we are not using get_weights for now
        weights = self.weights
        biases = self.biases
        """
        :param x: batch_size x d
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :param mask: tensor, batch_size x d
        :param regimes: np.ndarray, shape=(batch_size,)
        :return: batch_size x d * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0

        for layer in range(self.num_layers + 1):
            # First layer, apply the mask
            if layer == 0:
                # sample M to mask MLP inputs
                adj = self.adjacency.unsqueeze(0)


                M = self.gumbel_adjacency(bs)
                if mode == 'pretrain':
                    # put random dropout into mask
                    M = torch.ones_like(M)
                    M.bernoulli_(0.5)
                    # TODO little test, encode missingess as super high values
                    # This idea doesn't really seem to fly unfortunately
                    # M = torch.where(M==0, torch.ones_like(M)*9, M)

                if self.indicate_missingness:
                    # TODO is this a good way to handle missingness??
                    # expand M to fiter mask and missingness indicator
                    M = torch.cat((M, M), axis=1)
                    # expand adj and x with missingness dummies
                    adj = torch.cat((adj, torch.ones_like(adj)), axis=1)
                    x = torch.cat((x, torch.ones_like(x)), axis=1)

                if not self.intervention or (self.intervention_type == "perfect" and self.intervention_knowledge == "known"):
                    # mask applied in loss term
                    x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
                elif self.intervention_type == "change" and self.intervention_knowledge == 'known':
                    # mask applied in loss term, but different MLP per regime
                    assert regimes is not None, 'Regime is not set!'
                    regimes = torch.from_numpy(regimes)
                    # R -> bs x d (we do not use a mask unlike them)
                    R = torch.zeros(regimes.shape[0], self.num_regimes).scatter_(1, regimes, 1)
                    x = torch.einsum('tijk, bk, bjt, ljt, bj -> bti',
                                     weights[layer], R, M, adj, x)
                    x += torch.einsum('ijk, bk -> bij', biases[layer], R)
                else:
                    raise ValueError('No such configuration')

            # 2nd layer and more
            else:
                if not self.intervention or (self.intervention_type == "perfect" and self.intervention_knowledge == "known"):
                    x = torch.einsum("tij, btj -> bti", weights[layer], x) + biases[layer]
                elif self.intervention_type == "change" and self.intervention_knowledge == 'known':
                    x = torch.einsum("tijk, bk, btj -> bti", 
                                     weights[layer], R, x)
                    x += torch.einsum('ijk, bk-> bij', biases[layer], R)
                else:
                    raise ValueError('Invalid configuration')

            # count number of zeros
            num_zero_weights += weights[layer].numel() - weights[layer].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return self.gumbel_adjacency.get_proba() * self.adjacency

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.d):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        params = []

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b'in mode:
            biases = []
            for b in self.biases:
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                if ep.requires_grad:
                    extra_params.append(ep)
            params.append(extra_params)

        return tuple(params)

    def set_parameters(self, params, mode="wbx"):
        """
        Will set only parameters with requires_grad == True
        :param params: tuple of parameter lists to set, the order should be coherent with `get_parameters`
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: None
        """
        with torch.no_grad():
            k = 0
            if 'w' in mode:
                for i, w in enumerate(self.weights):
                    w.copy_(params[k][i])
                k += 1

            if 'b' in mode:
                for i, b in enumerate(self.biases):
                    b.copy_(params[k][i])
                k += 1

            if 'x' in mode and len(self.extra_params) > 0:
                for i, ep in enumerate(self.extra_params):
                    if ep.requires_grad:
                        ep.copy_(params[k][i])
                k += 1

    def get_grad_norm(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True, simply get the .grad
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        grad_norm = 0

        if 'w' in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad ** 2)

        if 'b'in mode:
            for b in self.biases:
                grad_norm += torch.sum(b.grad ** 2)

        if 'x' in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad ** 2)

        return torch.sqrt(grad_norm)

    def save_parameters(self, exp_path, mode="wbx"):
        params = self.get_parameters(mode=mode)
        # save
        with open(os.path.join(exp_path, "params_"+mode), 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, exp_path, mode="wbx"):
        with open(os.path.join(exp_path, "params_"+mode), 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params, mode=mode)

    def get_distribution(self, density_params):
        raise NotImplementedError


# TODO think about the 'perfect' vs. 'imperfect' interventions once more!
# The 'imperfect' interventions are their code word for 'completely different relationship'
# TODO but, unlike in their reasoning, if only the relationship changes, we can still use the same mask as before! (no need to follow their funky masking business)
# (we do not really need to deal with unknown targets)