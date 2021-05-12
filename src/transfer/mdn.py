import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma


class GMM(nn.Module):
    def __init__(self, n_gaussians):
        super(GMM, self).__init__()
        self.n_gaussians = n_gaussians
        self.pi = torch.nn.Parameter(torch.ones(1, n_gaussians))
        self.mu = torch.nn.Parameter(torch.empty(1, n_gaussians).normal_())
        self.sigma = torch.nn.Parameter(torch.ones(1, n_gaussians))

    def forward(self, like):
        return (F.softmax(self.pi, dim=-1).expand(like.shape[0], self.pi.shape[-1]),
                self.mu.repeat(like.shape[0], 1),
                torch.exp(self.sigma).expand(like.shape[0], self.sigma.shape[-1]))