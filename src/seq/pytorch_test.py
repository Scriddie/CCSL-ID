""" TODO get some einsum practice """

import torch

x = torch.ones(10, 2)
m = torch.ones(10, 2) * 1
c = torch.tile((torch.ones(2, 2)-torch.eye(2,2))*torch.tensor([1,2]),(10, 1, 1))

res = torch.einsum('ij,ij,ijk->ijk', x, m, c)