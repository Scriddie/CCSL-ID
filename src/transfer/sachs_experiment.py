import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt 
from matplotlib import colors
import torch
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
from copy import deepcopy
from argparse import Namespace
from gmm import GaussianMixture as GMM
from mdn import MDN
from meta_mvp import snap, gmm, mdn, nll


def transfer_rwd(opt, marg, cond, train_df, trans_df, cause, effect):
    """ transfer regret plot for rwd """
    models = {
        'train_marg': None,
        'trans_marg': None,
        'train_cond': None,
        'trans_cond': None
    }
    # eval sample
    eval_idx = np.random.randint(0, train_df.shape[0], opt.EVAL_SAMPLES)
    inp_eval = torch.FloatTensor(train_df.loc[eval_idx, cause].values).view(-1,1)
    tar_eval = torch.FloatTensor(train_df.loc[eval_idx, effect].values).view(-1,1)
    # marginal
    train_marg = deepcopy(marg)
    train_marg.fit(inp_eval)
    models['train_marg'] = train_marg
    with torch.no_grad():
        nll_train_marg = nll(train_marg(inp_eval), inp_eval).item()
    # conditional
    train_cond = deepcopy(cond)
    optim = torch.optim.Adam(train_cond.parameters(), lr=opt.LR)
    train_res = {'nll_cond': [], 'nll_marg': [], 'iter': []}
    for iter_num in range(opt.ITER):
        eval_idx = np.random.randint(0, train_df.shape[0], opt.SAMPLES)
        inp = torch.FloatTensor(train_df.loc[eval_idx, cause].values).view(-1,1)
        tar = torch.FloatTensor(train_df.loc[eval_idx, effect].values).view(-1,1)
        out = train_cond(inp)
        loss = nll(out, tar)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_res['nll_cond'].append(loss.item())
        train_res['nll_marg'].append(nll_train_marg)
    models['train_cond'] = train_cond
    # results
    with torch.no_grad():
        nll_train_cond = nll(train_cond(inp_eval), tar_eval).item()
    print(f'TRAIN\t NLL cond: {nll_train_cond}\t NLL marg: {nll_train_marg}\t NLL TOTAL: {nll_train_cond+nll_train_marg}')    
    
    # transfer eval
    trans_res = {'nll_cond': [], 'nll_marg': [], 'iter': []}
    eval_idx = np.random.randint(0, trans_df.shape[0], opt.EVAL_SAMPLES)
    inp_eval = torch.FloatTensor(trans_df.loc[eval_idx, cause].values).view(-1,1)
    tar_eval = torch.FloatTensor(trans_df.loc[eval_idx, effect].values).view(-1,1)
    # transfer marginal
    trans_marg = deepcopy(marg)
    trans_marg.fit(inp_eval)
    models['trans_marg'] = trans_marg
    with torch.no_grad():
        nll_trans_marg = nll(trans_marg(inp_eval), inp_eval).item()
    # transfer conditional
    trans_cond = deepcopy(cond)
    optim = torch.optim.Adam(trans_cond.parameters(), lr=opt.LR)
    for j in range(opt.ITER):
        eval_idx = np.random.randint(0, trans_df.shape[0], opt.SAMPLES)
        inp = torch.FloatTensor(trans_df.loc[eval_idx, cause].values).view(-1,1)
        tar = torch.FloatTensor(trans_df.loc[eval_idx, effect].values).view(-1,1)
        nll_trans_cond = nll(trans_cond(inp), tar)
        optim.zero_grad()
        nll_trans_cond.backward()
        optim.step()
        # eval conditional
        with torch.no_grad():
            nll_trans_cond = nll(trans_cond(inp_eval), tar_eval)
        trans_res['nll_cond'].append(nll_trans_cond.detach().item())
        trans_res['nll_marg'].append(nll_trans_marg)
        trans_res['iter'].append(j)
    models['trans_cond'] = trans_cond
    print(f'TRANS\t NLL cond: {nll_trans_cond}\t NLL marg: {nll_trans_marg}\t NLL TOTAL: {nll_trans_cond+nll_trans_marg}')

    return models, train_res, trans_res



if __name__ == '__main__':
    opt = Namespace()
    # Model
    opt.CAPACITY = 16
    opt.NUM_COMPONENTS = 5
    opt.GMM_NUM_COMPONENTS = 5
    # Training
    opt.ITER = 100
    opt.SAMPLES = 100
    opt.EVAL_SAMPLES = 100
    opt.LR = 1e-4
    snap(opt) 

    base_path = './data/sachs/Data Files/'
    obs = pd.read_excel(os.path.join(base_path, '1. cd3cd28.xls'))
    int = pd.read_excel(os.path.join(base_path, '8. pma.xls'))
    print(type(obs))
    for index, i in enumerate((obs, int)):
        i.columns = [c.upper() for c in i.columns]
        i['type'] = ['obs', 'ind'][index]
    trans_df = pd.concat((obs, int), axis=0).reset_index()
    
    print(trans_df.columns)
    # sns.scatterplot(data=trans_df, x='PKC', y='PRAF', hue='type')
    # plt.show()

    # causal
    models, train_res, trans_res = transfer_rwd(
        opt=opt,
        marg=gmm(opt),
        cond=mdn(opt),
        train_df=obs,
        trans_df=trans_df,
        cause='PKC',
        effect='PRAF'
    )

    # anti-causal
    models, train_res, trans_res = transfer_rwd(
        opt=opt,
        marg=gmm(opt),
        cond=mdn(opt),
        train_df=obs,
        trans_df=trans_df,
        cause='PRAF',
        effect='PKC'
    )

# TODO plot learning curves 
# TODO plot losses akin to synth data