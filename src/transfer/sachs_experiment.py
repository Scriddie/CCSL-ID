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
    for iter in range(opt.ITER):
        eval_idx = np.random.randint(0, train_df.shape[0], opt.SAMPLES)
        inp = torch.FloatTensor(train_df.loc[eval_idx, cause].values).view(-1,1)
        tar = torch.FloatTensor(train_df.loc[eval_idx, effect].values).view(-1,1)
        out = train_cond(inp)
        loss = nll(out, tar)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_res['iter'].append(iter)
        train_res['nll_cond'].append(loss.item())
        train_res['nll_marg'].append(nll_train_marg)
    models['train_cond'] = train_cond
    # results
    with torch.no_grad():
        nll_train_cond = nll(train_cond(inp_eval), tar_eval).item()
    print(f'{cause} -> {effect} \t TRAIN\t NLL marg: {nll_train_marg:.3f}\t NLL cond: {nll_train_cond:.3f}\t NLL TOTAL: {nll_train_cond+nll_train_marg:.3f}')    
    
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
    for iter in range(opt.ITER):
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
        trans_res['iter'].append(iter)
    models['trans_cond'] = trans_cond
    print(f'{cause} -> {effect} \t TRANS\t NLL marg: {nll_trans_marg:.3f}\t NLL cond: {nll_trans_cond:.3f}\t NLL TOTAL: {nll_trans_cond+nll_trans_marg:.3f}')

    return models, train_res, trans_res

def viz_lc(opt, res, title):
    df = pd.DataFrame(res)
    sns.lineplot(data=df, x='iter', y='nll_cond', label='nll_cond')
    plt.legend()
    plt.title(title)
    plt.savefig(f'{opt.FIGPATH}/{title}.png')
    plt.close()

def viz_res(opt, causal_trans, anti_trans, title):
    df_causal = pd.DataFrame(causal_trans)
    df_anti = pd.DataFrame(anti_trans)
    sns.lineplot(data=df_causal, x='iter', y='nll_cond', label='causal_cond')
    sns.lineplot(data=df_causal, x='iter', y='nll_marg', label='causal_marg')
    sns.lineplot(data=df_anti, x='iter', y='nll_cond', label='anti_cond')
    sns.lineplot(data=df_anti, x='iter', y='nll_marg', label='anti_marg')
    plt.ylim((0, 100))
    plt.title(title)
    plt.savefig(f'{opt.FIGPATH}/{title}.png')
    plt.close()

def viz_data(opt, df, cause, effect, title):
    sns.jointplot(data=df, x=cause, y=effect, hue='type', alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.savefig(f'{opt.FIGPATH}/data_{title}.png')
    plt.close()


if __name__ == '__main__':
    opt = Namespace()
    # Model
    opt.CAPACITY = 16
    opt.NUM_COMPONENTS = 5
    opt.GMM_NUM_COMPONENTS = 5
    # Training
    opt.ITER = 1000
    opt.SAMPLES = 1000
    opt.EVAL_SAMPLES = 1000
    opt.LR = 1e-3
    snap(opt) 

    base_path = './data/sachs/Data Files/'

    # # PKC -> PRAF
    obs = pd.read_excel(os.path.join(base_path, '1. cd3cd28.xls'))
    int = pd.read_excel(os.path.join(base_path, '8. pma.xls'))
    cause = 'PKC'
    effect = 'PRAF'

    # # PKA -> PRAF
    # obs = pd.read_excel(os.path.join(base_path, '1. cd3cd28.xls'))
    # int = pd.read_excel(os.path.join(base_path, '9. b2camp.xls'))
    # cause = 'PKA'
    # effect = 'PRAF'

    print(f'Obs: {obs.shape}, Int: {int.shape}')
    for index, i in enumerate((obs, int)):
        i.columns = [c.upper() for c in i.columns]
        i['type'] = ['obs', 'int'][index]
    trans_df = pd.concat((obs, int), axis=0).reset_index()
    
    print(trans_df.columns)
    viz_data(opt, trans_df, cause, effect, 'causal')

    # causal
    c_models, c_train_res, c_trans_res = transfer_rwd(
        opt=opt,
        marg=gmm(opt),
        cond=mdn(opt),
        train_df=obs,
        trans_df=trans_df,
        cause=cause,
        effect=effect
    )
    viz_lc(opt, c_train_res, 'causal_train')
    viz_lc(opt, c_trans_res, 'causal_trans')
    # anti-causal
    a_models, a_train_res, a_trans_res = transfer_rwd(
        opt=opt,
        marg=gmm(opt),
        cond=mdn(opt),
        train_df=obs,
        trans_df=trans_df,
        cause=effect,
        effect=cause
    )
    viz_lc(opt, a_train_res, 'anti_train')
    viz_lc(opt, a_trans_res, 'anti_trans')

    # results viz
    viz_res(opt, c_trans_res, a_trans_res, 'comparison')


# TODO why do train and transfer lc look so different??