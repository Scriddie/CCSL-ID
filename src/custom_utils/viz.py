import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from pathlib import Path
from networkx.drawing.nx_agraph import graphviz_layout
from argparse import Namespace

# -------------------MDN START------------------------------

def viz_marginal(opt, model, dgp, polarity=''):
    """ show marginal distribution """
    n = int(opt.N_VIZ)
    inp = torch.FloatTensor(n).uniform_(-10, 10).view(-1, 1)
    orig = dgp(opt.SWEEP(n), n).view(-1).numpy()

    pi, mu, sigma = model(inp)
    mixture = torch.distributions.Normal(loc=mu, scale=sigma)
    pred_vals = mixture.sample().numpy()
    pi_np = pi.numpy()
    pi_np = pi_np / np.sum(pi_np, axis=1, keepdims=True)
    pred = [np.random.choice(pred_vals[i, :], p=pi_np[i, :]) 
            for i in range(len(pred_vals))]

    sns.kdeplot(x=orig, label='original')
    sns.kdeplot(x=pred, label='predicted')
    plt.legend()
    plt.title('Marginal Distribution - ' + polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_marginal')
    plt.close()

def viz_learning_curve(opt, frames, polarity=''):
    """ plot learning curve """
    iter = np.array([i.iter_num for i in frames])
    loss = np.array([i.loss for i in frames])
    plt.plot(iter, loss, label=polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_lc.png')
    plt.close()

def viz_learning_curve(frames, polarity=''):
    """ plot learning curve """
    iter = np.array([i.iter_num for i in frames])
    loss = np.array([i.loss for i in frames])
    plt.plot(iter, loss, label=polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_lc.png')
    plt.close()

def viz_cond(opt, model, polarity, actual=None, name=''):
    """ show generative distributions before transfer """
    n = int(opt.N_VIZ)
    inp = torch.FloatTensor(n).uniform_(-10, 10).view(-1, 1)

    with torch.no_grad():
        pi, mu, sigma = model(inp)
    mixture = torch.distributions.Normal(loc=mu, scale=sigma)

    pred_vals = mixture.sample().numpy()
    pi_np = pi.numpy()
    pi_np = pi_np / np.sum(pi_np, axis=1, keepdims=True)
    pred = [np.random.choice(pred_vals[i, :], p=pi_np[i, :]) 
            for i in range(len(pred_vals))]

    plt.scatter(inp.numpy(), pred, s=.2)
    plt.title(polarity)
    plt.xlabel('input')
    plt.ylabel('prediction')
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.title('Conditional Distribution')
    plt.savefig(f'{opt.FIGPATH}/{name}{polarity}_cond.png')
    plt.close()


def viz_dgp(opt, scm, polarity):
    """ Visualize data-generating process """
    plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)
    mus = [-opt.SPAN, opt.SPAN, 0]
    colors = ['C3', 'C2', 'C0']
    labels = [r'Transfer ($\mu = -4$)', r'Transfer ($\mu = +4$)', 'Training']

    for mu, color, label in zip(mus, colors, labels):
        X = mu + 2 * torch.randn((1000, 1))
        kwargs = {'color': color, 'marker': '+', 'alpha': 0.3, 'label': label}
        if polarity == 'x2y':
            ax.scatter(X.squeeze(1).numpy(), scm(X).squeeze(1).numpy(), **kwargs)
        else:
            ax.scatter(scm(X).squeeze(1).numpy(), X.squeeze(1).numpy(),
            **kwargs)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc=4, prop={'size': 13})
    ax.set_xlabel(polarity[0].upper(), fontsize=14)
    ax.set_ylabel(polarity[-1].upper(), fontsize=14)
    plt.title('Data Generating Process')
    plt.savefig(f'{opt.FIGPATH}/{polarity}_data.png')
    plt.close()


def viz_transfer(opt, x2y_df, y2x_df):
    """ Compare transfer regret for competing models """
    x2y_df['loss'] = x2y_df['nll_cond'] + x2y_df['nll_marg']
    y2x_df['loss'] = y2x_df['nll_cond'] + y2x_df['nll_marg']
    
    # # loss components
    sns.lineplot(data=x2y_df, x='iter', y='nll_cond', label='x2y_cond')
    sns.lineplot(data=y2x_df, x='iter', y='nll_cond', label='y2x_cond')
    sns.lineplot(data=x2y_df, x='iter', y='nll_marg', label='x2y_marg')
    sns.lineplot(data=y2x_df, x='iter', y='nll_marg', label='y2x_marg')

    sns.lineplot(data=x2y_df, x='iter', y='loss', label='x2y')
    sns.lineplot(data=y2x_df, x='iter', y='loss', label='y2x')
    plt.title('Transfer Learning Adaptation')
    plt.ylabel('nll')
    plt.savefig(f'{opt.FIGPATH}/0transfer.png')
    plt.close()


def viz_cond_separate(opt, model, polarity='', name=''):
    """ show generative distributions before transfer """
    n = int(opt.N_VIZ)
    inp = torch.FloatTensor(n).uniform_(-10, 10).view(-1, 1)
    with torch.no_grad():
        pi, mu, sigma = model(inp)
    # mu
    for i in range(mu.shape[1]):
        rgba_colors = np.zeros((len(inp), 4))
        rgba_colors[:, 0:3] = colors.to_rgb('C'+str(i))
        rgba_colors[:, 3] = pi[:, i]
        plt.scatter(inp, mu[:, i], color=rgba_colors, s=.3, label=str(i))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.legend(markerscale=5)
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{name}{polarity}_cond_mu.png')
    plt.close()
    # sigma
    for i in range(sigma.shape[1]):
        rgba_colors = np.zeros((len(inp), 4))
        rgba_colors[:, 0:3] = colors.to_rgb('C'+str(i))
        rgba_colors[:, 3] = pi[:, i]
        plt.scatter(inp, sigma[:, i], color=rgba_colors, s=.3, label=str(i))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.legend(markerscale=5)
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{name}{polarity}_cond_sigma.png')
    plt.close()


# -------------------MDN END------------------------------


# -------------------SEQ START----------------------------
def learning(opt, log, W_true):
    """ viz nll loss per variable during training """
    for i in ['iter', 'losses', 'mat']:
        assert i in log.keys()
    d = list(range(len(log['losses'][0])))
    var_cols = [str(i) for i in d]

    # fit
    df = pd.concat((pd.DataFrame(log['losses'], columns=var_cols), pd.DataFrame(log['iter'], columns=['iter'])), axis=1)
    melted = pd.melt(df, id_vars=['iter'], value_vars=var_cols, value_name='NLL')
    sns.lineplot(data=melted, x='iter', y='NLL', hue='variable')
    plt.ylim(0, 5)
    plt.savefig(opt.out_dir + '/lc.png')
    plt.close('all')

    # edges
    var_cols = [str(i) for i in range(len(d)**2)]
    df = pd.concat((pd.DataFrame([np.ravel(i) for i in log['mat']], columns=var_cols), pd.DataFrame(log['iter'], columns=['Iteration'])), axis=1)
    melted = pd.melt(df, id_vars=['Iteration'], value_vars=var_cols, value_name='Edge')
    color_dict = {str(idx): 'green' if i!=0 else 'red' for idx, i in enumerate(W_true.flatten())}
    # melted['edges'] = melted['variable'].replace(color_dict)
    sns.lineplot(data=melted, x='Iteration', y='Edge', hue='variable', palette=color_dict)
    plt.axhline(y=opt.zombie_threshold)
    plt.legend('', frameon=False)
    plt.savefig(opt.out_dir + '/edges.png')
    plt.close('all')

def mat(opt, mat):
    """ adjacency matrix as heatmap """
    sns.heatmap(mat, vmin=0, vmax=1)
    plt.savefig(opt.out_dir+'/mat.png')
    plt.close('all')

def model_fit(opt, model, X):
    """ viz function fit by model """
    d = X.shape[1]
    # some uniform
    input = torch.tensor(np.concatenate(
        tuple([np.linspace(-10., 10., 30000).reshape(-1, 1) for _ in range(d)]), axis=1), dtype=torch.float32)
    # forward through model
    with torch.no_grad():
        output = model.forward(input, nomask=True)
    # plot
    fig, ax = plt.subplots(ncols=d, nrows=1, figsize=(12, 5))
    for idx, i in enumerate(output):
        sns.scatterplot(x=input[:, idx].ravel(), y=i.numpy().ravel(), ax=ax[idx], label='model', color='blue', s=1)
        ax[idx].legend()
    plt.subplots_adjust(wspace=.1)
    plt.savefig(opt.out_dir + '/fit.png')
    plt.close('all')

def conditionals(opt, model, X):
    """ viz function fit by model """
    # TODO don't plot the interventional data, won't have to be fit anyways!
    d = X.shape[1]
    # some uniform
    input = torch.zeros_like(X).uniform_(-2, 2)
    # forward through model
    with torch.no_grad():
        output = model.forward(input)
    # plot
    fig, ax = plt.subplots(ncols=d, nrows=1, figsize=(12, 5))
    for idx, i in enumerate(output):
        sns.kdeplot(data=X[:, idx], ax=ax[idx], label='gt', color='blue')
        sns.kdeplot(data=i.numpy(), ax=ax[idx], label='fit', color='orange')
        ax[idx].legend()
    plt.subplots_adjust(wspace=.1)
    plt.tight_layout()
    plt.savefig(opt.out_dir + '/dist.png')
    plt.close('all')

def bivariate(opt, X, n_obs=0):
    n = X.shape[0]
    sns.scatterplot(x=X[:n-n_obs, 0], y=X[:n-n_obs, 1])  # interventional part
    sns.scatterplot(x=X[n-n_obs:, 0], y=X[n-n_obs:, 1])  # observational part
    plt.xlabel('Cause')
    plt.ylabel('Effect')
    plt.savefig(opt.out_dir + '/bivariate.png')
    plt.close('all')
# -------------------SEQ END------------------------------



# -------------------MISCELLANEOUS START ------------------------------

def graph(graph, save=False):
    """ viz graph """
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    # G = nx.relabel_nodes(G, {i: list(graph.columns)[i] for i in list(range(len(G)))})
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, node_color="lightblue", edge_color="black",
        with_labels=True, arrows=True,)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

def heat(data, save=False):
    sns.heatmap(data)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

def marg(data, save=False):
    sns.kdeplot(data=data)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

    # nrows, ncols = 2, 5
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    # for i in range(len(M)):
    #     row = int(i/ncols)
    #     col = int(i % ncols)
    #     sns.kdeplot(data[:, i], ax=ax[row][col])
    #     print(np.mean(data[:, i]), np.var(data[:, i]))
    # plt.tight_layout()
    # plt.show()