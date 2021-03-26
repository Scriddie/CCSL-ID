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
from gmm import GaussianMixture
from mdn import MDN


def nll(pi_mu_sigma, y, reduce=True):
    pi, mu, sigma = pi_mu_sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob_y = m.log_prob(y)
    log_prob_pi_y = log_prob_y + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi_y, dim=1)
    if reduce:
        return torch.mean(loss)
    else:
        return loss

class RandomSplineSCM(nn.Module): 
    def __init__(self, input_noise=False, output_noise=True, 
                 span=6, num_anchors=10, order=3, range_scale=1.): 
        super(RandomSplineSCM, self).__init__()
        self._span = span
        self._num_anchors = num_anchors
        self._range_scale = range_scale

        if opt.v_structure:
            self._x = np.array([-span, 0, span])
            self._y = np.array([span, -span, span])
        elif opt.ABLINE:
            self._x = np.array([-span, span])
            self._y = np.array([-range_scale*span, range_scale*span])
        else:
            self._x = np.linspace(-span, span, num_anchors)
            self._y = np.random.uniform(-range_scale * span, range_scale * span, size=(num_anchors,))
        
        self._spline_spec = interpolate.splrep(self._x, self._y, k=order)
        self.input_noise = input_noise
        self.output_noise = output_noise
    
    def forward(self, X, Z=None):
        if Z is None: 
            Z = self.sample(X.shape[0])
        if self.input_noise: 
            X = X + Z * self.input_noise
        X_np = X.detach().cpu().numpy().squeeze()
        _Y_np = interpolate.splev(X_np, self._spline_spec)
        _Y = torch.from_numpy(_Y_np).view(-1, 1).float().to(X.device)
        if self.output_noise:
            Y = _Y + Z * self.output_noise
        else: 
            Y = _Y
        return Y
        
    def sample(self, N): 
        return torch.normal(torch.zeros(N), torch.ones(N)).view(-1, 1)
    
    def plot(self, X, title="Samples from the SCM", label=None, show=True): 
        Y = self(X)
        if show:
            plt.figure()
            plt.title(title)
        plt.scatter(X.squeeze().numpy(), Y.squeeze().numpy(), marker='+', label=label)
        if show:
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

def mdn(opt): 
    return MDN(opt.CAPACITY, opt.NUM_COMPONENTS)

def gmm(opt): 
    return GaussianMixture(opt.GMM_NUM_COMPONENTS)

def viz_learning_curve(frames, polarity=''):
    """ plot learning curve """
    iter = np.array([i.iter_num for i in frames])
    loss = np.array([i.loss for i in frames])
    plt.plot(iter, loss, label=polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_lc.png')
    plt.close()

def viz_marginal(model, dgp, polarity=''):
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

def viz_cond(model, polarity, actual=None, name=''):
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

def viz_cond_separate(model, polarity='', name=''):
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

def polarity_hlp(polarity, X, Y):

    if opt.NOISE_X:
        N = len(X)
        X = X + torch.normal(torch.zeros(N), torch.ones(N)).view(-1, 1)*opt.NOISE_X

    if polarity == 'x2y':
        inp, tar = X, Y
    elif polarity == 'y2x':
        inp, tar = Y, X
    else:
        raise ValueError
    return inp, tar

def train_nll(opt, model, scm, train_distr_fn, polarity, 
    loss_fn):
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    frames = []
    for iter_num in range(opt.ITER):
        # Generate samples from the train distr
        X = train_distr_fn(opt.SAMPLES)
        with torch.no_grad():
            Y = scm(X)

        inp, tar = polarity_hlp(polarity, X, Y)
        # Train
        out = model(inp)
        loss = loss_fn(out, tar)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Append info
        if iter_num % opt.REC_FREQ or iter_num == (opt.ITER - 1):
            info = Namespace(loss=loss.item(),
                             iter_num=iter_num)
            frames.append(info)

    # Eval sample
    X_eval = train_distr_fn(opt.N_EVAL)
    with torch.no_grad():
        Y_eval = scm(X_eval)
    inp_eval, tar_eval = polarity_hlp(polarity, X_eval, Y_eval)
    
    # print model marginal and conditional
    marginal = gmm(opt)
    marginal.fit(inp_eval)
    with torch.no_grad():
        nll_marg = loss_fn(marginal(inp_eval), inp_eval).item()
        nll_cond = loss_fn(model(inp_eval), tar_eval).item()
    print(f'TRAIN {polarity.upper()}\t NLL conditional: {nll_cond}\t NLL marginal: {nll_marg}\t NLL TOTAL: {nll_cond+nll_marg}')
    
    return frames

def train_transfer(opt, model, scm,  polarity):
    """ reproduce adaptation plots """
    res = {'nll_cond': [], 'nll_marg': [], 'iter': [],}
    for i in tqdm(range(opt.TRANS_TASKS)):
        model = deepcopy(model)
        optim = torch.optim.Adam(model.parameters(), lr=opt.TRANS_LR)
        trans_mean = opt.SWEEP(1)
        x = opt.TRANS_DISTR(trans_mean, opt.TRANS_SAMPLES)
        with torch.no_grad():
            y = scm(x)
        x, y = polarity_hlp(polarity, x, y)
        # # eval data
        # X_eval = opt.TRANS_DISTR(trans_mean, opt.N_EVAL)
        # with torch.no_grad():
        #     Y_eval = scm(X_eval)
        # inp_eval, tar_eval = polarity_hlp(polarity, X_eval, Y_eval)
        inp_eval, tar_eval = x, y
        # marginal
        marginal = gmm(opt)
        marginal.fit(inp_eval)
        nll_marg = nll(marginal(inp_eval), inp_eval).item()
        # transfer of conditional
        for j in range(opt.TRANS_ITER):
            nll_cond = nll(model(x), y)
            optim.zero_grad()
            nll_cond.backward()
            optim.step()
            # eval conditional
            with torch.no_grad():
                nll_cond = nll(model(inp_eval), tar_eval)
            res['nll_cond'].append(nll_cond.detach().item())
            res['nll_marg'].append(nll_marg)
            res['iter'].append(j)
    print(f'TRANS {polarity.upper()}\t NLL conditional: {nll_cond}\t NLL marginal: {nll_marg}\t NLL TOTAL: {nll_cond+nll_marg}')
    return marginal, res

def viz_transfer(x2y_df, y2x_df):
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

def viz_dgp(scm, polarity):
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

def normal(mean, std, N): 
    return torch.normal(torch.ones(N).mul_(mean),
                        torch.ones(N).mul_(std)).view(-1, 1)

def snap(opt):
    exp_dir = os.path.join('src', 'transfer', 'experiments') 
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    fig_path = os.path.join(exp_dir,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.mkdir(fig_path)
    opt.FIGPATH = fig_path
    with open(os.path.join(opt.FIGPATH, 'options.txt'), 'w') as f:
        f.write(str(opt))


if __name__ == "__main__":
    opt = Namespace()
    opt.N_VIZ = 1e3
    # DGP
    opt.v_structure = True
    opt.ABLINE = False
    opt.NOISE_X = 1
    opt.INPUT_NOISE = False
    opt.OUTPUT_NOISE = 1
    opt.SPAN = 4
    opt.ANCHORS = 10
    opt.ORDER = 1
    opt.X_SCALE = 1
    opt.SCALE = 1
    # Model
    opt.CAPACITY = 32
    opt.NUM_COMPONENTS = 10
    opt.GMM_NUM_COMPONENTS = 10
    # Training
    opt.LR = 0.001
    opt.ITER = 300
    opt.REC_FREQ = 10
    opt.SAMPLES = 1000
    opt.N_EVAL = int(1e5)
    # Sampling 
    opt.TRAIN_DISTR = lambda n: normal(0, 2, n)*opt.X_SCALE
    opt.SWEEP = lambda n: torch.tensor(
        np.random.randint(-opt.SPAN, opt.SPAN, n))
    opt.TRANS_DISTR = lambda i, n: normal(i, 2, n)*opt.X_SCALE
    # Meta
    opt.TRANS_LR = 0.001  # they transfer with a higher learning rate?
    opt.TRANS_TASKS = 30
    opt.TRANS_ITER = 50
    opt.TRANS_SAMPLES = opt.SAMPLES  # is same in theirs
    opt.N_EXP = 1

    snap(opt)

    # Transfer training and regret comparison for both models

    scm = RandomSplineSCM(
        input_noise=opt.INPUT_NOISE, 
        output_noise=opt.OUTPUT_NOISE, 
        span=opt.SPAN*2, 
        num_anchors=opt.ANCHORS, 
        order=opt.ORDER, 
        range_scale=opt.SCALE
    )
    viz_dgp(scm, 'x2y')
    viz_dgp(scm, 'y2x')
    # models
    model_x2y = mdn(opt)
    model_y2x = mdn(opt)
    viz_cond_separate(model_x2y, polarity='x2y', name='PRE_')
    viz_cond(model_x2y, polarity='x2y', name='PRE_')
    viz_cond_separate(model_y2x, polarity='y2x', name='PRE_')
    viz_cond(model_y2x, polarity='y2x', name='PRE_')
    # causal conditional
    frames_x2y = train_nll(opt, model_x2y, scm, opt.TRAIN_DISTR,
        polarity='x2y', loss_fn=nll)
    # anti-causal conditional
    frames_y2x = train_nll(opt, model_y2x, scm, opt.TRAIN_DISTR,
        polarity='y2x', loss_fn=nll)
    viz_learning_curve(frames_x2y, polarity='x2y')
    viz_cond_separate(model_x2y, polarity='x2y', name='TRAIN_')
    viz_cond(model_x2y, polarity='x2y', name='TRAIN_')
    viz_learning_curve(frames_y2x, polarity='y2x')
    viz_cond_separate(model_y2x, polarity='y2x', name='TRAIN_')
    viz_cond(model_y2x, polarity='y2x', name='TRAIN_')
    # transfer
    x2y_marginal, x2y_res = train_transfer(opt, model_x2y, scm, 'x2y')
    y2x_marginal, y2x_res = train_transfer(opt, model_y2x, scm, 'y2x')
    viz_cond(model_x2y, polarity='x2y', name='TRANS_')
    viz_cond(model_y2x, polarity='y2x', name='TRANS_')
    viz_cond_separate(model_x2y, polarity='x2y', name='TRANS_')
    viz_cond_separate(model_y2x, polarity='y2x', name='TRANS_')

    # viz marginals
    viz_marginal(x2y_marginal, opt.TRANS_DISTR, polarity=f'x2y')
    viz_marginal(y2x_marginal, lambda i, n: scm(opt.TRANS_DISTR(i, n)), polarity=f'y2x')

    # viz transfer learning
    viz_transfer(pd.DataFrame(x2y_res), pd.DataFrame(y2x_res))
