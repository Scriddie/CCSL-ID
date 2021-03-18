import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt 
import torch
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# TODO concentrate all relevant logic in this file, compress to MVP
from causal_meta.modules.mdn import mdn_nll as nll
from causal_meta.utils.data_utils import RandomSplineSCM
from argparse import Namespace

from causal_meta.modules.gmm import GaussianMixture
from causal_meta.modules.mdn import MDN

class RandomSplineSCM(nn.Module): 
    def __init__(self, input_noise=False, output_noise=True, 
                 span=6, num_anchors=10, order=3, range_scale=1.): 
        super(RandomSplineSCM, self).__init__()
        self._span = span
        self._num_anchors = num_anchors
        self._range_scale = range_scale
        self._x = np.linspace(-span, span, num_anchors)
        self._y = np.random.uniform(-range_scale * span, range_scale * span, 
                                    size=(num_anchors,))
        self._spline_spec = interpolate.splrep(self._x, self._y, k=order)
        self.input_noise = input_noise
        self.output_noise = output_noise
    
    def forward(self, X, Z=None):
        if Z is None: 
            Z = self.sample(X.shape[0])
        if self.input_noise: 
            X = X + Z
        X_np = X.detach().cpu().numpy().squeeze()
        _Y_np = interpolate.splev(X_np, self._spline_spec)
        _Y = torch.from_numpy(_Y_np).view(-1, 1).float().to(X.device)
        if self.output_noise:
            Y = _Y + Z
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
    n = 1000
    x = torch.FloatTensor(n).uniform_(-4, 4).view(-1, 1)
    orig = dgp(n).view(-1).numpy()
    
    # x = normal(torch.tensor(np.random.randint(-4, 4, n)), 2, n)
    pi, mu, sigma = model(x)
    mixture = torch.distributions.Normal(loc=mu, scale=sigma)
    pred_vals = mixture.sample().numpy()
    pi_np = pi.numpy()
    pi_np = pi_np / np.sum(pi_np, axis=1, keepdims=True)
    pred = [np.random.choice(pred_vals[i, :], p=pi_np[i, :]) 
            for i in range(len(pred_vals))]

    sns.kdeplot(x=orig, label='original')
    sns.kdeplot(x=pred, label='predicted')
    plt.legend()
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_marginal')
    plt.close()

def viz_gen(model, polarity='', actual=None):
    """ show generative distributions before transfer """
    n = 1000
    x = normal(torch.tensor(np.random.randint(-4, 4, n)), 2, n)
    with torch.no_grad():
        pi, mu, sigma = model(x)
    mixture = torch.distributions.Normal(loc=mu, scale=sigma)
    pred = torch.sum(mixture.sample() * pi, axis=1)
    plt.scatter(x.numpy(), pred.numpy(), s=.2)
    if actual is not None:
        plt.scatter(x.numpy(), actual(n), s=.2)
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_cond.png')
    plt.close()

def viz_gen_separate(model, polarity=''):
    """ show generative distributions before transfer """
    n = 1000
    samples, values = np.zeros(1000), np.zeros((1000, 10))
    for i in range(n):
        x = normal(np.random.randint(-4, 4), 2, 1)
        with torch.no_grad():
            pi, mu, sigma = model(x)
        mixture = torch.distributions.Normal(loc=mu, scale=sigma)
        pred = mixture.sample().squeeze()  # pi.squeeze() * 
        samples[i] = x.item()
        values[i, :] = pred.numpy()
    for i in range(values.shape[1]):
        plt.scatter(samples, values[:, i], s=.2, label=str(i))
    plt.legend(markerscale=5)
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{polarity}_cond_sep.png')
    plt.close()

def train_nll(opt, model, scm, train_distr_fn, polarity='X2Y', 
    loss_fn=nn.MSELoss(), decoder=None, encoder=None):
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    frames = []
    for iter_num in tqdm(range(opt.ITER)):
        # Generate samples from the training distry
        X = train_distr_fn(opt.SAMPLES)
        with torch.no_grad():
            Y = scm(X)
        if polarity == 'X2Y':
            inp, tar = X, Y
        elif polarity == 'Y2X':
            inp, tar = Y, X
        else:
            raise ValueError
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
    return frames

def train_transfer(opt, model, scm,  polarity):
    """formerly in train_utils, reproduce different adaptation plots """

    # eval dataset
    if polarity == 'x2y':
        X_eval = opt.TRANS_DISTR(opt.N_EVAL)
        with torch.no_grad():
            Y_eval = scm(X_eval)
    elif polarity == 'y2x':
        Y_eval = opt.TRANS_DISTR(opt.N_EVAL)
        with torch.no_grad():
            X_eval = scm(Y_eval)
    else:
        raise ValueError('No such polarity')
    
    marginal = gmm(opt)
    marginal.fit(X_eval)
    viz_marginal(marginal, opt.TRANS_DISTR, polarity=f'{polarity}')
    nll_marginal = nll(marginal(X_eval), X_eval)

    # online SGD and eval
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    frames = []
    for i in range(opt.TRANS_ITER):
        samples = 100
        if polarity == 'x2y':
            x = opt.TRANS_DISTR(samples)
            with torch.no_grad():
                y = scm(x)
        else:
            y = opt.TRANS_DISTR(samples)
            with torch.no_grad():
                x = scm(y)
        prd = model(x)
        loss_conditional = nll(prd, y)
        optim.zero_grad()
        loss_conditional.backward()
        optim.step()

        # eval
        with torch.no_grad():
            nll_cond = nll(model(X_eval), Y_eval)
            frames.append(Namespace(iter_num=i,
                loss=nll_marginal.item() + nll_cond.item()))
    return frames

def viz_transfer(df):
    """ Compare transfer regret for competing models """
    sns.lineplot(data=df, x='iter', y='x2y', label='x2y')
    sns.lineplot(data=df, x='iter', y='y2x', label='y2x')
    plt.ylabel('nll')
    plt.savefig(f'{opt.FIGPATH}/transfer.png')
    plt.close()

def viz_dgp(scm):
    """ Visualize data-generating process """
    plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)
    mus = [-4., 4., 0]
    colors = ['C3', 'C2', 'C0']
    labels = [r'Transfer ($\mu = -4$)', r'Transfer ($\mu = +4$)', 'Training']

    for mu, color, label in zip(mus, colors, labels):
        X = mu + 2 * torch.randn((1000, 1))
        ax.scatter(X.squeeze(1).numpy(), scm(X).squeeze(1).numpy(),             color=color, marker='+', alpha=0.3, label=label)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc=4, prop={'size': 13})
    ax.set_xlabel('A', fontsize=14)
    ax.set_ylabel('B', fontsize=14)
    plt.savefig(f'{opt.FIGPATH}/data_x2y.png')
    plt.close()

def normal(mean, std, N): 
    return torch.normal(torch.ones(N).mul_(mean),
                        torch.ones(N).mul_(std)).view(-1, 1)


if __name__ == "__main__":
    opt = Namespace()
    # Model
    opt.CAPACITY = 32
    opt.NUM_COMPONENTS = 10
    opt.GMM_NUM_COMPONENTS = 10
    # Training
    opt.LR = 0.02
    opt.ITER = 500
    opt.REC_FREQ = 10
    opt.SAMPLES = 1000
    opt.N_EVAL = int(1e4)
    # Meta
    opt.TRANS_ITER = 200
    opt.N_EXP = 50
    # Sampling 
    opt.TRAIN_DISTR = lambda n: normal(0, 2, n)
    opt.TRANS_DISTR = lambda n: normal(
        torch.tensor(np.random.randint(-4, 4, n)), 2, n)
    opt.FIGPATH = 'src/transfer/figures'

    # Data Generation
    scm = RandomSplineSCM(False, True, 8, 8, 3, range_scale=1.)
    viz_dgp(scm)
    
    # causal conditional
    model_x2y = mdn(opt)
    frames_x2y = train_nll(opt, model_x2y, scm, opt.TRAIN_DISTR,
        polarity='X2Y', loss_fn=nll, decoder=None, encoder=None)
    viz_learning_curve(frames_x2y, polarity='x2y')
    viz_gen_separate(model_x2y, polarity='x2y')
    viz_gen(model_x2y, polarity='x2y')

    # anti-causal conditional
    model_y2x = mdn(opt)
    frames_y2x = train_nll(opt, model_y2x, scm, opt.TRAIN_DISTR,
        polarity='Y2X', loss_fn=nll, decoder=None, encoder=None)
    viz_learning_curve(frames_y2x, polarity='y2x')
    viz_gen_separate(model_y2x, polarity='y2x')
    viz_gen(model_y2x, polarity='y2x')

    # Transfer training and regret comparison for both models
    res = {
        'x2y': [],
        'y2x': [],
        'iter': [],
    }
    for i in tqdm(range(opt.N_EXP)):
        x2y_frames = train_transfer(opt, model_x2y, scm, 'x2y')
        y2x_frames = train_transfer(opt, model_y2x, scm, 'y2x')
        res['x2y'] += [frame.loss for frame in x2y_frames]
        res['y2x'] += [frame.loss for frame in y2x_frames]
        res['iter'] += [frame.iter_num for frame in y2x_frames]
    viz_transfer(pd.DataFrame(res))

