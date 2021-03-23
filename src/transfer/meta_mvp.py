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
import os
from datetime import datetime


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

        # TODO testing w=1
        self._y = np.linspace(-span, span, num_anchors)
        # self._y = np.random.uniform(-range_scale * span, range_scale * span, 
        #                             size=(num_anchors,))
        
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
    n = int(opt.N_VIZ)
    # inp = torch.FloatTensor(opt.N_EVAL).uniform_(-10, 10).view(-1, 1)
    inp = normal(torch.tensor(np.random.randint(-10, 10, n)), 2, n)
    orig = dgp(opt.N_EVAL).view(-1).numpy()

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
    # inp = torch.FloatTensor(opt.N_EVAL).uniform_(-10, 10).view(-1, 1)
    inp = normal(torch.tensor(np.random.randint(-10, 10, n)), 2, n)

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
    # inp = torch.FloatTensor(opt.N_EVAL).uniform_(-10, 10).view(-1, 1)
    inp = normal(torch.tensor(np.random.randint(-10, 10, n)), 2, n)
    with torch.no_grad():
        pi, mu, sigma = model(inp)
    # TODO use pi for alpha!
    # alpha=pi[:, i]
    for i in range(mu.shape[1]):
        plt.scatter(inp, mu[:, i], s=.1, alpha=.1, label=str(i))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.legend(markerscale=5)
    plt.title(polarity)
    plt.savefig(f'{opt.FIGPATH}/{name}{polarity}_cond_sep.png')
    plt.close()

def polarity_hlp(polarity, X, Y):
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
        # Generate samples from the training distry
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
    X_eval = train_distr_fn(opt.SAMPLES*10)
    with torch.no_grad():
        Y_eval = scm(X_eval)
    inp, tar = polarity_hlp(polarity, X_eval, Y_eval)
    
    # print model marginal and conditional
    marginal = gmm(opt)
    marginal.fit(inp)
    with torch.no_grad():
        nll_marg = loss_fn(marginal(inp), inp).item()
        nll_cond = loss_fn(model(inp), tar).item()
    print(f'{polarity.upper()}\t NLL conditional: {nll_cond}\t NLL marginal: {nll_marg}\t NLL TOTAL: {nll_cond+nll_marg}')
    
    # # Predict joint log likelihood!
    # # this uniform -> marginal seems to give us the same distribution as dgp!
    # unif = torch.FloatTensor(100000).uniform_(-opt.SPAN, opt.SPAN).view(-1, 1)
    # pi, mu, sigma = marginal(unif)
    # mixture = torch.distributions.Normal(loc=mu, scale=sigma)
    # with torch.no_grad():
    #     pred_vals = mixture.sample().numpy()
    # pi_np = pi.numpy()
    # pi_np = pi_np / np.sum(pi_np, axis=1, keepdims=True)
    # pred = [np.random.choice(pred_vals[i, :], p=pi_np[i, :]) 
    #         for i in range(len(pred_vals))]

    # # just plot some stuff to make sure things look good
    # sns.kdeplot(inp.view(-1,).numpy())
    # sns.kdeplot(pred)
    # plt.title(polarity)
    # plt.show()

    # TODO looks really good in terms of loss now. Get this into transfer learning!
    return frames

def train_transfer(opt, model, scm,  polarity):
    """formerly in train_utils, reproduce different adaptation plots """

    # sample eval dataset from all across the distribution
    # TODO is this a fair evaluation???
    X_eval = opt.TRANS_DISTR(opt.SWEEP(opt.N_EVAL), opt.N_EVAL)
    with torch.no_grad():
        Y_eval = scm(X_eval)
    X_eval, Y_eval = polarity_hlp(polarity, X_eval, Y_eval)
    
    marginal = gmm(opt)
    marginal.fit(X_eval)
    nll_marginal = nll(marginal(X_eval), X_eval)

    # online SGD and eval
    optim = torch.optim.Adam(model.parameters(), lr=opt.TRANS_LR)
    frames = []
    for i in tqdm(range(opt.TRANS_ITER)):
        x = opt.TRANS_DISTR(opt.SWEEP(1), opt.TRANS_SAMPLES)
        with torch.no_grad():
            y = scm(x)
        x, y = polarity_hlp(polarity, x, y)
        for j in range(opt.FINETUNE_NUM_ITER):
            loss_cond = nll(model(x), y)
            optim.zero_grad()
            loss_cond.backward()
            optim.step()
        # eval
        with torch.no_grad():
            nll_cond_eval = nll(model(X_eval), Y_eval)
            # TODO why don't we get the same loss here in the end??
        frames.append(Namespace(iter_num=i,
                        loss=nll_marginal.item() + nll_cond_eval.item()))
    return frames, marginal

def viz_transfer(df):
    """ Compare transfer regret for competing models """
    sns.lineplot(data=df, x='iter', y='x2y', label='x2y')
    sns.lineplot(data=df, x='iter', y='y2x', label='y2x')
    plt.title('Transfer Learning Adaptation')
    plt.ylabel('nll')
    plt.savefig(f'{opt.FIGPATH}/transfer.png')
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
    opt.INPUT_NOISE = False
    opt.OUTPUT_NOISE = True
    opt.SPAN = 4
    opt.ANCHORS = 2
    opt.ORDER =1
    opt.SCALE =1.
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
    opt.TRAIN_DISTR = lambda n: normal(0, 2, n)
    opt.SWEEP = lambda n: torch.tensor(
        np.random.randint(4, 5, n))
    opt.TRANS_DISTR = lambda i, n: normal(i, 2, n)
    # Meta
    opt.TRANS_LR = 0.01  # they transfer with a higher learning rate?
    opt.TRANS_ITER = 200
    opt.FINETUNE_NUM_ITER = 1
    opt.TRANS_SAMPLES = opt.SAMPLES  # is same in theirs
    opt.N_EXP = 1

    snap(opt)

    # Transfer training and regret comparison for both models
    res = {
        'x2y': [],
        'y2x': [],
        'iter': [],
    }
    for i in range(opt.N_EXP):
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

        # causal conditional
        model_x2y = mdn(opt)
        frames_x2y = train_nll(opt, model_x2y, scm, opt.TRAIN_DISTR,
            polarity='x2y', loss_fn=nll)
        # anti-causal conditional
        model_y2x = mdn(opt)
        frames_y2x = train_nll(opt, model_y2x, scm, opt.TRAIN_DISTR,
            polarity='y2x', loss_fn=nll)
        if i == opt.N_EXP-1:
            viz_learning_curve(frames_x2y, polarity='x2y')
            viz_cond_separate(model_x2y, polarity='x2y', name='TRAIN_')
            viz_cond(model_x2y, polarity='x2y', name='TRAIN_')
            viz_learning_curve(frames_y2x, polarity='y2x')
            viz_cond_separate(model_y2x, polarity='y2x', name='TRAIN_')
            viz_cond(model_y2x, polarity='y2x', name='TRAIN_')
        # transfer
        x2y_frames, x2y_marginal = train_transfer(opt, model_x2y, scm, 'x2y')
        y2x_frames, y2x_marginal = train_transfer(opt, model_y2x, scm, 'y2x')
        res['x2y'] += [frame.loss for frame in x2y_frames]
        res['y2x'] += [frame.loss for frame in y2x_frames]
        res['iter'] += [frame.iter_num for frame in y2x_frames]
        if i == opt.N_EXP-1:
            viz_cond(model_x2y, polarity='x2y', name='TRANS_')
            viz_cond(model_y2x, polarity='y2x', name='TRANS_')
            viz_cond_separate(model_x2y, polarity='x2y', name='TRANS_')
            viz_cond_separate(model_y2x, polarity='y2x', name='TRANS_')

    # viz some marginals
    viz_marginal(x2y_marginal, opt.TRAIN_DISTR, polarity=f'x2y')
    viz_marginal(y2x_marginal, lambda x: scm(opt.TRAIN_DISTR(x)), polarity=f'y2x')

    # viz transfer learning
    viz_transfer(pd.DataFrame(res))

# TODO we should do xlim ylim on plots to show it actually learns reasonable stuff!

### Current theory:
# Whichever model has the higher nll, that's the model the meta-learner will move towards?
# Alternatively, maybe meta learning will work once we have output noise??
# We do have a lot of random fluctuations either way...
# With output noise, the conditional loss of y2x seems to be a lot lower. Not anymore without output noise?

# TODO am I doing the evaluation right? I never retrain the gmm...
# TODO what's a sufficiently large eval data set?

# When I take n different randints to sample from in trans distr, convergence seems better... (Which I guess makes sense)

# output noise makes training a lot more stable
# TODO be very careful about not overtraining and divergence! some early stopping might make sense!

# TODO maybe there is catastrophic forgetting in only one direction???
# say, if in one direction it would be easier to re-fit the already fitted components / refitting would require some lasting change!

# TODO so what role exactly does the noise play again? is it really necessary??

# TODO Which sampling is fair, the uniform or the normal one???