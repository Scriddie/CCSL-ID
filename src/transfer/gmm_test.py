from causal_meta.modules.gmm import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

n=1000
gmm = GaussianMixture(2)

x = torch.FloatTensor(1000).normal_().view(-1, 1)
gmm.fit(x)

grid = torch.FloatTensor(n).uniform_(-1, 1)
pi, mu, sigma = gmm(grid)
m = torch.distributions.Normal(mu, sigma)
pred = m.sample().numpy()
pi_np = np.round(pi.numpy(), 3)
vals = [np.random.choice(pred[i, :], p=pi_np[i, :]) for i in range(len(pred))]

sns.kdeplot(x.view(-1), label='actual')
sns.kdeplot(vals, label='fit')
plt.legend()
plt.show()