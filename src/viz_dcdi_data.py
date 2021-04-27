import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('src/notears/notears/')
from linear import notears_linear
import utils.viz as viz

def test_lr():
    lr = LinearRegression()
    lr.fit(data[:, 1:], data[:, 0].ravel())
    print()
    for i, val in enumerate(lr.coef_):
        print(f'{i+1}\t {val}')


# path = 'src/dcdi/data/perfect/data_p10_e10_n10000_linear_struct'
path = 'src/dcdi/data/custom_data/data_p2_e1.0_n10000_custom'

M = np.load(f'{path}/DAG1.npy')
print(M)

data = np.load(f'{path}/data_interv1.npy')
print(data.shape)
viz.graph(M, save='src/experiments/figures/graph.png')
viz.marg(data, save='src/experiments/figures/marg.png')
viz.heat(M, save='src/experiments/figures/gt_graph.png')
# TODO viz learnt graph, see if the intervened-upon nodes are an easier fit
# viz.graph(M, save='src/experiments/figures/est_graph.png')

## Notears for comparison
# nt_res = notears_linear(data, lambda1=0.001, loss_type='l2')
# print(np.where(np.abs(nt_res)>0, 1., 0.))