import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('src/notears/notears/')
from linear import notears_linear

def viz_marg():
    nrows, ncols = 2, 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(len(M)):
        row = int(i/ncols)
        col = int(i % ncols)
        sns.kdeplot(data[:, i], ax=ax[row][col])
        print(np.mean(data[:, i]), np.var(data[:, i]))
    plt.show()

def test_lr():
    lr = LinearRegression()
    lr.fit(data[:, 1:], data[:, 0].ravel())
    print()
    for i, val in enumerate(lr.coef_):
        print(f'{i+1}\t {val}')


# TODO so where exactly does the data come from? why is the observational case solvable at all?
# TODO how does NoTears perform on this data?

path = 'src/dcdi/data/perfect/data_p10_e10_n10000_linear_struct'

M = np.load(f'{path}/DAG1.npy')
print(M)

data = np.load(f'{path}/data1.npy')
print(data.shape)

nt_res = notears_linear(data, lambda1=0.001, loss_type='l2')
print(np.where(np.abs(nt_res)>0, 1., 0.))