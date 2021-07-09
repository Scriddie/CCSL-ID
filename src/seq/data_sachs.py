# TODO make sure we do useful stuff here?
import sys
sys.path.append(sys.path[0]+'/..')
import pandas as pd
import numpy as np
import os
import shutil
import csv
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import shutil
from argparse import Namespace
from copy import deepcopy
import itertools
import custom_utils as utils
import custom_utils.viz as viz


OBS_DATASETS = {
    'general1': '1. cd3cd28.xls',
    'general2': '2. cd3cd28icam2.xls'
}

# naming scheme is important
INT_DATASETS = {
    'activation-PKA': '9. b2camp.xls',  # TODO they are not using this?
    # is their reasoning for not using it valid given that we don't fit the interventions anyways?
    'inhibition-AKT': '3. cd3cd28+aktinhib.xls',
    'inhibition-MEK': '6. cd3cd28+u0126.xls',
    'activation-PKC': '8. pma.xls',
    'inhibition-PKC': '4. cd3cd28+g0076.xls',
    'inhibition-PIP2': '5. cd3cd28+psitect.xls',
    'inhibition2-PIP3': '7. cd3cd28+ly.xls',  # looks like this in their chart
}

def load_dag(opt):
    W_true = pd.read_csv(f'{opt.base_path}/../consensus_adj_mat.csv', index_col=0)
    W_true.columns = [i.upper() for i in W_true.columns]
    W_true.index = [i.upper() for i in W_true.index]
    W = W_true.loc[opt.vars_ord, opt.vars_ord]
    print(W)
    return W


def align_cols(df):
    """ align column names """
    df.columns = [c.upper() for c in df.columns]
    df = df.rename({'P44/42': 'ERK',
            'PAKTS473': 'AKT',
            'PJNK': 'JNK',
            'PRAF': 'RAF',
            'PMEK': 'MEK',
            'PLCG': 'PLCG'}, axis=1)
    return df


def load_data(opt, datasets, files, type=''):
    sets = deepcopy(datasets)
    for k, v in sets.items():
        sets[k] = pd.read_excel(os.path.join(opt.base_path, v), index_col=False)
        sets[k]['source'] = k
        print(f'{type.upper()}, {v}, {sets[k].shape}')
    data = [sets[i] for i in files]
    data = [align_cols(i) for i in data]
    return data


def show_marginals(obs_data, int_data):
    print('obs', obs_data.columns)
    sns.histplot(data=obs_data['MEK'])
    plt.title('observational')
    plt.show()
    plt.close()

    print('int', int_data.columns)
    sns.histplot(data=int_data['MEK'])
    plt.title('interventional')
    plt.show()
    plt.close()


def remove_outliers(opt, df):
    df = deepcopy(df)
    if opt.remove_outliers:
        df = df[(np.abs(stats.zscore(df[opt.vars_ord])) < opt.remove_outliers).all(axis=1)]
    return df


def merge_data(int_data, obs_data):
    df = pd.concat(int_data + obs_data, axis=0).reset_index(drop=True)
    df['TYPE'] = pd.Series(['intervention' for i in range(len(int_data)) for _ in range(len(int_data[i]))] + ['observation' for i in range(len(obs_data)) for _ in range(len(obs_data[i]))])
    return df


# TODO warning that this always selects the first two!
def viz_joint(opt, df, name, hue=None):
    for a, b in itertools.combinations(opt.vars_ord, 2):
        i, j = opt.vars_ord.index(a), opt.vars_ord.index(b)
        sns.jointplot(data=df, x=opt.vars_ord[i], y=opt.vars_ord[j], hue=hue, alpha=.1)
        plt.title(name)
        plt.savefig(f'{opt.out_dir}/figures/distribution_{name}_{a}-{b}.png')
        plt.close()


def viz_lm(opt, df, name, hue=None):
    if len(opt.int_vars) > 0:
        sns.lmplot(data=df, x=opt.int_vars[0], y=opt.obs_vars[0], hue=hue)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f'{opt.out_dir}/trends_{name}.png')
        plt.close()
    else:
        print('No int vars')


def standardize(opt, df):
    df[opt.vars_ord] = (df[opt.vars_ord]-df[opt.vars_ord].mean())/df[opt.vars_ord].std()
    return df


def log_transform(opt, df):
    df[opt.vars_ord] = np.log(df[opt.vars_ord].values)
    return df


if __name__ == '__main__':
    utils.create_folder('src/seq/data')
    utils.overwrite_folder('src/seq/data/sachs')
    utils.overwrite_folder('src/seq/data/sachs/figures')

    opt = Namespace()
    opt.base_path = './data/sachs/Data Files/'
    # activation PKA led to good results so far, think about it!
    opt.int_files = [
        # TODO they only use inhibitions! Is this the way to go?
        # 'activation-PKA',
        'inhibition-AKT',
        'inhibition-MEK',
        # 'activation-PKC',
        'inhibition-PKC',
        'inhibition-PIP2',
        'inhibition2-PIP3',  # TODO are we sure about this target?
    ]
    opt.int_vars = list(set([i.split('-')[-1] for i in opt.int_files]))
    opt.obs_files = ['general1', 'general2']
    opt.obs_vars = ['RAF', 'ERK', 'JNK', 'P38', 'PLCG', 'PKA']
    # TODO rename PLCG to PLC?
    opt.out_dir = 'src/seq/data/sachs'
    opt.vars_ord = opt.int_vars + opt.obs_vars
    opt.standardize = True
    opt.standardize_globally = True
    opt.log_transform = False
    opt.remove_outliers=False

    # # create folder
    # if os.path.exists(opt.out_dir):
    #     shutil.rmtree(opt.out_dir)
    # os.mkdir(opt.out_dir)

    # load data
    obs_data = load_data(opt, OBS_DATASETS, opt.obs_files, 'obs')
    int_data = load_data(opt, INT_DATASETS, opt.int_files, 'int')

    # TODO dirty hack to get rid of some obs
    # obs_data = [i.iloc[0:10, :] for i in obs_data]
    # int_data = [i.iloc[0:100, :] for i in int_data]
    print('OBS:\t', [i.shape for i in obs_data])
    print('INT:\t', [i.shape for i in int_data])

    # # show marginals
    # show_marginals(obs_data, int_data[0])

    # DAG
    W = load_dag(opt)

    # intervention masks & regimes
    interv = []
    regime = []
    # TODO check this whole thing?
    # TODO the encoding of int data is wrong still!
    for idx, d in enumerate(int_data):
        # encode the right target
        target = opt.int_files[idx].split('-')[-1]
        if target in W.columns:
            target_idx = list(W.columns).index(target)
            interv.extend([[target_idx] for _ in range(len(d))])
            regime.extend([idx+1 for _ in range(len(d))])
        else:
            interv.extend([[] for _ in range(len(d))])
            regime.extend([0 for _ in range(len(d))])
    # obs data
    interv.extend([[] for _ in range(sum([len(i) for i in obs_data]))])
    regime.extend([0 for _ in range(sum([len(i) for i in obs_data]))])

    # log transform
    if opt.log_transform:
        int_data = [log_transform(opt, i) for i in int_data]
        obs_data = [log_transform(opt, i) for i in obs_data]

    # standardize locally
    if opt.standardize and (not opt.standardize_globally):
        int_data = [standardize(opt, i) for i in int_data]
        obs_data = [standardize(opt, i) for i in obs_data]

    # merge data sources
    df = merge_data(int_data, obs_data)

    # viz joints
    print(df.head(), df.shape)

    # remove outliers globally
    if opt.remove_outliers:
        df = remove_outliers(opt, df)

    # select variables
    source = df['SOURCE']
    df = df.loc[:, opt.vars_ord]
    df.reset_index(inplace=True, drop=True)

    # standardize globally
    # TODO doesn't work for multiple interventional datasets yet!!!
    if opt.standardize and opt.standardize_globally:
        n_int = [len(i) for i in int_data]
        tmp = df.values
        for idx, i in enumerate(opt.vars_ord):
            if i in opt.int_vars:
                n_before_int = sum(n_int[:idx])
                n_this_int = n_int[idx]
                mask = np.ones(df.shape[0])
                mask[n_before_int:n_before_int+n_this_int] = 0
                mask = mask.astype(bool)
                tmp[mask, idx] =  (df.values[mask, idx] - np.mean(df.values[mask, idx])) / np.std(df.values[mask, idx])
            else:
                tmp[:, idx] =  (df.values[:, idx] - np.mean(df.values[:, idx])) / np.std(df.values[:, idx])
        df = pd.DataFrame(tmp, columns=df.columns)

    # show first int and obs variable
    viz_joint(opt, df, 'final', hue=source)

    # Save
    # TODO save graph as heatmap!
    print(df.head(), df.shape)
    viz.mat(opt, W)
    np.save(f'{opt.out_dir}/DAG1.npy', W)
    with open(f'{opt.out_dir}/DAG.txt', 'w+') as f:
        f.write(str(W))
    np.save(f'{opt.out_dir}/data_interv1.npy', df.values)
    df.to_csv(f'{opt.out_dir}/data_interv1.csv', index=False)
    with open(f'{opt.out_dir}/intervention1.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(interv)
    pd.Series(regime).to_csv(f'{opt.out_dir}/regime1.csv', index=False, header=False)
