import sys

from pandas.core.indexes import base
sys.path.append(sys.path[0]+'/..')
import custom_utils as utils
import os
import numpy as np
import pickle as pk
from DataGenerator import DataGenerator
from Scalers import Identity
from copy import deepcopy
import pandas as pd
import csv
from argparse import Namespace
from copy import deepcopy
import custom_utils.viz as viz


# TODO implement shift interventions here...
# def generate(opt):
#     description = utils.DatasetDescription(graph=opt.exp_name,
#                                            noise=opt.noise,
#                                            noise_variance=opt.noise_variance,
#                                            edge_weight_range=opt.edge_weight_range,
#                                            n_nodes=opt.n_nodes,
#                                            n_obs=opt.n_obs,
#                                            random_seed=opt.random_seed)
#     dg = DataGenerator(exp_name=opt.exp_name, base_dir=False, scaler=Identity())
#     dataset = dg.create_data(description, opt.W_true)
#     return dataset.data

def generate_nonlinear(opt, t=lambda x: x**2):
    """ 
    just generate x**2 for now, could go for the nonlinear noise types later
    """
    assert opt.n_nodes == 2  # crappy implementation for now
    x = np.zeros((opt.n_obs, opt.n_nodes))
    x[:, 0] = np.random.normal(opt.noise_means[0], 1, opt.n_obs)
    x[:, 1] = t(x[:, 0]) + np.random.normal(opt.noise_means[1], np.random.uniform(*opt.noise_variance), opt.n_obs)
    return x


def create_dataset(opt, obs=True, targets=[], nonlinear=False):
    gen = lambda opt: generate_nonlinear(opt, nonlinear) if nonlinear else generate
    utils.overwrite_folder(opt.out_dir)

    # int data
    # TODO int data is hack for now, only works for bivariate!
    int_data = []
    for i in targets:
        if sum(opt.W_true[:, i]) == 0:  # root cause
            # shift intervention
            temp_opt = deepcopy(opt)
            temp_opt.noise_means = [2.5, 0.]  # some random high value
            temp = gen(temp_opt)
            int_data.append(temp)  # do the usual thing
        else:  # effect node
            temp = gen(opt)
            temp[:, i] = np.random.normal(2.5, .1, opt.n_obs)  # random high value
            int_data.append(np.copy(temp))

    # obs data
    if obs:
        obs1 = gen(opt)
        obs_data = [obs1]
    else:
        obs_data = []

    # standardize
    def standardize(x):
        return (x-x.mean(axis=0))/x.std(axis=0)

    if opt.standardize_individually:
        int_data = [standardize(i) for i in int_data]
        obs_data = [standardize(i) for i in obs_data]

    # merge
    df = pd.concat([pd.DataFrame(i) for i in int_data] + 
                   [pd.DataFrame(i) for i in obs_data], 
                   axis=0).reset_index(drop=True)

    # TODO again just a bivariate hack!
    if not opt.standardize_individually:
        n_int = sum([len(i) for i in int_data])
        for i in range(df.values.shape[1]):
            if i in targets:
                df.values[n_int:, i] = standardize(df.values[n_int:, i])
            else:
                df.values[:, i] = standardize(df.values[:, i])

    # DAG1.npy
    np.save(os.path.join(opt.out_dir, f'DAG1.npy'), opt.W_true)

    # TODO not 100% sure about regimes but doesn't matter for now
    # intervention1.csv & regime1.csv
    interv = []
    regime = []
    for idx, i in enumerate(targets):
        interv.extend([[i] for _ in range(len(int_data[idx]))])
        regime.extend([idx+1 for _ in range(len(int_data[idx]))])
    interv.extend([[] for _ in range(sum([len(i) for i in obs_data]))])
    regime.extend([0 for _ in range(sum([len(i) for i in obs_data]))])
    df.to_csv(f'{opt.out_dir}/data_interv1.csv', index=False)
    with open(f'{opt.out_dir}/intervention1.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(interv)
    pd.Series(regime).to_csv(f'{opt.out_dir}/regime1.csv', index=False, header=False)

    # # TODO data_interv1.csv
    # fname = utils.dataset_description(dataset)
    # with open(os.path.join(opt.out_dir, f"{fname}.pk"), "wb") as f:
    #     pk.dump(dataset, f)

    # snap options
    utils.snap(opt, fname='options')

    # visualize all the data (with appropriate colors)?)
    viz.bivariate(opt, df.values)


def inspect_pk(base_dir):
    files = utils.load_pk_files(base_dir)
    if len(files) > 1:
        print("More than one file found.")
    else:
        print(files[0].data[:10, :])

def inspect(base_dir):
    """ Check manually if data has been saved correctly """
    # Data
    data = pd.read_csv(os.path.join(base_dir, 'data_interv1.csv'))
    print('Data:\n', data.values[0:10, :])
    # Masks
    masks = []
    with open(os.path.join(base_dir, 'intervention1.csv'), 'r') as f:
        interventions_csv = csv.reader(f)
        for row in interventions_csv:
            mask = [int(x) for x in row]
            masks.append(mask)
    print('Masks:', masks[0:10])
    # Regimes
    regimes = []
    with open(os.path.join(base_dir, 'regime1.csv'), 'r') as f:
        interventions_csv = csv.reader(f)
        for row in interventions_csv:
            regime = [int(x) for x in row]
            regimes.append(regime)
    print('Regimes:', regimes[0:10])
    print('colmeans', data.values.mean(axis=0))
    print('colvars', data.values.var(axis=0))
