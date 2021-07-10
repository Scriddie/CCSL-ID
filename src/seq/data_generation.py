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


# old version of data generation
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

def generate_nonlinear(opt, target, transformation):
    """ 
    just generate a chain right now, could go for the nonlinear noise types later
    """
    d = len(opt.W_true)
    X = np.zeros((opt.n_obs, d))
    filled = np.zeros(d)
    while sum(filled) != d:
        for i in range(d):
            # fill in all the ones that have all filled-in parents
            parents_filled_in = True
            parent_indices = opt.W_true[:, i].nonzero()[0]
            for parent in parent_indices:
                if sum(X[:, parent] == 0) == opt.n_obs:
                    parents_filled_in = False
                if not parents_filled_in:
                    break

            if not parents_filled_in:
                continue
            else:
                # all parents filled in, generate some data
                if i == target:
                    # target -> shift intervention
                    X[:, i] = np.random.normal(2.5, 1., opt.n_obs)  # random high value
                else:
                    # observational data
                    if len(parent_indices) == 0:
                        # root node
                        X[:, i] = np.random.normal(np.random.uniform(*opt.noise_means), np.random.uniform(*opt.noise_variance), opt.n_obs)
                    else:
                        for parent in parent_indices:
                            X[:, i] += transformation(X[:, parent])
                            X[:, i] += np.random.normal(np.random.uniform(*opt.noise_means), np.random.uniform(*opt.noise_variance), opt.n_obs)

            # advance break condition
            filled[i] = 1
    return X


def create_dataset(opt, obs, targets, transformation):
    """ 
    opt: options
    obs_data: Boolean, do we want obs data
    targets: intervention targets
    transformation: for now singular transformation function
    """
    utils.overwrite_folder(opt.out_dir)

    # empty int data
    int_data = []

    # generate interventional data
    for target in targets: 
        int_data.append(generate_nonlinear(opt, target, transformation))

    # generate observational data
    obs_data = []
    if obs:
        obs_data.append(generate_nonlinear(opt, None, transformation))

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

    # # TODO replace with better logic in sachs!
    # # TODO in both, rescale interventional sections using obs mean and var!
    # if not opt.standardize_individually:
    #     n_int = sum([len(i) for i in int_data])
    #     for i in range(df.values.shape[1]):
    #         if i in targets:
    #             df.values[n_int:, i] = standardize(df.values[n_int:, i])
    #         else:
    #             df.values[:, i] = standardize(df.values[:, i])

    # TODO check again
    if not opt.standardize_individually:
        n_int = [len(i) for i in int_data]
        tmp = df.values
        non_targets = list(set(range(len(opt.W_true))).difference(set(targets)))
        vars_ord = targets + non_targets
        for idx, i in enumerate(vars_ord):
            if i in targets:
                n_before_int = sum(n_int[:idx])
                n_this_int = n_int[idx]
                mask = np.ones(df.shape[0])
                mask[n_before_int:n_before_int+n_this_int] = 0
                mask = mask.astype(bool)
                non_int_mean = np.mean(df.values[mask, idx])
                non_int_std = np.std(df.values[mask, idx])
                # standardize all with non_int values
                tmp[:, idx] =  (df.values[:, idx] - non_int_mean) / non_int_std
            else:
                tmp[:, idx] =  (df.values[:, idx] - np.mean(df.values[:, idx])) / np.std(df.values[:, idx])
        df = pd.DataFrame(tmp, columns=df.columns)

    # DAG1.npy
    np.save(os.path.join(opt.out_dir, f'DAG1.npy'), opt.W_true)

    # write outputs
    interv = []
    regime = []
    for idx, i in enumerate(targets):
        interv.extend([[i] for _ in range(len(int_data[idx]))])
        regime.extend([idx+1 for _ in range(len(int_data[idx]))])
    interv.extend([[] for _ in range(sum([len(i) for i in obs_data]))])
    regime.extend([0 for _ in range(sum([len(i) for i in obs_data]))])
    df.to_csv(f'{opt.out_dir}/data_interv1.csv', index=False, header=False)
    with open(f'{opt.out_dir}/intervention1.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(interv)
    pd.Series(regime).to_csv(f'{opt.out_dir}/regime1.csv', index=False, header=False)
    viz.mat(opt, opt.W_true)
    with open(f'{opt.out_dir}/DAG.txt', 'w+') as f:
        f.write(str(opt.W_true))

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
