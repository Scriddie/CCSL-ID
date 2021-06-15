import sys

from pandas.core.indexes import base
sys.path.append(sys.path[0]+'/..')
import custom_utils as utils
import os
import numpy as np
import pickle as pk
from DataGenerator import DataGenerator
from Scalers import Identity
import pandas as pd
import csv
from argparse import Namespace
from copy import deepcopy


def generate(opt):
    description = utils.DatasetDescription(graph=opt.exp_name,
                                           noise=opt.noise,
                                           noise_variance=opt.noise_variance,
                                           edge_weight_range=opt.edge_weight_range,
                                           n_nodes=opt.n_nodes,
                                           n_obs=opt.n_obs,
                                           random_seed=opt.random_seed)
    dg = DataGenerator(exp_name=opt.exp_name, base_dir=False, scaler=Identity())
    dataset = dg.create_data(description, opt.W_true)
    return dataset.data

def create_dataset(opt, targets=[]):
    utils.overwrite_folder(opt.exp_dir)

    # int data
    # TODO int data is hack for now! Check this later!
    int_data = []
    for i in targets:
        if sum(opt.W_true[:, i]) == 0:  # root cause
            # adjust params a bit? use some wrong noise etc.?
            temp = generate(opt)
            int_data.append(np.copy(temp))  # do the usual thing
        else:  # effect node
            temp = generate(opt)
            temp[:, i] = np.random.normal(0, 1, opt.n_obs)
            int_data.append(np.copy(temp))

    # obs data
    obs1 = generate(opt)
    obs_data = [obs1]

    # standardize
    def standardize(x):
        return (x-x.mean(axis=0))/x.std(axis=0)
    int_data = [standardize(i) for i in int_data]
    obs_data = [standardize(i) for i in obs_data]

    # merge
    df = pd.concat([pd.DataFrame(i) for i in int_data] + 
                   [pd.DataFrame(i) for i in obs_data], 
                   axis=0).reset_index(drop=True)

    # DAG1.npy
    np.save(os.path.join(opt.exp_dir, f'DAG1.npy'), opt.W_true)

    # TODO not 100% sure about regimes but doesn't matter for now
    # intervention1.csv & regime1.csv
    interv = []
    regime = []
    for idx, i in enumerate(targets):
        interv.extend([[i] for _ in range(len(int_data[idx]))])
        regime.extend([idx+1 for _ in range(len(int_data[idx]))])
    interv.extend([[] for _ in range(sum([len(i) for i in obs_data]))])
    regime.extend([0 for _ in range(sum([len(i) for i in obs_data]))])
    pd.DataFrame(df).to_csv(f'{opt.exp_dir}/data_interv1.csv', index=False)
    with open(f'{opt.exp_dir}/intervention1.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(interv)
    pd.Series(regime).to_csv(f'{opt.exp_dir}/regime1.csv', index=False, header=False)

    # # TODO data_interv1.csv
    # fname = utils.dataset_description(dataset)
    # with open(os.path.join(opt.exp_dir, f"{fname}.pk"), "wb") as f:
    #     pk.dump(dataset, f)

    # snap options
    utils.snap(opt, fname='options')


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


if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "sanity"
    opt.exp_dir = "src/seq/data/" + opt.exp_name
    # data
    opt.noise = 'gauss'
    opt.noise_variance = (1, 1)
    opt.edge_weight_range = (1, 1)
    opt.W_true = np.array([[0, 1],
                           [0, 0]])
    opt.n_nodes = len(opt.W_true)
    opt.n_obs = 10000
    opt.random_seed = 0

    # create dataset
    create_dataset(opt, targets=[0])

    # make sure everything went right
    inspect(opt.exp_dir)


# TODO generate some interventional data on A!
# TODO generate some interventional data on B!