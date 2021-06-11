import sys
sys.path.append(sys.path[0]+'/..')
import custom_utils as utils
import os
import numpy as np
import pickle as pk
from DataGenerator import DataGenerator
from Scalers import Identity
from argparse import Namespace

def create_dataset(opt):
    # create data
    description = utils.DatasetDescription(graph=opt.exp_name,
                                           noise=opt.noise,
                                           noise_variance=opt.noise_variance,
                                           edge_weight_range=opt.edge_weight_range,
                                           n_nodes=opt.n_nodes,
                                           n_obs=opt.n_obs,
                                           random_seed=opt.random_seed)
    dg = DataGenerator(exp_name=opt.exp_name, base_dir=opt.base_dir, scaler=Identity())
    dataset = dg.create_data(description, opt.W_true)

    # TODO create some interventional data later!
    # TODO rewrite this to DCDI standard!
    # TODO save settings in opt snap
    # # We need:
    # DAG1.npy
    # data_interv1.csv
    # intervention1.csv
    # regime1.csv

    # save data
    data_dir = os.path.join(opt.base_dir, opt.exp_name, "_data")
    utils.create_folder(data_dir)
    dirname = utils.dataset_dirname(dataset)
    exp_folder = os.path.join(data_dir, dirname)
    utils.create_folder(exp_folder)
    fname = utils.dataset_description(dataset)
    with open(os.path.join(exp_folder, f"{fname}.pk"), "wb") as f:
        pk.dump(dataset, f)


def inspect(base_dir):
    files = utils.load_pk_files(base_dir)
    if len(files) > 1:
        print("More than one file found.")
    else:
        print(files[0].data[:10, :])

if __name__ == "__main__":
    opt = Namespace()
    # files
    opt.exp_name = "sanity"
    opt.base_dir = "src/seq/data/" + opt.exp_name
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
    create_dataset(opt)

    # make sure everything went right
    inspect(opt.base_dir)