import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from networkx.drawing.nx_agraph import graphviz_layout
from argparse import Namespace
from utils.viz import *

def graph(graph, save=False):
    """ viz graph """
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    # G = nx.relabel_nodes(G, {i: list(graph.columns)[i] for i in list(range(len(G)))})
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, node_color="lightblue", edge_color="black",
        with_labels=True, arrows=True,)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

def heat(data, save=False):
    sns.heatmap(data)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

def marg(data, save=False):
    sns.kdeplot(data=data)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

    # nrows, ncols = 2, 5
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    # for i in range(len(M)):
    #     row = int(i/ncols)
    #     col = int(i % ncols)
    #     sns.kdeplot(data[:, i], ax=ax[row][col])
    #     print(np.mean(data[:, i]), np.var(data[:, i]))
    # plt.tight_layout()
    # plt.show()