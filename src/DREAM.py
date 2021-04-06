import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from networkx.drawing.nx_agraph import graphviz_layout
from argparse import Namespace


def overview():
    """ look at some training data """
    path = 'data/DreamData/Training/insilico/CSV/insilico.csv'
    df = pd.read_csv(path)
    print(df.head)
    print(df.columns)
    print(df.shape)


def load_gs():
    """ load ground truth graph """
    gs_path = 'data/DreamData/Test/SC1B_GS/SC1B_GS'
    names = pd.read_csv(f'{gs_path}/nodeNames.csv', header=None)
    gs = pd.read_csv(f'{gs_path}/trueGraph.csv', header=None)
    gs.columns = list(names.iloc[:,0])
    gs.index = list(names.iloc[:, 0])
    return gs

def viz_graph(opt, graph, name='gs_graph'):
    """ viz graph """
    G = nx.from_numpy_array(graph.values, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i: list(graph.columns)[i] for i in list(range(len(G)))})
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, node_color="lightblue", edge_color="black",
        with_labels=True, arrows=True,)
    plt.savefig(f"{opt.figpath}/{name}.png")

if __name__ == '__main__':
    opt = Namespace()
    opt.figpath = 'src/experiments/figures'
    gs = load_gs()
    viz_graph(opt, gs)
