import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from networkx.drawing.nx_agraph import graphviz_layout
from argparse import Namespace


def load_gs(verbose=False):
    """ load ground truth graph """
    gs_path = 'data/DreamData/Test/SC1B_GS/SC1B_GS'
    names = pd.read_csv(f'{gs_path}/nodeNames.csv', header=None)
    gs = pd.read_csv(f'{gs_path}/trueGraph.csv', header=None)
    for i in range(len(gs)):
        gs.iloc[i, i] = 0
    if verbose:
        print(gs)
    gs.columns = list(names.iloc[:,0])
    gs.index = list(names.iloc[:, 0])
    return gs

def viz_graph(opt, graph, name='gs_graph'):
    """ viz graph """
    G = nx.from_numpy_array(graph.values, create_using=nx.DiGraph)
    # G = nx.relabel_nodes(G, {i: list(graph.columns)[i] for i in list(range(len(G)))})
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, node_color="lightblue", edge_color="black",
        with_labels=True, arrows=True,)
    plt.savefig(f"{opt.figpath}/{name}.png")

def viz_node(opt, df, node, name=''):
    nrows = 2
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 7))
    stimuli = list(df.Stimulus.unique())
    for i, s in enumerate(stimuli):
        row = int(i / ncols)
        col = i % ncols
        df_red = df.loc[df.Stimulus==s, :]
        sns.lineplot(data=df_red, x='Timepoint', y=node, hue='Inhibitor', ax=ax[row][col])
        ax[row][col].set_title(s)
    plt.tight_layout()
    plt.subplots_adjust(wspace=.4, hspace=.4, top=0.9)
    plt.suptitle(f'{node} across Inhibitors and Stimuli')
    plt.savefig(f'{opt.figpath}/{node}.png')
    plt.close()

def load_train(verbose=False):
    """ look at some training data """
    path = 'data/DreamData/Training/insilico/CSV/insilico.csv'
    df = pd.read_csv(path)
    df.drop(columns=['Cell Line'], inplace=True) # contains no info
    df['Timepoint'] = df['Timepoint'].apply(lambda x: int(x.strip('min')))
    df['Inhibitor'].fillna('None', inplace=True)
    if verbose:
        print(df.head)
        print(df.columns)
        print(df.shape)
    return df


if __name__ == '__main__':
    opt = Namespace()
    opt.figpath = 'src/experiments/figures'
    
    # The gold standard is not a DAG
    # gold standard network
    gs = load_gs(True)
    viz_graph(opt, gs)

    # train data
    df = load_train(False)
    print(df.Stimulus.value_counts())
    for AB in ('AB1', 'AB11'):
        viz_node(opt, df, AB)

