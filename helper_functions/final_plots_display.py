#!/usr/bin/env python3

import matplotlib.pyplot as plt
from os.path import isfile, join
import sys
import numpy as np
import networkx as nx
import csv
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
import random
import os
import itertools
from tqdm import trange

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    #"font.size"   : 15,
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
        "\\usepackage{units}",  # load additional packages
        "\\usepackage{metalogo}",
        "\\usepackage{unicode-math}",  # unicode math setup
        r"\setmathfont{xits-math.otf}",
        r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ])
})

def all_dictionaries_display(list_network_files, motif_sizes=[6, 11, 21, 51], name='1', plot_graph=True):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"

    ncols = len(motif_sizes)
    nrows = len(list_network_files)
    fig = plt.figure(figsize=(ncols * (10 / 4), nrows * (15 / 4)), constrained_layout=False)
    n_components = 25
    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.05)
    for row in range(nrows):
        for col in range(ncols):
            sub_rows = np.round(np.sqrt(n_components))
            sub_rows = sub_rows.astype(int)
            if sub_rows ** 2 == n_components:
                sub_cols = sub_rows
            else:
                sub_cols = sub_rows + 1

            if not ((motif_sizes[col] == 101) and (list_network_files[row] in [])):
                ### Load results file
                # Make nested gridspecs.
                inner_grid = outer_grid[row * ncols + col].subgridspec(sub_rows, sub_cols, wspace=0.1, hspace=0.1)

                ### Load results file
                ntwk = list_network_files[row]
                network_name = ntwk.replace('.txt', '')
                network_name = network_name.replace('.', '')
                print('!!!!!', str(motif_sizes[col]))
                path = save_folder + '/full_result_' + str(network_name) + "_k_" + str(motif_sizes[col]) + "_r_" + str(
                    n_components) + ".npy"
                result_dict = np.load(path, allow_pickle=True).item()
                W = result_dict.get('Dictionary learned')
                At = result_dict.get('Code COV learned')
                k = result_dict.get('Motif size')

                # Add plot labels and remove remainder of axis.
                ax_outer = fig.add_subplot(outer_grid[row * ncols + col])
                # remove boarders
                ax_outer.spines['top'].set_visible(False)
                ax_outer.spines['right'].set_visible(False)
                ax_outer.spines['bottom'].set_visible(False)
                ax_outer.spines['left'].set_visible(False)
                if col == 0:
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_005':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{1}$'
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_01':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{2}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_50':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{1}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_100':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{2}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_25':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{1}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_50':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{2}$'
                    if network_name == 'Caltech36':
                        ntwk_label = '$\\textsc{\\texttt{Caltech}}$'
                    if network_name == 'MIT8':
                        ntwk_label = '$\\textsc{\\texttt{MIT}}$'
                    if network_name == 'UCLA26':
                        ntwk_label = '$\\textsc{\\texttt{UCLA}}$'
                    if network_name == 'Harvard1':
                        ntwk_label = '$\\textsc{\\texttt{Harvard}}$'
                    if network_name == 'COVID_PPI':
                        ntwk_label = '$\\textsc{\\texttt{Coronavirus}}$'
                    if network_name == 'facebook_combined':
                        ntwk_label = '$\\textsc{\\texttt{SNAP FB}}$'
                    if network_name == 'arxiv':
                        ntwk_label = '$\\textsc{\\texttt{arXiv}}$'
                    if network_name == 'node2vec_homosapiens_PPI':
                        ntwk_label = '$\\textsc{\\texttt{H. sapiens}}$'
                    if network_name == 'SBM1':
                        ntwk_label = '$\\textsc{\\texttt{SBM}}_{1}$'
                    if network_name == 'SBM2':
                        ntwk_label = '$\\textsc{\\texttt{SBM}}_{2}$'

                    ax_outer.set_ylabel(str(ntwk_label), fontsize=15)
                    ax_outer.yaxis.set_label_position('left')

                if row == 0:
                    ax_outer.set_title('$k$ = ' + str(k), fontsize=15)

                ax_outer.axes.xaxis.set_ticks([])
                ax_outer.axes.yaxis.set_ticks([])

                ### Use the code covariance matrix At to compute importance
                importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
                idx = np.argsort(importance)
                idx = np.flip(idx)

                ### Add subplot

                for i in range(sub_rows * sub_cols):
                    a = i // sub_cols
                    b = i % sub_cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    if not plot_graph:
                        ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=10)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        A_sub = W.T[idx[i]].reshape(k, k)
                        H = nx.from_numpy_matrix(A_sub)
                        G1 = nx.Graph()
                        for a in np.arange(k):
                            for b in np.arange(k):
                                u = list(H.nodes())[a]
                                v = list(H.nodes())[b]
                                if H.has_edge(u,v):
                                    if np.abs(a-b) == 1:
                                        G1.add_edge(u,v, color='r', weight=A_sub[a,b])
                                    else:
                                        G1.add_edge(u,v, color='b', weight=A_sub[a,b])

                        pos = nx.spring_layout(G1)
                        edges = G1.edges()
                        colors = [G1[u][v]['color'] for u,v in edges]
                        weights = [7*G1[u][v]['weight'] for u,v in edges]
                        if network_name == "COVID_PPI" and motif_sizes[col] > 20:
                            weights = [(2/(motif_sizes[col]-10)) *G1[u][v]['weight'] for u,v in edges]

                        node_size=50/motif_sizes[col]
                        nx.draw(G1, with_labels=False, node_size=node_size, ax=ax, width=weights, edge_color=colors, label='Graph')

    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0)
    fig.savefig(save_folder + '/all_dictionaries_' + str(name) + '.png', bbox_inches='tight')


def all_dictionaries_display_rank(list_network_files, motif_size=21, rank=[9, 16, 25, 36, 49], name='1', plot_graph=True):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"

    ncols = len(rank)
    nrows = len(list_network_files)
    fig = plt.figure(figsize=(ncols * (10 / 4), nrows * (18 / 4)), constrained_layout=False)
    k = motif_size

    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.15, hspace=0.05)
    for row in range(nrows):
        for col in range(ncols):
            n_components = rank[col]
            sub_rows = np.round(np.sqrt(n_components))
            sub_rows = sub_rows.astype(int)
            if sub_rows ** 2 == n_components:
                sub_cols = sub_rows
            else:
                sub_cols = sub_rows + 1

            if not ((k == 101) and (list_network_files[row] in [])):
                ### Load results file
                # Make nested gridspecs.
                inner_grid = outer_grid[row * ncols + col].subgridspec(sub_rows, sub_cols, wspace=0.2, hspace=0.1)

                ### Load results file
                ntwk = list_network_files[row]
                network_name = ntwk.replace('.txt', '')
                network_name = network_name.replace('.', '')
                print('!!!!! rank=', str(rank[col]))
                if network_name == "COVID_PPI":
                    k = 11
                else:
                    k = motif_size

                path = save_folder + '/full_result_' + str(network_name) + "_k_" + str(k) + "_r_" + str(
                    n_components) + ".npy"
                result_dict = np.load(path, allow_pickle=True).item()
                W = result_dict.get('Dictionary learned')
                At = result_dict.get('Code COV learned')
                k = result_dict.get('Motif size')

                # Add plot labels and remove remainder of axis.
                ax_outer = fig.add_subplot(outer_grid[row * ncols + col])
                # remove boarders
                ax_outer.spines['top'].set_visible(False)
                ax_outer.spines['right'].set_visible(False)
                ax_outer.spines['bottom'].set_visible(False)
                ax_outer.spines['left'].set_visible(False)
                if col == 0:
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_005':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{1}$'
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_01':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{2}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_50':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{1}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_100':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{2}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_25':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{1}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_50':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{2}$'
                    if network_name == 'Caltech36':
                        ntwk_label = '$\\textsc{\\texttt{Caltech}}$'
                    if network_name == 'MIT8':
                        ntwk_label = '$\\textsc{\\texttt{MIT}}$'
                    if network_name == 'UCLA26':
                        ntwk_label = '$\\textsc{\\texttt{UCLA}}$'
                    if network_name == 'Harvard1':
                        ntwk_label = '$\\textsc{\\texttt{Harvard}}$'
                    if network_name == 'COVID_PPI':
                        ntwk_label = '$\\textsc{\\texttt{Coronavirus}}$'
                    if network_name == 'facebook_combined':
                        ntwk_label = '$\\textsc{\\texttt{SNAP FB}}$'
                    if network_name == 'arxiv':
                        ntwk_label = '$\\textsc{\\texttt{arXiv}}$'
                    if network_name == 'node2vec_homosapiens_PPI':
                        ntwk_label = '$\\textsc{\\texttt{H. sapiens}}$'
                    if network_name == 'SBM1':
                        ntwk_label = '$\\textsc{\\texttt{SBM}}_{1}$'
                    if network_name == 'SBM2':
                        ntwk_label = '$\\textsc{\\texttt{SBM}}_{2}$'

                    ax_outer.set_ylabel(str(ntwk_label), fontsize=15)
                    ax_outer.yaxis.set_label_position('left')

                if row == 0:
                    ax_outer.set_title('$r$ = ' + str(n_components), fontsize=15)

                ax_outer.axes.xaxis.set_ticks([])
                ax_outer.axes.yaxis.set_ticks([])

                ### Use the code covariance matrix At to compute importance
                importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
                idx = np.argsort(importance)
                idx = np.flip(idx)

                ### Add subplot

                for i in range(sub_rows * sub_cols):
                    a = i // sub_cols
                    b = i % sub_cols
                    ax = fig.add_subplot(inner_grid[a, b])

                    if not plot_graph:
                        ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        fontsize = 10
                        if rank[col] >= 49:
                            fontsize = 8
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=fontsize)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        A_sub = W.T[idx[i]].reshape(k, k)
                        H = nx.from_numpy_matrix(A_sub)
                        G1 = nx.Graph()
                        for a in np.arange(k):
                            for b in np.arange(k):
                                u = list(H.nodes())[a]
                                v = list(H.nodes())[b]
                                if H.has_edge(u,v):
                                    if np.abs(a-b) == 1:
                                        G1.add_edge(u,v, color='r', weight=A_sub[a,b])
                                    else:
                                        G1.add_edge(u,v, color='b', weight=A_sub[a,b])

                        pos = nx.spring_layout(G1)
                        edges = G1.edges()
                        colors = [G1[u][v]['color'] for u,v in edges]
                        weights = [7*G1[u][v]['weight'] for u,v in edges]
                        if network_name == "COVID_PPI" and motif_size > 10:
                            #weights = [(20/(rank[col]+20)) *G1[u][v]['weight'] for u,v in edges]
                            weights = [7*G1[u][v]['weight'] for u,v in edges]

                        node_size=50/rank[col]
                        nx.draw(G1, with_labels=False, node_size=node_size, ax=ax, width=weights, edge_color=colors, label='Graph')

    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0)
    fig.savefig(save_folder + '/all_dictionaries_rank_' + str(name) + '.png', bbox_inches='tight')

def top_dictionaries_display(motif_sizes=[6, 11, 21, 51],
                             latent_motif_rank_list=[1,2],
                             fig_size=[15,8],
                             plot_graph=True):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
    list_network_files = ['COVID_PPI.txt',
                          'facebook_combined.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt',
                          'Caltech36.txt',
                          'MIT8.txt',
                          'UCLA26.txt',
                          'Harvard1.txt',
                          'true_edgelist_for_ER_5000_mean_degree_50.txt',
                          'true_edgelist_for_ER_5000_mean_degree_100.txt',
                          'true_edgelist_for_SW_5000_k_50_p_0.05.txt',
                          'true_edgelist_for_SW_5000_k_50_p_0.1.txt',
                          'true_edgelist_for_BA_5000_m_25.txt',
                          'true_edgelist_for_BA_5000_m_50.txt',
                          'SBM1.txt',
                          'SBM2.txt']

    nrows = len(motif_sizes)
    ncols = len(list_network_files)
    fig = plt.figure(figsize=fig_size, constrained_layout=False)
    n_components = 25
    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0, hspace=0)
    for row in trange(nrows):
        for col in range(ncols):
            if not ((motif_sizes[row] > 100) and (list_network_files[col] in ['COVID_PPI.txt'])):
                if not ((motif_sizes[row] > 60) and (list_network_files[col] in ['SBM1.txt'])):
                    ### Load results file
                    ntwk = list_network_files[col]
                    network_name = ntwk.replace('.txt', '')
                    network_name = network_name.replace('.', '')
                    path = save_folder + '/full_result_' + str(network_name) + "_k_" + str(motif_sizes[row]) + "_r_" + str(
                        n_components) + ".npy"
                    result_dict = np.load(path, allow_pickle=True).item()
                    W = result_dict.get('Dictionary learned')
                    At = result_dict.get('Code COV learned')
                    k = result_dict.get('Motif size')

            # Add plot labels and remove remainder of axis.
            ax_outer = fig.add_subplot(outer_grid[row * ncols + col])
            if col == 0:
                ax_outer.set_ylabel('$k$ = ' + str(motif_sizes[row]), fontsize=14)
                ax_outer.yaxis.set_label_position('left')

            if row == 0:
                if network_name == 'true_edgelist_for_SW_5000_k_50_p_005':
                    ntwk_label = '$\\textsc{\\texttt{WS}}_{1}$'
                if network_name == 'true_edgelist_for_SW_5000_k_50_p_01':
                    ntwk_label = '$\\textsc{\\texttt{WS}}_{2}$'
                if network_name == 'true_edgelist_for_ER_5000_mean_degree_50':
                    ntwk_label = '$\\textsc{\\texttt{ER}}_{1}$'
                if network_name == 'true_edgelist_for_ER_5000_mean_degree_100':
                    ntwk_label = '$\\textsc{\\texttt{ER}}_{2}$'
                if network_name == 'true_edgelist_for_BA_5000_m_25':
                    ntwk_label = '$\\textsc{\\texttt{BA}}_{1}$'
                if network_name == 'true_edgelist_for_BA_5000_m_50':
                    ntwk_label = '$\\textsc{\\texttt{BA}}_{2}$'
                if network_name == 'Caltech36':
                    ntwk_label = '$\\textsc{\\texttt{Caltech}}$'
                if network_name == 'MIT8':
                    ntwk_label = '$\\textsc{\\texttt{MIT}}$'
                if network_name == 'UCLA26':
                    ntwk_label = '$\\textsc{\\texttt{UCLA}}$'
                if network_name == 'Harvard1':
                    ntwk_label = '$\\textsc{\\texttt{Harvard}}$'
                if network_name == 'COVID_PPI':
                    ntwk_label = '$\\textsc{\\texttt{Coronavirus  }}$'
                if network_name == 'facebook_combined':
                    ntwk_label = '$\\textsc{\\texttt{SNAP FB}}$'
                if network_name == 'arxiv':
                    ntwk_label = '$\\textsc{\\texttt{arXiv}}$'
                if network_name == 'node2vec_homosapiens_PPI':
                    ntwk_label = '$\\textsc{\\texttt{H. sapiens}}$'
                if network_name == 'SBM1':
                    ntwk_label = '$\\textsc{\\texttt{SBM}}_{1}$'
                if network_name == 'SBM2':
                    ntwk_label = '$\\textsc{\\texttt{SBM}}_{2}$'


                ax_outer.set_title(str(ntwk_label), fontsize=12)

            ax_outer.axes.xaxis.set_ticks([])
            ax_outer.axes.yaxis.set_ticks([])

            ### Use the code covariance matrix At to compute importance
            importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
            idx = np.argsort(importance)
            idx = np.flip(idx)

            ### Add subplot
            inner_grid = outer_grid[row, col].subgridspec(len(latent_motif_rank_list), 1, wspace=0.1, hspace=0.1)

            if not ((motif_sizes[row] > 100) and (list_network_files[col] in ['COVID_PPI.txt'])):
                if not ((motif_sizes[row] > 60) and (list_network_files[col] in ['SBM1.txt'])):
                    for j in np.arange(len(latent_motif_rank_list)):
                        motif_rank = latent_motif_rank_list[j]


                        ax = fig.add_subplot(inner_grid[j, 0])
                        A_sub = W.T[idx[motif_rank - 1]].reshape(k, k)
                        if not plot_graph:
                            ax.imshow(W.T[idx[motif_rank - 1]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        else:
                            H = nx.from_numpy_matrix(A_sub)
                            G1 = nx.Graph()
                            for a in np.arange(k):
                                for b in np.arange(k):
                                    u = list(H.nodes())[a]
                                    v = list(H.nodes())[b]
                                    if H.has_edge(u,v):
                                        if np.abs(a-b) == 1:
                                            G1.add_edge(u,v, color='r', weight=A_sub[a,b])
                                        else:
                                            G1.add_edge(u,v, color='b', weight=A_sub[a,b])

                            pos = nx.spring_layout(G1)
                            edges = G1.edges()
                            colors = [G1[u][v]['color'] for u,v in edges]
                            weights = [5*G1[u][v]['weight'] for u,v in edges]
                            if motif_sizes[row] in range(10,20):
                                weights = [7*G1[u][v]['weight'] for u,v in edges]
                            if motif_sizes[row] in range(20,100):
                                weights = [10*G1[u][v]['weight'] for u,v in edges]


                            if network_name == "COVID_PPI" and motif_sizes[row] > 20:
                                weights = [(5/(motif_sizes[row]+20)) *G1[u][v]['weight'] for u,v in edges]

                            node_size=70/motif_sizes[row]
                            nx.draw(G1, with_labels=False, node_size=node_size, ax=ax, width=weights, edge_color=colors, label='Graph')

            # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=10)  # get the largest first
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(save_folder + '/top_latent_motifs' + '.png', bbox_inches='tight', dpi=300)


def few_dictionaries_display(list_network_files, motif_sizes=[21], name='1'):
    save_folder = "Network_dictionary/NDL_nofolding_dictionary_5"

    nrows = len(motif_sizes)
    ncols = len(list_network_files)
    fig = plt.figure(figsize=(ncols * (10 / 4), nrows * (18 / 4)), constrained_layout=False)
    n_components = 25
    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.05)
    for row in range(nrows):
        for col in range(ncols):
            sub_rows = np.round(np.sqrt(n_components))
            sub_rows = sub_rows.astype(int)
            if sub_rows ** 2 == n_components:
                sub_cols = sub_rows
            else:
                sub_cols = sub_rows + 1

            if not ((motif_sizes[row] == 101) and (list_network_files[col] in [])):
                ### Load results file
                # Make nested gridspecs.
                inner_grid = outer_grid[row * ncols + col].subgridspec(sub_rows, sub_cols, wspace=0.1, hspace=0.1)

                ### Load results file
                ntwk = list_network_files[col]
                network_name = ntwk.replace('.txt', '')
                network_name = network_name.replace('.', '')
                print('!!!!!', str(motif_sizes[row]))
                path = save_folder + '/full_result_' + str(network_name) + "_k_" + str(motif_sizes[row]) + "_r_" + str(
                    n_components) + "_Pivot.npy"
                result_dict = np.load(path, allow_pickle=True).item()
                W = result_dict.get('Dictionary learned')
                At = result_dict.get('Code COV learned')
                k = result_dict.get('Motif size')

                # Add plot labels and remove remainder of axis.
                ax_outer = fig.add_subplot(outer_grid[row * ncols + col])
                # remove boarders
                ax_outer.spines['top'].set_visible(False)
                ax_outer.spines['right'].set_visible(False)
                ax_outer.spines['bottom'].set_visible(False)
                ax_outer.spines['left'].set_visible(False)
                if row == 0:
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_005':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{1}$'
                    if network_name == 'true_edgelist_for_SW_5000_k_50_p_01':
                        ntwk_label = '$\\textsc{\\texttt{WS}}_{2}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_50':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{1}$'
                    if network_name == 'true_edgelist_for_ER_5000_mean_degree_100':
                        ntwk_label = '$\\textsc{\\texttt{ER}}_{2}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_25':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{1}$'
                    if network_name == 'true_edgelist_for_BA_5000_m_50':
                        ntwk_label = '$\\textsc{\\texttt{BA}}_{2}$'
                    if network_name == 'Caltech36':
                        ntwk_label = '$\\textsc{\\texttt{Caltech}}$'
                    if network_name == 'MIT8':
                        ntwk_label = '$\\textsc{\\texttt{MIT}}$'
                    if network_name == 'UCLA26':
                        ntwk_label = '$\\textsc{\\texttt{UCLA}}$'
                    if network_name == 'Harvard1':
                        ntwk_label = '$\\textsc{\\texttt{Harvard}}$'
                    if network_name == 'COVID_PPI':
                        ntwk_label = '$\\textsc{\\texttt{Coronavirus}}$'
                    if network_name == 'facebook_combined':
                        ntwk_label = '$\\textsc{\\texttt{SNAP FB}}$'
                    if network_name == 'arxiv':
                        ntwk_label = '$\\textsc{\\texttt{arXiv}}$'
                    if network_name == 'node2vec_homosapiens_PPI':
                        ntwk_label = '$\\textsc{\\texttt{H. sapiens}}$'

                    ax_outer.set_title(str(ntwk_label), fontsize=15)

                if col == 0:
                    ax_outer.set_ylabel('$k$ = ' + str(k), fontsize=15)
                    ax_outer.yaxis.set_label_position('left')

                ax_outer.axes.xaxis.set_ticks([])
                ax_outer.axes.yaxis.set_ticks([])

                ### Use the code covariance matrix At to compute importance
                importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
                idx = np.argsort(importance)
                idx = np.flip(idx)

                ### Add subplot

                for i in range(sub_rows * sub_cols):
                    a = i // sub_cols
                    b = i % sub_cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=10)  # get the largest first
                    ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0)
    fig.savefig('few_dictionaries_' + str(name) + '.pdf', bbox_inches='tight')



def diplay_ROC_plots(path):
    # path = "Network_dictionary/NDL_denoising_1"
    file_list = os.listdir(path=path)
    n_edges = {}
    n_edges.update({'Caltech36': 16651})
    n_edges.update({'facebook': 88234})
    n_edges.update({'arxiv': 198110})
    n_edges.update({'COVID': 2481})
    n_edges.update({'node2vec': 390633})

    for file in file_list:
        ROC_dict = np.load(path + "/" + file, allow_pickle=True).item()
        print('file', file)
        items = file.split('_')
        ROC_dict.update({'Network name': items[items.index('dict') + 1]})
        ROC_dict.update({'# edges of original ntwk': n_edges.get(items[2])})
        ROC_dict.update({'Use_corrupt_dict': items[items.index('n') - 2]})
        ROC_dict.update({'n_nodes': items[items.index('n') - 1]})
        ROC_dict.update({'noise_type': items[items.index('noisetype') + 1]})
        ROC_dict.update({'n_corrupt_edges': items[items.index('edges') + 1]})
        ROC_dict.update({'denoising_iter': items[items.index('iter') + 1]})
        np.save(path + "/" + file, ROC_dict)

    # print("!!! file list")

    ### Make gridspec
    fig1 = plt.figure(figsize=(14, 7), constrained_layout=False)
    gs1 = fig1.add_gridspec(nrows=8, ncols=18, wspace=1.5, hspace=1)

    font = font_manager.FontProperties(family="Times New Roman", size=11)

    for ntwk in ["Caltech36", "CaltechFromMIT", "coronavirus", "arxiv", "facebook", "homosapiens"]:
        # add subplot
        if ntwk == "Caltech36" or ntwk == "CaltechFromMIT":
            ax = fig1.add_subplot(gs1[:, :6])
        elif ntwk == "coronavirus":
            ax = fig1.add_subplot(gs1[4:8, 6:10])
        elif ntwk == "facebook":
            ax = fig1.add_subplot(gs1[0:4, 6:10])
        elif ntwk == "homosapiens":
            ax = fig1.add_subplot(gs1[0:4, 10:14])
        elif ntwk == "arxiv":
            ax = fig1.add_subplot(gs1[0:4, 14:18])


        network_name = ntwk
        loc = "dict"
        if ntwk == "CaltechFromMIT":
            network_name = "MIT"
            loc = "Use"
        if ntwk == "homosapiens":
            network_name = "node2vec"
        if ntwk == "coronavirus":
            network_name = "COVID"


        sub_list_files_pos = []
        sub_list_files_neg = []
        for file in file_list:
            split_list = file.split('_')
            if split_list[split_list.index(loc) + 1] == str(network_name):
                if not (network_name=="Caltech36" and split_list[split_list.index(loc) + 3] == "MIT"):
                    if split_list[split_list.index('noisetype') + 1] == 'ER':
                        sub_list_files_pos.append(file)
                    elif split_list[split_list.index('noisetype') + 1] == '-ER':
                        sub_list_files_neg.append(file)

        for noise_rate in [0.1, 0.5]:
            for list in [sub_list_files_pos, sub_list_files_neg]:
                for file in list:
                    ROC_single = np.load(path + "/" + file, allow_pickle=True).item()
                    percentage = 10*np.round(10 * float(ROC_single.get('n_corrupt_edges')) / float(
                        ROC_single.get('# edges of original ntwk'))).astype(int)

                    if ((noise_rate == 0.1) and (percentage < 15)) or ((noise_rate == 0.5) and (percentage > 15)):
                        print('!!!!!!! %.2f, %i' % (noise_rate, percentage))

                        if list == sub_list_files_pos:
                            signed_pct = "$+" + str(percentage) + "\%$"
                            line_style = '-'
                        else:
                            signed_pct = "$-" + str(percentage) + "\%$"
                            line_style = '--'

                        print('!!!!!!', ntwk)

                        if ntwk == "CaltechFromMIT":
                            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls=line_style, marker="*",
                                    label=str(signed_pct) + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))
                            print('ntwk', ntwk)
                        else:
                            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls=line_style,
                            label = str(signed_pct) + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

                        if ntwk=="Caltech36" or ntwk == "CaltechFromMIT":
                            ax.set_xlabel('false-positive rate', font=font, fontsize=15)
                            ax.set_ylabel('true-positive rate', font=font, fontsize=15)
                            ax.legend(prop=font)
                            ax.set_title('$\\textsc{\\texttt{Caltech}}$', font=font, size=15)

                        elif ntwk=="coronavirus":
                            ax.legend(prop=font)
                            ax.set_xlabel('false-positive rate', font=font, fontsize=15)
                            ax.set_title('$\\textsc{\\texttt{Coronavirus PPI}}$', font=font, size=15)

                        elif ntwk=="arxiv":
                            ax.set_ylabel('true-positive rate', font=font, fontsize=15)
                            ax.legend(prop=font)
                            ax.yaxis.set_label_position("right")
                            ax.set_title('$\\textsc{\\texttt{arXiv}}$', font=font, size=15)

                        elif ntwk == "facebook":
                            ax.set_xticks([])
                            ax.legend(prop=font)
                            ax.set_title('$\\textsc{\\texttt{SNAP Facebook}}$', font=font, size=15)

                        elif ntwk == "homosapiens":
                            ax.legend(prop=font)
                            ax.set_title('$\\textsc{\\texttt{Homo sapiens PPI}}$', font=font, size=15)

                        ax.plot([0, 1], [0, 1], "--", color='#BFE3F4')

    fig1.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
    fig1.savefig('Network_dictionary/' + 'ROC_final_pivot.pdf', bbox_inches='tight')
    plt.show()


def recons_display_simple():
  # Load Data
  save_folder = "Network_dictionary/Figure3_Data"
  nodes = 5000

  list_of_nc = [i**2 for i in range(3,11)]

  facebook_networks_file_names = [ "MIT8",  "Harvard1", "UCLA26", "Caltech36"]
  scores_path_new = f"{save_folder}/"

  f_f_scores = np.zeros((len(list_of_nc),len(facebook_networks_file_names),len(facebook_networks_file_names)))
  for ind_nc, num_components in enumerate(list_of_nc):
    for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
      for ind_dic, network_dict_used in enumerate(facebook_networks_file_names):
      # for ind_dic, network_dict_used in enumerate(synth_networks):
        path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.txt"
        with open(path) as file:
          f_f_scores[ind_nc][ind_rec][ind_dic] = float(file.read())


  threshold_scores_path = f"{save_folder}/"


  #  Plotting Code
  facebook_network_titles = [ "$\\textsc{\\texttt{MIT}}$", "$\\textsc{\\texttt{Harvard}}$", "$\\textsc{\\texttt{UCLA}}$", "$\\textsc{\\texttt{Caltech}}$"]
  figsizeinches = 3
  line_style = ['--','-.',':']
  marker_style = ["^","o","*"]
  colors = ["#000000","#009292","#ffb6db",
   "#490092","#006ddb","#b66dff",
   "#920000","#db6d00","#24ff24",'#a9a9a9', "#FD5956", "#03719C", "#343837", "#B04E0F"]
  random.shuffle(colors)

  fig = plt.figure(figsize=(10.5, 3))
  gs = gridspec.GridSpec(1,4,width_ratios=[1,1,1,1])
  axs = {}


  i=3 # Caltech
  axs[i] = plt.subplot(gs[:,0])
  for j in range(len(facebook_network_titles)):
     axs[i].plot(list_of_nc,f_f_scores[:,i,j],label=facebook_network_titles[j],color=colors[6+j],alpha=0.8)
  axs[i].text(36,0.95,facebook_network_titles[i]+r"$\leftarrow X$",fontsize=12)
  axs[i].set_ylim(0.6,1)
  axs[i].set_xticks(list_of_nc)
  axs[i].set_xlabel(r"$r\;\; ($with $\theta=0.5)$")
  axs[i].legend(loc='lower right',fontsize='medium')

  for i in range(len(facebook_network_titles)-1): #All but caltech
    axs[i] = plt.subplot(gs[:,i+1])
    for j in range(len(facebook_network_titles)):
      axs[i].plot(list_of_nc,f_f_scores[:,i,j],label=facebook_network_titles[j],color=colors[6+j],alpha=0.8)
    axs[i].text(36,0.85,facebook_network_titles[i]+r"$\leftarrow X$",fontsize=12)
    axs[i].set_ylim(0.6,1)
    axs[i].set_xticks(list_of_nc)
    axs[i].set_xlabel(r"$r\;\; ($with $\theta=0.5)$")
    axs[i].legend(loc='lower right',fontsize='medium')


  plt.tight_layout()
  plt.savefig("full_recons_plot_simple.pdf",bbox_inches='tight')
  plt.clf()


def community_box_plot(path_subgraphs="Network_dictionary/NDL_rev1/community_data_subgraphs.npy",
                       path_latent_motifs="Network_dictionary/NDL_rev1/community_data_motifs.npy"):

    #data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
    #data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

    subgraphs_community_list = np.load(path_subgraphs, allow_pickle=True)
    latentmotifs_community_list = np.load(path_latent_motifs, allow_pickle=True)

    data_a = subgraphs_community_list
    data_b = latentmotifs_community_list


    ticks = ['$\\textsc{\\texttt{SNAP FB}}$',
                    '$\\textsc{\\texttt{arXiv}}$',
                    '$\\textsc{\\texttt{H. Sapiens}}$',
                    '$\\textsc{\\texttt{Caltech}}$',
                    '$\\textsc{\\texttt{MIT}}$',
                    '$\\textsc{\\texttt{UCLA}}$',
                    '$\\textsc{\\texttt{Harvard}}$',
                    '$\\textsc{\\texttt{ER}}_{1}$',
                    '$\\textsc{\\texttt{ER}}_{2}$',
                    '$\\textsc{\\texttt{WS}}_{1}$',
                    '$\\textsc{\\texttt{WS}}_{2}$',
                    '$\\textsc{\\texttt{BA}}_{1}$',
                    '$\\textsc{\\texttt{BA}}_{2}$',
                    '$\\textsc{\\texttt{SBM}}_{1}$',
                    '$\\textsc{\\texttt{SBM}}_{2}$']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=[13,5])

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='r+', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='b+', widths=0.6)
    set_box_color(bpl, 'r') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, 'b')



    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='r', label='subgraphs')
    plt.plot([], c='b', label='latent motifs')
    plt.legend(fontsize=12)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=11.5)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 22)
    plt.ylabel('community size', fontsize=13)
    plt.tight_layout()
    plt.savefig('boxcompare.pdf')


def recons_display():
    # Load Data
    save_folder = "Network_dictionary/recons_plot_data"
    list_of_nc = [i ** 2 for i in range(3, 11)]

    nodes = 5000
    #p_values_ER = [50/(nodes-1), 100/(nodes-1)]
    p_values_ER = [100/(nodes-1)]
    ER = [ f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p*(nodes-1))}" for p in p_values_ER]

    #p_values_SW = [0.05, 0.1]
    p_values_SW = [0.1]
    k_values_SW = [50]
    SW = [ f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p,2)).replace('.','')}" for k in k_values_SW for p in p_values_SW]

    m_values_BA = [50]
    BA = [ f"true_edgelist_for_BA_{nodes}_m_{m}"for m in m_values_BA]

    SBM = ["SBM2"]
    synth_network_file_names =   ER+SW+BA+SBM

    synth_network_titles = ["$\\textsc{\\texttt{ER}}_{2}$",
                            "$\\textsc{\\texttt{WS}}_{2}$",
                            "$\\textsc{\\texttt{BA}}_{2}$",
                            "$\\textsc{\\texttt{SBM}}_{2}$"]
    facebook_networks_file_names = ["MIT8", "Harvard1", "UCLA26", "Caltech36"]
    scores_path_new = f"{save_folder}/"

    f_f_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(facebook_networks_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(facebook_networks_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.npy"
                a = np.load(path, allow_pickle=True).item()
                f_f_scores[ind_nc][ind_rec][ind_dic] = a.get("Jaccard_recons_accuracy")

    f_s_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(synth_network_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(synth_network_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.npy"
                a = np.load(path, allow_pickle=True).item()
                f_s_scores[ind_nc][ind_rec][ind_dic] = a.get("Jaccard_recons_accuracy")
                #print('synth accuracy', a.get("Jaccard_recons_accuracy"))


    threshold_scores_path = f"{save_folder}/"

    real_networks_file_names = ['COVID_PPI', 'node2vec_homosapiens_PPI', 'facebook_combined', "arxiv", "Caltech36"]
    real_network_titles = ["$\\textsc{\\texttt{Coronavirus}}$", "$\\textsc{\\texttt{H. sapiens}}$",
                           "$\\textsc{\\texttt{SNAP FB}}$", "$\\textsc{\\texttt{arXiv}}$",
                           "$\\textsc{\\texttt{Caltech}}$"]
    self_recons_score_threshold = np.zeros((len(real_networks_file_names), 101))
    for ind_network, network in enumerate(real_networks_file_names):
        self_recons_score_threshold[ind_network] = np.loadtxt(
            f"{threshold_scores_path}self_recons_{network}_vary_threshold.txt")

    #  Plotting Code
    facebook_network_titles = ["$\\textsc{\\texttt{MIT}}$", "$\\textsc{\\texttt{Harvard}}$",
                               "$\\textsc{\\texttt{UCLA}}$", "$\\textsc{\\texttt{Caltech}}$"]
    figsizeinches = 3
    line_style = ['--', '-.', ':']
    marker_style = ["^", "o", "*", "|", "x"]
    colors = ["#000000", "#009292", "#ffb6db",
              "#490092", "#006ddb", "#b66dff",
              "#920000", "#db6d00", "#24ff24", '#a9a9a9', "#FD5956", "#03719C", "#343837", "#B04E0F"]
    random.shuffle(colors)

    fig = plt.figure(figsize=(10.5, 6))
    gs = gridspec.GridSpec(3, 3, width_ratios=[5, 5, 4])

    ax0 = plt.subplot(gs[:, 0]) # panel (a)
    for j in range(len(real_network_titles)):
        if j == 4:
            ax0.plot(np.linspace(0.0, 1.0, num=101), self_recons_score_threshold[j], label=real_network_titles[j],
                     alpha=0.8, color=colors[9], marker=marker_style[j], markevery=5)
        else:
            ax0.plot(np.linspace(0.0, 1.0, num=101), self_recons_score_threshold[j], label=real_network_titles[j],
                     alpha=0.8, color=colors[10 + j], marker=marker_style[j], markevery=5)  # ,marker=marker_style[j//2],markevery=3)
    ax0.text(0.8, 0.95, r"$X \leftarrow X$", fontsize=12)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_xlabel(r"$\theta \;\; ($with $r=25)$")
    ax0.set_ylabel("accuracy")
    ax0.legend(loc='lower center', fontsize='medium')

    ax1 = plt.subplot(gs[:, 1])
    i = 3  # Caltech
    marker_style = ["^", "o", "|", "*"]
    for j in range(len(facebook_network_titles)):
        ax1.plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[6 + j], alpha=0.8, marker=marker_style[j])
    for j in range(len(synth_network_titles)):
        ax1.plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                 linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)
    ax1.text(9, 0.9625, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
    ax1.set_ylim(0.4, 1)
    ax1.set_xticks(list_of_nc)
    ax1.set_xlabel(r"$r\;\; ($with $\theta=0.4)$")
    ax1.legend(loc='lower right', fontsize='medium')

    axs = {}
    for i in range(len(facebook_network_titles) - 1):  # All but Caltech
        axs[i] = plt.subplot(gs[i, 2])
        for j in range(len(facebook_network_titles)):
            axs[i].plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[6 + j],
                        alpha=0.8, marker=marker_style[j])
        for j in range(len(synth_network_titles)):
            axs[i].plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                        linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)
        axs[i].text(40, 0.65, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
        axs[i].set_ylim(0.6, 1)
        axs[i].set_xticks(list_of_nc)
        if i == 2:
            axs[i].set_xlabel(r"$r\;\; ($with $\theta=0.4)$")

    plt.tight_layout()
    plt.savefig("full_recons_plot_scratch.pdf", bbox_inches='tight')
    print('!!! Fig3 saved')
    #plt.clf()
    plt.show()



def recons_error_bd_display():
    # Load Data
    save_folder = "Network_dictionary/recons_plot_data1"
    list_of_nc = [i ** 2 for i in range(3, 11)]
    print('!!!!!@@@@@@')


    nodes = 5000
    #p_values_ER = [50/(nodes-1), 100/(nodes-1)]
    p_values_ER = [100/(nodes-1)]
    ER = [ f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p*(nodes-1))}" for p in p_values_ER]

    #p_values_SW = [0.05, 0.1]
    p_values_SW = [0.1]
    k_values_SW = [50]
    SW = [ f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p,2)).replace('.','')}" for k in k_values_SW for p in p_values_SW]

    m_values_BA = [50]
    BA = [ f"true_edgelist_for_BA_{nodes}_m_{m}"for m in m_values_BA]

    SBM = ["SBM2"]
    synth_network_file_names =   ER+SW+BA+SBM

    synth_network_titles = ["$\\textsc{\\texttt{ER}}_{2}$",
                            "$\\textsc{\\texttt{WS}}_{2}$",
                            "$\\textsc{\\texttt{BA}}_{2}$",
                            "$\\textsc{\\texttt{SBM}}_{2}$"]
    facebook_networks_file_names = ["Caltech36", "UCLA26", "MIT8", "Harvard1"]
    scores_path_new = f"{save_folder}/"

    f_f_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(facebook_networks_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(facebook_networks_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}recons_error_bd_{network_to_recons}_from_{network_dict_used}_k_{20}.npy"
                a = np.load(path, allow_pickle=True).item().get(str(num_components))
                f_f_scores[ind_nc][ind_rec][ind_dic] = 1-a

    f_s_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(synth_network_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(synth_network_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}recons_error_bd_{network_to_recons}_from_{network_dict_used}_k_{20}.npy"
                a = np.load(path, allow_pickle=True).item().get(str(num_components))
                f_s_scores[ind_nc][ind_rec][ind_dic] = 1-a
                #print('synth accuracy', a.get("Jaccard_recons_accuracy"))


    #  Plotting Code
    facebook_network_titles = ["$\\textsc{\\texttt{Caltech}}$", "$\\textsc{\\texttt{UCLA}}$", "$\\textsc{\\texttt{MIT}}$", "$\\textsc{\\texttt{Harvard}}$"]
    figsizeinches = 3
    line_style = ['--', '-.', ':']
    marker_style = ["^", "o", "*", "|", "x"]
    colors = ["#000000", "#009292", "#ffb6db",
              "#490092", "#006ddb", "#b66dff",
              "#920000", "#db6d00", "#24ff24", '#a9a9a9', "#FD5956", "#03719C", "#343837", "#B04E0F"]
    random.shuffle(colors)

    fig = plt.figure(figsize=(10.5, 5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[5, 5, 4])#, wspace=0.2, hspace=0.2)

    for i in np.arange(4):
        if i in [0,1]: # Caltech or UCLA
            ax1 = plt.subplot(gs[:, i])
            marker_style = ["^", "o", "|", "*"]
            for j in range(len(facebook_network_titles)):
                ax1.plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[j], alpha=0.8, marker=marker_style[j])
            for j in range(len(synth_network_titles)):
                ax1.plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                         linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)
            ax1.text(9, 0.93, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
            ax1.set_ylim(0, 1)
            if i == 0:
                ax1.set_ylim(-0.3, 1)
            elif i == 1:
                ax1.set_ylim(-0.2, 1)

            ax1.set_xticks(list_of_nc)
            ax1.set_xlabel(r"$r$")
            if i == 0:
                ax1.set_ylabel(r"lower bound for Jaccard recons. accuracy", fontsize=13)
            #ax1.legend(loc='lower right', fontsize='medium', ncol=2)
                #ax1.legend(fontsize='medium', ncol=1, bbox_to_anchor=(1, 0.5))
        else:
            ax1 = plt.subplot(gs[i-2, 2])
            for j in range(len(facebook_network_titles)):
                ax1.plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[j],
                            alpha=0.8, marker=marker_style[j])
            for j in range(len(synth_network_titles)):
                ax1.plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                            linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)

            if i == 2:
                ax1.text(60, 0.25, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
            elif i == 3:
                ax1.text(60, 0.1, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)

            ax1.set_ylim(0, 1)
            ax1.set_xticks(list_of_nc)
            if i in [1,3]:
                ax1.set_xlabel(r"$r$")
            if i == 3:
                ax1.set_ylim(-0.15, 1)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=1, bbox_to_anchor=(1.09, 0.7))


    plt.tight_layout()
    plt.savefig("recons_bd_plot.pdf", bbox_inches='tight')
    print('!!! Fig3 saved')
    #plt.clf()
    plt.show()


#### Denoising plots

def bar_plot(ax, data, colors=None, xticks = None,
             total_width=0.5, single_width=1, legend=True,
             ylim = None, hline=None):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        colors = ["#000000","#009292","#ffb6db",
                   "#490092","#006ddb","#b66dff",
                   "#920000","#db6d00","#24ff24",'#a9a9a9', "#FD5956", "#03719C", "#343837", "#B04E0F"]
        #random.shuffle(colors)

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        labels0 = data.keys()
        labels = []
        for key in labels0:
            if key.split("_")[0] == "preferential":
                key = "$\\textsc{preferential attachment}$"
            elif key.split("_")[0] == "adamic":
                key = "$\\textsc{Adamic--Adar index}$"
            elif key.split("_")[0] == "jaccard":
                key = "$\\textsc{Jaccard index}$"
            elif key == "spectral":
                key = "$\\textsc{spectral embedding}$"
            elif key == "DeepWalk":
                key = "$\\textsc{DeepWalk}$"
            elif key == "node2vec":
                key = "$\\textsc{node2vec}$"
            elif key == "NDL+NDR":
                key = "$\\textsc{NDL+NDR}$"



            labels.append(key)
        print("labels", labels)
        ax.legend(bars, labels, ncol=2)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)),xticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(y=hline, color='grey', linestyle='--')

    """
    # Usage example:
    data = {
        "a": [1, 2, 3, 2, 1],
        "b": [2, 3, 4, 3, 1],
        "c": [3, 2, 1, 4, 2],
        "d": [5, 9, 2, 1, 8],
        "e": [1, 3, 2, 2, 3],
        "f": [4, 3, 1, 1, 4],
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, xticks=["a0", "b0", "c0", "d0", "e0"], total_width=.8, single_width=.9)
    plt.show()
    """

from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve

def display_denoising_bar(list_network_files=['COVID_PPI.txt',
                          'Caltech36.txt',
                          'facebook_combined.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt'],
                          noise_type_list=[ "WS", "ER", "-ER"],
                          methods_list = ["jaccard", "preferential_attachment", "adamic_adar_index", "spectral", "DeepWalk", "node2vec", "NDL+NDR"],
                         save_path="Network_dictionary/barplot/barplot1.pdf",
                         title=None,
                         metric = "AUC",
                         fig_size = [15,4.5]):

    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/barplot"

    Ndict_wspace = 0.05
    Ndict_hspace = 0.05
    #fig_size = [5,15]

    fig = plt.figure(figsize=fig_size, constrained_layout=False)
    outer_grid = gridspec.GridSpec(nrows=1, ncols=len(noise_type_list), wspace=0.1, hspace=0.4)

    for i in trange(len(noise_type_list)):
        noise_type = noise_type_list[i]

        if noise_type == "WS":
            noise_type0 = r"$+$WS"
        elif noise_type == "ER":
            noise_type0 = r"$+$ER"
        elif noise_type == "-ER":
            noise_type0 = r"$-$ER"


        ax = plt.Subplot(fig, outer_grid[0,i])
        ax.set_title("Noise type: {}".format(noise_type0))
        ax.axis('off')
        fig.add_subplot(ax)

        inner_grid = outer_grid[0,i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
        ax = fig.add_subplot(inner_grid[0, 0])


        # prepare output file
        output_dict = {}
        output_dict_new = {}

        for method in methods_list:
            output_dict.update({method: [0]*len(list_network_files)})
            for i in np.arange(len(list_network_files)):

                ntwk = list_network_files[i]
                network_name = ntwk.replace('.txt', '')
                network_name = network_name.replace('.', '')

                #print("method={}, ntwk={}".format(method, network_name))

                path = save_folder + "/" + network_name + "_noisetype_" + noise_type +"_"+ "output_dict.npy"
                a = np.load(path, allow_pickle=True).item()

                auc_list = []

                c = 0
                for key in list(a.keys()):
                    method0 = key.split("_")[0]
                    if key.split("_")[0] == "preferential":
                        method0 = "preferential_attachment"
                        method1 = "$\\textsc{preferential attachment}$"
                    elif key.split("_")[0] == "adamic":
                        method0 = "adamic_adar_index"
                        method1 = "$\\textsc{Adamic--Adar index}$"
                    elif key.split("_")[0] == "jaccard":
                        method1 = "$\\textsc{Jaccard index}$"

                    if method0 == method:
                        method1 = "$\\textsc{method}$"
                        c += 1

                if c ==0:
                    auc_list.append(0)
                else:
                    for key in list(a.keys()):
                        method0 = key.split("_")[0]
                        if key.split("_")[0] == "preferential":
                            method0 = "preferential_attachment"
                            method1 = "$\\textsc{preferential attachment}$"
                        elif key.split("_")[0] == "adamic":
                            method0 = "adamic_adar_index"
                            method1 = "$\\textsc{Adamic-Adar index}$"
                        elif key.split("_")[0] == "jaccard":
                            method1 = "$\\textsc{Jaccard index}$"

                        if method0 == method:
                            method1 = "$\\textsc{method}$"
                            auc = a.get(str(key)).get(metric)
                            if auc is None:
                                auc = 0
                            auc_list.append(auc)

                auc = np.mean(np.asarray(auc_list))

                l = output_dict.get(method)
                l[i] = auc
                output_dict.update({method: l})
                #output_dict_new.update({method1: l})

        xticks = ['$\\textsc{\\texttt{Coronavirus}}$', '$\\textsc{\\texttt{Caltech}}$', '$\\textsc{\\texttt{SNAP FB}}$', '$\\textsc{\\texttt{arXiv}}$', '$\\textsc{\\texttt{H. Sapiens}}$']
        bar_plot(ax, output_dict, xticks=xticks, total_width=.7,
                 single_width=.8, ylim=[0, 1.35], hline=1)


    #plt.suptitle(title, fontsize=15)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
