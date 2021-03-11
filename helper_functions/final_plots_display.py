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

def all_dictionaries_display(list_network_files, motif_sizes=[6, 11, 21, 51, 101], name='1'):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_nofolding_dictionary_all4"

    ncols = len(motif_sizes)
    nrows = len(list_network_files)
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
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=10)  # get the largest first
                    ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0)
    fig.savefig(save_folder + '/all_dictionaries_' + str(name) + '.pdf', bbox_inches='tight')


def all_dictionaries_display_rank(list_network_files, motif_size=21, rank=[9, 16, 25, 36, 49], name='1'):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_nofolding_fig3"

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
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    fontsize = 10
                    if rank[col] >= 49:
                        fontsize = 8
                    ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=fontsize)  # get the largest first
                    ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0)
    fig.savefig(save_folder + '/all_dictionaries_rank_' + str(name) + '.pdf', bbox_inches='tight')


def top_dictionaries_display(motif_sizes=[6, 11, 21, 51, 101], latent_motif_rank=0):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_nofolding_dictionary_all4"
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
                          'true_edgelist_for_BA_5000_m_50.txt']

    nrows = len(motif_sizes)
    ncols = len(list_network_files)
    fig = plt.figure(figsize=(13, 5), constrained_layout=False)
    n_components = 25
    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.2)
    for row in range(nrows):
        for col in range(ncols):
            if not ((motif_sizes[row] == 101) and (list_network_files[col] in [])):
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
                if col == 0:
                    ax_outer.set_ylabel('$k$ = ' + str(k), fontsize=10)
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

                    ax_outer.set_title(str(ntwk_label), fontsize=12)

                ax_outer.axes.xaxis.set_ticks([])
                ax_outer.axes.yaxis.set_ticks([])

                ### Use the code covariance matrix At to compute importance
                importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
                idx = np.argsort(importance)
                idx = np.flip(idx)

                ### Add subplot

                ax = fig.add_subplot(outer_grid[row, col])
                ax.imshow(W.T[idx[latent_motif_rank - 1]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=10)  # get the largest first
                # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                ax.set_xticks([])
                ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(save_folder + '/top_latent_motifs_' + str(latent_motif_rank) + '.pdf', bbox_inches='tight')



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


def recons_display():
    # Load Data
    save_folder = "Network_dictionary/Figure3_Data"
    nodes = 5000

    list_of_nc = [i ** 2 for i in range(3, 11)]

    p_values_ER = [50 / (nodes - 1), 100 / (nodes - 1)]
    ER = [f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p * (nodes - 1))}" for p in p_values_ER]
    p_values_SW = [0.05, 0.1]
    k_values_SW = [50]
    SW = [f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p, 2)).replace('.', '')}" for k in k_values_SW for p in
          p_values_SW]
    m_values_BA = [25, 50]
    BA = [f"true_edgelist_for_BA_{nodes}_m_{m}" for m in m_values_BA]
    synth_network_file_names = ER + SW + BA
    synth_network_titles = ["$\\textsc{\\texttt{ER}}_{1}$", "$\\textsc{\\texttt{ER}}_{2}$",
                            "$\\textsc{\\texttt{WS}}_{1}$", "$\\textsc{\\texttt{WS}}_{2}$",
                            "$\\textsc{\\texttt{BA}}_{1}$", "$\\textsc{\\texttt{BA}}_{2}$"]
    facebook_networks_file_names = ["MIT8", "Harvard1", "UCLA26", "Caltech36"]
    scores_path_new = f"{save_folder}/"

    f_f_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(facebook_networks_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(facebook_networks_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.txt"
                with open(path) as file:
                    f_f_scores[ind_nc][ind_rec][ind_dic] = float(file.read())

    f_s_scores = np.zeros((len(list_of_nc), len(facebook_networks_file_names), len(synth_network_file_names)))
    for ind_nc, num_components in enumerate(list_of_nc):
        for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
            for ind_dic, network_dict_used in enumerate(synth_network_file_names):
                # for ind_dic, network_dict_used in enumerate(synth_networks):
                path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.txt"
                with open(path) as file:
                    f_s_scores[ind_nc][ind_rec][ind_dic] = float(file.read())

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
    ax0.text(0.0, 0.95, r"$X \leftarrow X$", fontsize=12)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_xlabel(r"$\theta \;\; ($with $r=25)$")
    ax0.set_ylabel("accuracy")
    ax0.legend(loc='lower center', fontsize='medium')

    ax1 = plt.subplot(gs[:, 1])
    i = 3  # Caltech
    for j in range(len(synth_network_titles)):
        ax1.plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                 linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)
    for j in range(len(facebook_network_titles)):
        ax1.plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[6 + j], alpha=0.8)
    ax1.text(9, 0.9625, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
    ax1.set_ylim(0.3, 1)
    ax1.set_xticks(list_of_nc)
    ax1.set_xlabel(r"$r\;\; ($with $\theta=0.5)$")
    ax1.legend(loc='lower right', fontsize='medium')

    axs = {}
    for i in range(len(facebook_network_titles) - 1):  # All but Caltech
        axs[i] = plt.subplot(gs[i, 2])
        for j in range(len(synth_network_titles)):
            axs[i].plot(list_of_nc, f_s_scores[:, i, j], label=synth_network_titles[j], alpha=0.8, color=colors[j],
                        linestyle=line_style[j // 2])  # ,marker=marker_style[j//2],markevery=3)
        for j in range(len(facebook_network_titles)):
            axs[i].plot(list_of_nc, f_f_scores[:, i, j], label=facebook_network_titles[j], color=colors[6 + j],
                        alpha=0.8)
        axs[i].text(36, 0.65, facebook_network_titles[i] + r" $\leftarrow X$", fontsize=12)
        axs[i].set_ylim(0.6, 1)
        axs[i].set_xticks(list_of_nc)
        if i == 2:
            axs[i].set_xlabel(r"$r\;\; ($with $\theta=0.5)$")

    plt.tight_layout()
    plt.savefig("full_recons_plot_scratch.pdf", bbox_inches='tight')
    print('!!! Fig3 saved')
    plt.clf()
