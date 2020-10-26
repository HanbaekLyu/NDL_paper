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
                    n_components) + ".npy"
                result_dict = np.load(path, allow_pickle=True).item()
                W = result_dict.get('Dictionary learned')
                At = result_dict.get('Code COV learned')
                k = result_dict.get('Motif size')

                # Add plot labels and remove remainder of axis.
                ax_outer = fig.add_subplot(outer_grid[row * ncols + col])
                if col == 0:
                    ax_outer.set_ylabel('k = ' + str(k), fontsize=10)
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


def diplay_ROC_plots():
    path = "Network_dictionary/ROC_data.npy"
    ROC_data = np.load(path, allow_pickle=True).item()

    ### Make gridspec
    fig1 = plt.figure(figsize=(14, 7), constrained_layout=False)
    gs1 = fig1.add_gridspec(nrows=8, ncols=18, wspace=1.5, hspace=1)

    font = font_manager.FontProperties(family="Times New Roman", size=11)

    for i in range(5):

        if i == 0:  ### Denoising Caltech

            ax = fig1.add_subplot(gs1[:, :6])

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_Caltech36_Use_corrupt_dict_True_769_n_corrupt_edges_1575_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_Caltech36_Use_corrupt_dict_True_769_n_corrupt_edges_7867_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_Caltech36_Use_corrupt_dict_True_769_n_corrupt_edges_1665_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_Caltech36_Use_corrupt_dict_True_769_n_corrupt_edges_8328_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            ax.plot(ROC_data.get('FPR_list')[2], ROC_data.get('TPR_list')[2], marker='*', markevery=5,
                    label='$+10\%$ ' + ', MIT8 dict.' + ' (AUC $\\approx$ %.3f)' % ROC_data.get('AUC_list')[2])
            ax.plot(ROC_data.get('FPR_list')[3], ROC_data.get('TPR_list')[3], marker='*', markevery=5,
                    label='$+50\%$ ' + ', MIT8 dict.' + ' (AUC $\\approx$ %.3f)' % ROC_data.get('AUC_list')[3])
            ax.plot([0, 1], [0, 1], "--", color='#BFE3F4')

            path = "Network_dictionary/ROC_josh_update/ROC_dict_Caltech36_01_50000_MIT.npy"
            ROC_drop_data1 = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_drop_data1.get('False positive rate'), ROC_drop_data1.get('True positive rate'), ls='--',
                    marker='*', markevery=5,
                    label='$-10\%$, MIT8 dict.' + ' (AUC $\\approx$  %.3f)' % ROC_drop_data1.get('AUC'))

            path = "Network_dictionary/ROC_josh_update/ROC_dict_Caltech36_05_50000_MIT.npy"
            ROC_drop_data1 = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_drop_data1.get('False positive rate'), ROC_drop_data1.get('True positive rate'), ls='--',
                    marker='*', markevery=5,
                    label='$-50\%$, MIT8 dict.' + ' (AUC $\\approx$ %.3f)' % ROC_drop_data1.get('AUC'))

            ax.set_xlabel('False-positive rate', font=font, fontsize=15)
            ax.set_ylabel('True-positive rate', font=font, fontsize=15)
            ax.legend(prop=font)
            ax.set_title('$\\textsc{\\texttt{Caltech}}$', font=font, size=15)

        if i == 2:  ### Denoising SNAP facebook dataset
            ax = fig1.add_subplot(gs1[0:4, 6:10])
            ax.plot([0, 1], [0, 1], "--", color='#BFE3F4')

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_facebook_combined_Use_corrupt_dict_True_4039_n_corrupt_edges_8740_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_facebook_combined_Use_corrupt_dict_True_4039_n_corrupt_edges_43633_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_facebook_combined_Use_corrupt_dict_True_4039_n_corrupt_edges_8823_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_facebook_combined_Use_corrupt_dict_True_4039_n_corrupt_edges_44117_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            ax.legend(prop=font)

            # ax.set_xlabel('False-positive rate', font=font, fontsize=15)
            # ax.set_ylabel('True positive rate', fontsize=11)
            # ax.set_yticks([])
            ax.set_xticks([])
            ax.legend(prop=font)
            ax.set_title('$\\textsc{\\texttt{SNAP Facebook}}$', font=font, size=15)

        if i == 1:  ### Denoising COVID PPI
            ax = fig1.add_subplot(gs1[4:8, 6:10])

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_COVID_PPI_Use_corrupt_dict_True_1555_n_corrupt_edges_244_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_COVID_PPI_Use_corrupt_dict_True_1555_n_corrupt_edges_1240_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_COVID_PPI_Use_corrupt_dict_True_1555_n_corrupt_edges_248_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_COVID_PPI_Use_corrupt_dict_True_1555_n_corrupt_edges_496_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-20\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            ax.legend(prop=font)

            ax.set_xlabel('False-positive rate', font=font, fontsize=15)
            # ax.set_ylabel('True-positive rate', font=font, fontsize=15)
            # ax.set_yticks([])
            # ax.yaxis.tick_right()
            # ax.yaxis.set_label_position('right')
            # ax.set_xticks([])
            ax.legend(prop=font)
            ax.set_title('$\\textsc{\\texttt{Coronavirus PPI}}$', font=font, size=15)

        if i == 3:  ### Denoising Homo Sapiens PPI
            ax = fig1.add_subplot(gs1[0:4, 10:14])

            """
            ax.plot(ROC_data.get('FPR_list')[8], ROC_data.get('TPR_list')[8],
                    label='$+10\%$ ' + " (AUC $\\approx$ %.3f)" % ROC_data.get('AUC_list')[8])
            ax.plot(ROC_data.get('FPR_list')[9], ROC_data.get('TPR_list')[9],
                    # label='$+50\%$ ' + "(AUC $\\approx$ %.2f)" % ROC_data.get('AUC_list')[9])
                    label='$+50\%$ ' + " (AUC $\\approx$ %.3f)" % ROC_data.get('AUC_list')[9])
            ax.plot([0, 1], [0, 1], "--", color='#BFE3F4')
            """

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_node2vec_homosapiens_PPI_Use_corrupt_dict_True_24379_n_corrupt_edges_38985_noisetype_ER_edges_iter_200000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$+10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_node2vec_homosapiens_PPI_Use_corrupt_dict_True_24379_n_corrupt_edges_194967_noisetype_ER_edges_iter_200000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$+50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_node2vec_homosapiens_PPI_Use_corrupt_dict_True_24379_n_corrupt_edges_39039_noisetype_-ER_edges_iter_300000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_node2vec_homosapiens_PPI_Use_corrupt_dict_True_24379_n_corrupt_edges_195198_noisetype_-ER_edges_iter_300000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            ax.legend(fontsize=11)

            # ax.set_xlabel('False-positive rate', font=font, size=15)
            # ax.set_ylabel('True positive rate', fontsize=11)
            # ax.set_yticks([])
            #  ax.set_ylabel('True-positive rate', font=font, size=15)
            # ax.yaxis.set_label_position('right')
            ax.legend(prop=font)
            ax.set_title('$\\textsc{\\texttt{Homo sapiens PPI}}$', font=font, size=15)

        if i == 4:  ### Denoising arXiv network
            ax = fig1.add_subplot(gs1[0:4, 14:18])

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_arxiv_Use_corrupt_dict_True_18772_n_corrupt_edges_19791_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_arxiv_Use_corrupt_dict_True_18772_n_corrupt_edges_98945_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='-',
                    label='$+50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            """
            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_arxiv_Use_corrupt_dict_True_18772_n_corrupt_edges_19811_noisetype_-ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + ' (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))

            path = "Network_dictionary/NDL_nofolding_denoising5/ROC_dict_arxiv_Use_corrupt_dict_True_18772_n_corrupt_edges_98945_noisetype_ER_edges_iter_100000_Glauber_recons_False.npy"
            ROC_single = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_single.get('False positive rate'), ROC_single.get('True positive rate'), ls='--',
                    label='$-50\%$ ' + ' (AUC $\\approx$ %.3f)' % ROC_single.get('AUC'))
            """

            path = "Network_dictionary/ROC_josh_update/ROC_dict_arxiv_01_200000.npy"
            ROC_drop_data1 = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_drop_data1.get('False positive rate'), ROC_drop_data1.get('True positive rate'), ls='--',
                    label='$-10\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_drop_data1.get('AUC'))

            path = "Network_dictionary/ROC_josh_update/ROC_dict_arxiv_05_200000.npy"
            ROC_drop_data1 = np.load(path, allow_pickle=True).item()
            ax.plot(ROC_drop_data1.get('False positive rate'), ROC_drop_data1.get('True positive rate'), ls='--',
                    label='$-50\%$ ' + '  (AUC $\\approx$ %.3f)' % ROC_drop_data1.get('AUC'))

            ax.legend(fontsize=11)
            ax.set_ylabel('True-positive rate', font=font, size=15)
            ax.yaxis.set_label_position('right')
            # ax.set_xlabel('False-positive rate', font=font, size=15)
            # ax.set_yticks([])
            ax.legend(prop=font)
            ax.set_title('$\\textsc{\\texttt{arXiv ASTRO-PH}}$', font=font, size=15)

    fig1.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
    fig1.savefig('Network_dictionary/' + 'ROC_final_pivot.pdf', bbox_inches='tight')
    plt.show()

def recons_display():
  # Load Data
  save_folder = "Network_dictionary/test"
  nodes = 5000

  list_of_nc = [i**2 for i in range(3,11)]

  p_values_ER = [50/(nodes-1), 100/(nodes-1)]
  ER = [ f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p*(nodes-1))}" for p in p_values_ER]
  p_values_SW = [0.05, 0.1]
  k_values_SW = [50]
  SW = [ f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p,2)).replace('.','')}" for k in k_values_SW for p in p_values_SW]
  m_values_BA = [25,50]
  BA = [ f"true_edgelist_for_BA_{nodes}_m_{m}"for m in m_values_BA]
  synth_network_file_names =   ER+SW+BA
  synth_network_titles = ["ER 1","ER 2","WS 1","WS 2","BA 1","BA 2"]
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

  f_s_scores = np.zeros((len(list_of_nc),len(facebook_networks_file_names),len(synth_network_file_names)))
  for ind_nc, num_components in enumerate(list_of_nc):
    for ind_rec, network_to_recons in enumerate(facebook_networks_file_names):
      for ind_dic, network_dict_used in enumerate(synth_network_file_names):  
      # for ind_dic, network_dict_used in enumerate(synth_networks):
        path = f"{scores_path_new}{network_to_recons}_recons_score_for_nc_{num_components}_from_{network_dict_used}.txt"
        with open(path) as file: 
          f_s_scores[ind_nc][ind_rec][ind_dic] = float(file.read())

  threshold_scores_path = f"{save_folder}/"

  real_networks_file_names = ['COVID_PPI', 'node2vec_homosapiens_PPI','facebook_combined', "arxiv",  "Caltech36"]
  real_network_titles = [ "Coronavirus", "$\it{H. sapiens}$", "Facebook", "arXiv", "Caltech"]
  self_recons_score_threshold = np.zeros((len(real_networks_file_names),101))
  for ind_network, network in enumerate(real_networks_file_names):
    self_recons_score_threshold[ind_network] = np.loadtxt(f"{threshold_scores_path}self_recons_{network}_vary_threshold.txt")


  
  



  #  Plotting Code
  facebook_network_titles = [ "MIT", "Harvard", "UCLA", "Caltech"]
  figsizeinches = 3
  line_style = ['--','-.',':']
  marker_style = ["^","o","*"]
  colors = ["#000000","#009292","#ffb6db",
   "#490092","#006ddb","#b66dff",
   "#920000","#db6d00","#24ff24",'#a9a9a9', "#FD5956", "#03719C", "#343837", "#B04E0F"]
  random.shuffle(colors)

  fig = plt.figure(figsize=(10.5, 6)) 
  gs = gridspec.GridSpec(3,3,width_ratios=[5,5,4]) 


  ax0 = plt.subplot(gs[:,0])
  for j in range(len(real_network_titles)):
    if j == 4:
      ax0.plot(np.linspace(0.0,1.0,num=101),self_recons_score_threshold[j],label=real_network_titles[j],alpha=0.8,color=colors[9])
    else: 
      ax0.plot(np.linspace(0.0,1.0,num=101),self_recons_score_threshold[j],label=real_network_titles[j],alpha=0.8,color=colors[10+j])#,marker=marker_style[j//2],markevery=3)
  ax0.text(0.0,0.95,r"$X\leftarrow X$",fontsize=12)
  ax0.set_ylim(0.0,1.0)
  ax0.set_xlabel(r"$\theta \;\; ($with $r=25)$")
  ax0.set_ylabel("accuracy")
  ax0.legend(loc='lower center',fontsize='medium')


  ax1 = plt.subplot(gs[:,1])
  i = 3 #Caltech
  for j in range(len(synth_network_titles)):  
    ax1.plot(list_of_nc,f_s_scores[:,i,j],label=synth_network_titles[j],alpha=0.8,color=colors[j],linestyle=line_style[j//2])#,marker=marker_style[j//2],markevery=3)
  for j in range(len(facebook_network_titles)): 
    ax1.plot(list_of_nc,f_f_scores[:,i,j],label=facebook_network_titles[j],color=colors[6+j],alpha=0.8)
  ax1.text(9,0.9625,facebook_network_titles[i]+r"$\leftarrow X$",fontsize=12)
  ax1.set_ylim(0.3,1)
  ax1.set_xticks(list_of_nc)
  ax1.set_xlabel(r"$r\;\; ($with $\theta=0.5)$")
  ax1.legend(loc='lower right',fontsize='medium')



  axs = {}
  for i in range(len(facebook_network_titles)-1): #All but Caltech
    axs[i] = plt.subplot(gs[i,2])
    for j in range(len(synth_network_titles)):  
      axs[i].plot(list_of_nc,f_s_scores[:,i,j],label=synth_network_titles[j],alpha=0.8,color=colors[j],linestyle=line_style[j//2])#,marker=marker_style[j//2],markevery=3)
    for j in range(len(facebook_network_titles)): 
      axs[i].plot(list_of_nc,f_f_scores[:,i,j],label=facebook_network_titles[j],color=colors[6+j],alpha=0.8)
    axs[i].text(36,0.65,facebook_network_titles[i]+r"$\leftarrow X$",fontsize=12)
    axs[i].set_ylim(0.6,1)
    axs[i].set_xticks(list_of_nc)
    if i ==2 :
      axs[i].set_xlabel(r"$r\;\; ($with $\theta=0.5)$")


  plt.tight_layout()
  plt.savefig("full_recons_plot_scratch.pdf",bbox_inches='tight')
  plt.clf()










