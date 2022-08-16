import numpy as np
from utils.ndl import Network_Reconstructor
from helper_functions.helper_functions import Generate_corrupt_graph, compute_ROC_AUC_colored, display_denoising_stats_list_plot, compute_ROC_AUC
from NNetwork.NNetwork import NNetwork
import networkx as nx
import csv
import tracemalloc
import itertools
from multiprocessing import Pool
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def run_NDR_denoising(G, test_edges, k=20, r_list=[25], masking_params_list=[0], edges_changed=None, if_keep_visit_statistics = False):
    print("NDR denoising with k={}, r_list={}".format(k, r_list))

    W_list = []
    for r in r_list:
        # Learn dictionary from corrupted (observed) network
        reconstructor_corrupt = Network_Reconstructor(G=G,  # NNetwork simple graph
                                                  n_components=r,  # num of dictionaries
                                                  MCMC_iterations=5,
                                                  # MCMC steps (macro, grow with size of ntwk)
                                                  sample_size=1000,
                                                  # number of patches in a single batch
                                                  batch_size=100,
                                                  # number of columns used to train dictionary
                                                  # within a single batch step (keep it)
                                                  sub_iterations=10,
                                                  # number of iterations of the
                                                  # sub-batch learning (keep it)
                                                  k1=0, k2=k,  # left and right arm lengths
                                                  if_wtd_network=True,
                                                  sampling_alg = 'pivot',
                                                  ONMF_subsample=True,
                                                  Pivot_exact_MH_rule=False)
        ### Set up network dictionary
        reconstructor_corrupt.W = reconstructor_corrupt.train_dict(update_dict_save=True,
                                                                    skip_folded_hom=True)


        ### Denoising
        W_corrupt = reconstructor_corrupt.W
        W_list.append(W_corrupt)

    recons_iter = 5*int(len(G.vertices) * np.log(len(G.vertices)))
    recons_iter = np.minimum(recons_iter, 30000)
    recons_iter = np.minimum(recons_iter, 100000)

    print('!!! masking_params_list',masking_params_list)

    #W_corrupt = reconstructor_corrupt.W
    #W_rand = np.random.rand(W_corrupt.shape[0], W_corrupt.shape[1])
    #W_list = [W_corrupt] * 2 + [W_rand] * 2

    G_recons = reconstructor_corrupt.reconstruct_network_list(W_list = [W_corrupt]*len(masking_params_list),
                                                            #W_list = W_list,
                                                            recons_iter=1000,
                                                            jump_every=100,
                                                            if_save_history=True,
                                                            ckpt_epoch=50000,
                                                            #masking_params_list=[0, 0],
                                                            masking_params_list=masking_params_list,
                                                            #test_edges=test_edges,
                                                            edges_added=edges_changed,
                                                            #save_path = save_folder + "/" + save_filename,
                                                            if_keep_visit_statistics=if_keep_visit_statistics)

    denoising_dict = {}
    if if_keep_visit_statistics:
        denoising_dict = reconstructor_corrupt.result_dict.get("denoising_dict")

    recons_edge_wts = []
    num_test_edges_not_reconstructed = 0
    num_null_recons_edges = 0
    for i in np.arange(len(test_edges)):
        e = test_edges[i]
        #print('e[0]', e[0])
        weight = G_recons.get_colored_edge_weight(e[0], e[1])
        if weight is None:
            weight = [0]*len(masking_params_list)
            num_test_edges_not_reconstructed += 1
        if weight == 0:
            num_null_recons_edges += 1

        wtd_edge = [e[0], e[1], weight]  # weight = list
        recons_edge_wts.append(wtd_edge) #colored_edges

    print('!!! num_test_edges_not_reconstructed = {}'.format(num_test_edges_not_reconstructed))
    print('!!! num_null_recons_edges = {}'.format(num_null_recons_edges))
    #return G_recons.get_wtd_edgelist()
    #print('!!! recons_edge_wts', recons_edge_wts)

    return recons_edge_wts, denoising_dict


def run_NDR_denoising_CV(G, train_edges_false, train_edges_true, test_edges_false, test_edges_true,
                         G_original, path_corrupt, noise_type):

    k_list = [20]
    r_list = [2, 9, 25]
    masking_params_list=[1, 0]


    X = train_edges_true + train_edges_false
    y = [1]*len(train_edges_true) + [0]*len(train_edges_false)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    train_edges_true_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==1]
    train_edges_false_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==0]
    val_edges_true = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==1]
    val_edges_false = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==0]
    val_edges = val_edges_true + val_edges_false
    test_edges = test_edges_false + test_edges_true

    aucs = []
    ks = []
    rs = []
    maskings = []
    dicts = []
    recons_wtd_edgelist_list = []


    for k in k_list:
        for r in r_list:
            for mp in masking_params_list:
                ks.append(k)
                rs.append(r)
                maskings.append(mp)

    for k in k_list:
        recons_wtd_edgelist, denoising_dict = run_NDR_denoising(G, test_edges, k=k, r_list=r_list,
                                                masking_params_list=masking_params_list,
                                                edges_changed = val_edges_false)

        G_recons = NNetwork()
        G_recons.add_colored_edges(recons_wtd_edgelist)
        #print('!!! recons_edge_wts', G_recons.colored_edges)

        edge0 = recons_wtd_edgelist[0]
        n_layers = len(G_recons.get_colored_edge_weight(edge0[0], edge0[1]))
        print('!!! n_layers in colored_reconstruction = {}'.format(n_layers))

        for a in np.arange(n_layers):
            val_edges_recons = [  ]
            for i in np.arange(len(val_edges)):
                e = val_edges[i]
                wt = G_recons.get_colored_edge_weight(e[0], e[1])
                #print('!!! wt', wt)
                #print('@@@ a', a)
                if wt is not None:
                    if wt[a] > 0: # for -ER noise, wt[a] could be none: Fix this
                        val_edges_recons.append([e[0], e[1], wt[a]])

            ROC_dict = compute_ROC_AUC(G_original=G_original,
                                               path_corrupt=path_corrupt,
                                               recons_wtd_edgelist=val_edges_recons,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               subtractive_noise=(noise_type == '-ER_edges'))

            aucs.append(ROC_dict['AUC'])
            recons_wtd_edgelist_list.append(recons_wtd_edgelist)

    print('!!! aucs', aucs)
    j = np.argmax(aucs)
    print("optimal case: k={}, r={}, masking={}".format(ks[j], rs[j], maskings[j]))

    #print('!!! recons_wtd_edgelist_list[j]', recons_wtd_edgelist_list[j])

    G_recons = NNetwork()
    G_recons.add_colored_edges(recons_wtd_edgelist_list[j])
    recons_edge_wts = []
    num_test_edges_not_reconstructed = 0

    #mask_idx = np.where(np.asarray(masking_params_list)==maskings[j])[0][0]
    num_null_recons_edges = 0
    for i in np.arange(len(test_edges)):
        e = test_edges[i]
        weight = G_recons.get_colored_edge_weight(e[0], e[1])
        if weight is None:
            weight = 0
            num_test_edges_not_reconstructed += 1
        else:
            #weight = weight[mask_idx]
            weight = weight[j]

        if weight == 0:
            num_null_recons_edges += 1
        else: # only account properly reconstructed edges
            wtd_edge = [e[0], e[1], weight]
            recons_edge_wts.append(wtd_edge)

    print('!!! num_null_recons_edges = {}'.format(num_null_recons_edges))
    return recons_edge_wts, denoising_dict
