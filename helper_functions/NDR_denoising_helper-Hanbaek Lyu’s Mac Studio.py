import numpy as np
from utils.ndl import Network_Reconstructor, Generate_corrupt_graph, compute_ROC_AUC_colored, display_denoising_stats_list_plot
from utils.NNetwork import NNetwork
import networkx as nx
import csv
import tracemalloc
import itertools
from multiprocessing import Pool
import copy
from pathlib import Path
import matplotlib.pyplot as plt



def run_NDR_denoising(G, test_edges, k=10, r=25):

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

    recons_iter = 5*int(len(G.vertices) * np.log(len(G.vertices)))
    recons_iter = np.max(recons_iter, 100000)

    """
    G_recons = reconstructor_corrupt.reconstruct_network(W = W_corrupt,
                                                        recons_iter=recons_iter,
                                                        jump_every=1000,
                                                        if_save_history=False,
                                                        ckpt_epoch=10000,
                                                        omit_chain_edges=True, # 0 for denoising
                                                        save_path = None,
                                                        if_keep_visit_statistics=False)
    """

    G_recons = reconstructor_corrupt.reconstruct_network_list(W_list = [W_corrupt],
                                                            recons_iter=recons_iter,
                                                            jump_every=1000,
                                                            if_save_history=True,
                                                            ckpt_epoch=50000,
                                                            #masking_params_list=[1, 0, 1, 0],
                                                            masking_params_list=[0],
                                                            #edges_added=edges_changed,
                                                            #save_path = save_folder + "/" + save_filename,
                                                            if_keep_visit_statistics=False)
    """
    recons_edge_wts = []
    recons_edges = G_recons.get_edges()
    for e in recons_edges:
        wt = G_recons.get_colored_edge_weight(e[0], e[1])[0]
        recons_edge_wts.append([e[0], e[1], wt])

    """
    recons_edge_wts = []
    for i in np.arange(len(test_edges)):
        e = test_edges[i]
        weight = G_recons.get_colored_edge_weight(e[0], e[1])
        if weight is None:
            weight = 0
        else:
            weight = weight[0]

        wtd_edge = [e[0], e[1], weight]
        recons_edge_wts.append(wtd_edge)


    #return G_recons.get_wtd_edgelist()
    return recons_edge_wts



def run_NDR_denoising(G, test_edges, k=10, r=25):
    print("NDR denoising with k={}, r={}".format(k, r))
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

    recons_iter = 5*int(len(G.vertices) * np.log(len(G.vertices)))
    recons_iter = np.minimum(recons_iter, 100000)

    """
    G_recons = reconstructor_corrupt.reconstruct_network(W = W_corrupt,
                                                        recons_iter=recons_iter,
                                                        jump_every=1000,
                                                        if_save_history=False,
                                                        ckpt_epoch=10000,
                                                        omit_chain_edges=True, # 0 for denoising
                                                        save_path = None,
                                                        if_keep_visit_statistics=False)
    """

    G_recons = reconstructor_corrupt.reconstruct_network_list(W_list = [W_corrupt],
                                                            recons_iter=100000,
                                                            jump_every=1000,
                                                            if_save_history=True,
                                                            ckpt_epoch=50000,
                                                            #masking_params_list=[1, 0, 1, 0],
                                                            masking_params_list=[0],
                                                            #edges_added=edges_changed,
                                                            #save_path = save_folder + "/" + save_filename,
                                                            if_keep_visit_statistics=False)

    """
    recons_edge_wts = []
    recons_edges = G_recons.get_edges()
    for e in recons_edges:
        wt = G_recons.get_colored_edge_weight(e[0], e[1])[0]
        recons_edge_wts.append([e[0], e[1], wt])
    """


    recons_edge_wts = []
    for i in np.arange(len(test_edges)):
        e = test_edges[i]
        weight = G_recons.get_colored_edge_weight(e[0], e[1])
        if weight is None:
            weight = 0
        else:
            weight = weight[0]

        wtd_edge = [e[0], e[1], weight]
        recons_edge_wts.append(wtd_edge)


    #return G_recons.get_wtd_edgelist()
    return recons_edge_wts
