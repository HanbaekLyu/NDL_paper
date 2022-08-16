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


def read_BIOGRID_network(path, save_file_name):
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
    edgelist = []
    for i in np.arange(1, len(data)):
        if data[i][3] != '-' and data[i][4] != '-':
            edgelist.append([data[i][3], data[i][4]])
            print([data[i][3], data[i][4]])

    G = NNetwork()
    G.add_wtd_edges(edges=edgelist, increment_weights=False)
    print('test edge', edgelist[0])

    G_nx = nx.Graph()
    for i in np.arange(1, len(data)):
        if data[i][3] != '-' and data[i][4] != '-':
            G_nx.add_edge(data[i][3], data[i][4], weight=1)
    nx.write_edgelist(G_nx, "Data/Facebook/" + save_file_name + ".txt", data=False, delimiter=',')
    return G


def load_reconstructed_ntwk(path):
    full_results = np.load(path, allow_pickle=True).item()
    edges = full_results.get('Edges reconstructed')
    G = NNetwork()
    G.add_edges(edges=edges, edge_weight=1, increment_weights=False)
    return G


def run_NDL_NDR(  # ========================== Master parameters
        directory_network_files="",
        save_folder="",
        # -------------------------- loop parameters
        list_network_files=[],
        list_k=[20],  # list of number of nodes in the chain motif -- scale parameter
        list_n_components=[25],  # number of latent motifs to be learned)
        ND_list_noise_type=[],
        ND_list_noise_density=[],
        # -------------------------- chose functions
        if_learn_fresh=False,
        if_save_fig=False,
        if_recons=False,
        learn_from_reconstruction=False,
        if_denoise=False,
        if_corrupt_and_denoise=False,
        generate_corrupted_ntwk=False,
        use_dict_from_corrupted_ntwk=False,
        use_dict_from_ER=False,
        # -------------------------- Global parameters
        sampling_alg = 'pivot', # 'pivot' or 'glabuer' or 'idla' or 'pivot_inj'
        #is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        #is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        Pivot_exact_MH_rule = False, # If true, use exact Metropolis-Hastings rejection rule for Pivot chain
        omit_folded_edges=False,
        omit_chain_edges_denoising=True,
        show_importance=True,
        if_wtd_network=True,
        if_tensor_ntwk=False,
        # ========================= Parameters for Network Dictionary Learning (NDL)
        NDL_MCMC_iterations=100,
        NDL_minibatch_size=1000,
        NDL_onmf_sub_minibatch_size=20,
        NDL_onmf_sub_iterations=100,
        NDL_alpha=1,  # L1 sparsity regularizer for sparse coding
        NDL_onmf_subsample=True,  # subsample from minibatches
        NDL_skip_folded_hom = True, # if true, only use injective homomorphisms during dictionary learning (denoising not affected)
        NDL_jump_every=10,
        # ========================= Parameters for Network Reconstruction (NR)
        NR_recons_iter=50000,
        NR_if_save_history=True,
        NR_ckpt_epoch=10000, # not used if None
        NR_if_save_wtd_reconstruction=False,
        NR_edge_threshold=0.5,
        # ========================= Parameters for Network Reconstruction (ND)
        ND_recons_iter=50000,
        ND_dictionary_for_denoising = None,
        ND_original_network = None, # in NNetwork class
        ND_original_network_path = None, # If no original network is given, assign a path to read it
        ND_noise_sign = "negative", # could also be "positive"
        ND_if_save_history=True,
        ND_ckpt_epoch=10000, # Not used if None
        ND_edge_threshold=0.5,
        ND_if_compute_ROC_AUC = True,
        ND_flip_TP = False,
        ND_use_refreshing_random_dict = False, # If true, resample randomly the dictionary every iteration (for sanity check)
        # ========================= Parameters for Corruption-Denoising experiments
        # ------------------------- ND-NDL from corrupted Network
        CD_NDL_MCMC_iterations=50,
        # ------------------------- ND denoising paramters
        CD_masking_params_list=[1,0],
        CD_recons_iter=50000,
        CD_custom_dict_path = None,
        CD_recons_jump_every=1000,
        CD_if_save_history=False,
        CD_ckpt_epoch=10000, # not used if None
        CD_if_save_wtd_reconstruction=True,
        CD_if_keep_visit_statistics=False,
        CD_edge_threshold=None,
        #---------------------------
        CD_if_compute_ROC_AUC=True
):
    for (k, ntwk, n_components) in itertools.product(list_k, list_network_files, list_n_components):
        print('!!! Network reconstructor initialized with (network, k, r)=', (ntwk, k, n_components))
        path = directory_network_files + ntwk
        network_name = ntwk.replace('.txt', '')
        network_name = network_name.replace('.', '')
        #mcmc = "Glauber"
        #if not is_glauber_dict:
        #     mcmc = "Pivot"

        G = NNetwork()
        G.load_add_wtd_edges(path, increment_weights=False, use_genfromtxt=True)
        print('!!!', G.get_edges()[0])
        print('num edges in G', len(G.edges))
        print('num nodes in G', len(G.nodes()))

        reconstructor = Network_Reconstructor(G=G,  # networkx simple graph
                                              n_components=n_components,  # num of dictionaries
                                              MCMC_iterations=NDL_MCMC_iterations,
                                              # MCMC steps (macro, grow with size of ntwk)
                                              loc_avg_depth=1,  # keep it at 1
                                              sample_size=NDL_minibatch_size,  # number of patches in a single batch
                                              batch_size=NDL_onmf_sub_minibatch_size,
                                              # number of columns used to train dictionary
                                              # within a single batch step (keep it)
                                              sub_iterations=NDL_onmf_sub_iterations,  # number of iterations of the
                                              # sub-batch learning (keep it)
                                              k1=0, k2=k - 1,  # left and right arm lengths
                                              alpha=NDL_alpha,
                                              # parameter for sparse coding, higher for stronger smoothing
                                              if_wtd_network=if_wtd_network,
                                              if_tensor_ntwk=if_tensor_ntwk,
                                              sampling_alg = sampling_alg,
                                              # keep false to use Pivot chain for recons.
                                              ONMF_subsample=NDL_onmf_subsample,
                                              # whether use i.i.d. subsampling for each batch
                                              omit_folded_edges=omit_folded_edges,
                                              Pivot_exact_MH_rule=Pivot_exact_MH_rule)

        reconstructor.result_dict.update({'Network name': network_name})
        reconstructor.result_dict.update({'# of nodes': len(G.vertices)})

        if if_learn_fresh:
            reconstructor.train_dict(jump_every=NDL_jump_every,
                                     skip_folded_hom=NDL_skip_folded_hom)
        elif if_save_fig:
            # network_name = 'UCLA26'
            reconstructor.result_dict = np.load('Network_dictionary/full_result_' + str(network_name) + '.npy',
                                                allow_pickle=True).item()
            reconstructor.W = reconstructor.result_dict.get('Dictionary learned')
            reconstructor.code = reconstructor.result_dict.get('Code COV learned')

        np.save(
            save_folder + "/full_result_" + str(network_name) + "_k_" + str(k) + "_r_" + str(n_components),
            reconstructor.result_dict)

        if if_save_fig:
            ### save dictionaytrain_dict figures

            reconstructor.display_dict(title='Latent motifs learned from ' + str(network_name),
                                       save_path = save_folder + '/Network_dict' + '_' + str(network_name) + '_' + str(
                                           k) + '_' + str(n_components), make_first_atom_2by2=False,
                                       show_importance=show_importance)

        print('Finished dictionary learning from network ' + str(ntwk))

        if if_recons:
            # iter = np.floor(len(G.vertices) * np.log(len(G.vertices)) / 2)

            G_recons = reconstructor.reconstruct_network(recons_iter=NR_recons_iter,
                                                         if_save_history=NR_if_save_history,
                                                         ckpt_epoch=NR_ckpt_epoch,
                                                         omit_chain_edges=False,
                                                         if_save_wtd_reconstruction=NR_if_save_wtd_reconstruction,
                                                         edge_threshold=NR_edge_threshold)

            recons_accuracy = reconstructor.compute_recons_accuracy(G_recons, if_baseline=True)

        if if_denoise:
            if ND_dictionary_for_denoising is not None:
                reconstructor.W = ND_dictionary_for_denoising
            else:
                reconstructor.train_dict(jump_every=NDL_jump_every,
                                         skip_folded_hom=NDL_skip_folded_hom)

            G_recons = reconstructor.reconstruct_network(recons_iter=ND_recons_iter,
                                                         if_save_history=ND_if_save_history,
                                                         ckpt_epoch=ND_ckpt_epoch,
                                                         omit_chain_edges=True,
                                                         if_save_wtd_reconstruction=True,
                                                         edge_threshold=ND_edge_threshold)

            recons_wtd_edgelist = reconstructor.result_dict.get('Edges in weighted reconstruction')
            recons_accuracy = reconstructor.compute_recons_accuracy(G_recons, if_baseline=True)

            if ND_if_compute_ROC_AUC:
                ROC_file_name = network_name + "_" + str(NR_recons_iter)
                if ND_original_network is not None:
                    ROC_Dict = compute_ROC_AUC(G_original=ND_original_network,
                                               G_corrupted=G,
                                               path_original=None,
                                               path_corrupt=None,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               save_file_name=ROC_file_name,
                                               save_folder=save_folder,
                                               flip_TF=ND_flip_TP,
                                               subtractive_noise=(ND_noise_sign == 'negative'))
                else:
                    ROC_dict = compute_ROC_AUC(G_original=None,
                                               G_corrupted=G,
                                               path_original=ND_original_network_path,
                                               path_corrupt=None,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               save_file_name=ROC_file_name,
                                               save_folder=save_folder,
                                               flip_TF=ND_flip_TP,
                                               subtractive_noise=(ND_noise_sign == 'negative'))

                reconstructor_corrupt.result_dict.update({'False positive rate': ROC_dict.get('False positive rate')})
                reconstructor_corrupt.result_dict.update({'True positive rate': ROC_dict.get('True positive rate')})
                reconstructor_corrupt.result_dict.update({'AUC': ROC_dict.get('AUC')})

                ROC_dict.update({'Dictionary learned': reconstructor_corrupt.W})
                ROC_dict.update({'Motif size': reconstructor_corrupt.k2 + 1})
                # ROC_dict.update({'Code learned': reconstructor_corrupt.code})

                if not use_dict_from_ER:
                    save_file_name = save_folder + "/ROC_dict_" + str(
                        network_name) + "_" + "Use_corrupt_dict_" + str(
                        use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                        len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                        iter) + "_Glauber_recons_" + str(is_glauber_recons)
                else:
                    save_file_name = save_folder + "/ROC_dict_" + str(
                        network_name) + "_" + "Use_ER_dict_" + str(
                        use_dict_from_ER) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                        len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                        iter) + "_Glauber_recons_" + str(is_glauber_recons)

                np.save(save_file_name, ROC_dict)

        if learn_from_reconstruction:
            path = save_folder + '/full_result_' + str(network_name) + "_k_" + str(k) + "_r_" + str(
                n_components) + '.npy'
            G = load_reconstructed_ntwk(path)

            reconstructor = Network_Reconstructor(G=G,  # networkx simple graph
                                                  n_components=n_components,  # num of dictionaries
                                                  MCMC_iterations=NDL_MCMC_iterations,
                                                  # MCMC steps (macro, grow with size of ntwk)
                                                  loc_avg_depth=1,  # keep it at 1
                                                  sample_size=NDL_minibatch_size,  # number of patches in a single batch
                                                  batch_size=NDL_onmf_sub_minibatch_size,
                                                  # number of columns used to train dictionary
                                                  # within a single batch step (keep it)
                                                  sub_iterations=NDL_onmf_sub_iterations,  # number of iterations of the
                                                  # sub-batch learning (keep it)
                                                  k1=0, k2=k - 1,  # left and right arm lengths
                                                  alpha=NDL_alpha,
                                                  # parameter for sparse coding, higher for stronger smoothing
                                                  if_wtd_network=if_wtd_network,
                                                  if_tensor_ntwk=if_tensor_ntwk,
                                                  sampling_alg = sampling_alg,
                                                  # keep false to use Pivot chain for recons.
                                                  ONMF_subsample=NDL_onmf_subsample,
                                                  # whether use i.i.d. subsampling for each batch
                                                  omit_folded_edges=omit_folded_edges,
                                                  Pivot_exact_MH_rule=Pivot_exact_MH_rule)

            reconstructor.train_dict(skip_folded_hom=NDL_skip_folded_hom)
            ### save dictionaytrain_dict figures
            # reconstructor.display_dict(title, save_filename)
            reconstructor.display_dict(title='Dictionary learned from Recons. Network ' + str(network_name),
                                       save_path = save_folder + '/Network_dict_recons_Caltech_from_UCLA_chain_included' + '_' + str(
                                           network_name) + str(k),
                                       show_importance=show_importance)
            np.save(save_folder + "/full_result_" + str(network_name) + "_k_" + str(k) + "_r_" + str(n_components),
                    reconstructor.result_dict)

        if if_corrupt_and_denoise:
            print('!!!! corrupt and denoise')
            # print('!!! sampling alg', sampling_alg)
            for (noise_type, rate) in itertools.product(ND_list_noise_type, ND_list_noise_density):

                n_edges = len(nx.Graph(G.edges).edges)
                print('!!! n_edges', n_edges)
                print('!!! noise rate:', rate)

                path_original = path
                p = np.floor(n_edges * rate)
                if (ntwk == 'COVID_PPI.txt') and (p == np.floor(n_edges * 0.5)) and (noise_type == "-ER_edges"):
                    p = np.floor(n_edges * 0.2)

                if generate_corrupted_ntwk:

                    #noise_nodes = len(G.vertices)
                    # noise_nodes = 200
                    # noise_type = 'WS'
                    #parameter = p.astype(int)


                    noise_sign = "added"
                    if noise_type == '-ER_edges':
                        noise_sign = "deleted"


                    rate = 0.5
                    if noise_type in ['ER', '-ER']:
                        parameter = 0.5
                        if (ntwk == 'COVID_PPI.txt' or ntwk == 'COVID_PPI_new.txt') and (noise_type in ["-ER_edges", "-ER"]):
                            parameter = 0.2

                        noise_nodes=len(G.vertices)
                        noise_type+"_edges"

                    elif noise_type == 'WS':
                        # WS network with 100 nodes, 500 edges, rewiring prob. 0.3
                        #noise_nodes=int(len(G.vertices)*0.1)
                        noise_nodes = 100
                        parameter= 1000
                        if ntwk == "node2vec_homosapiens_PPI.txt":
                            noise_nodes = 500
                            parameter= 30000

                        noise_type='WS'

                    path_save = save_folder + "/" + network_name + "_noise_nodes_" + str(
                        noise_nodes) + "_noisetype_" + noise_type

                    G_corrupt, edges_changed = Generate_corrupt_graph(path_load=path_original,
                                                                      delimiter=' ',
                                                                      G_original=G,
                                                                      path_save=path_save,
                                                                      noise_nodes=noise_nodes,
                                                                      parameter=parameter,
                                                                      noise_type=noise_type)

                    path_corrupt = path_save + "_n_edges_" + noise_sign + "_" + str(len(edges_changed)) + ".txt"

                    print('Corrupted network generated with %i edges ' % len(edges_changed) + noise_sign)
                    print('path_corrupt', path_corrupt)
                else:
                    G_corrupt = Wtd_NNetwork()
                    G_corrupt.load_add_wtd_edges(path_corrupt, increment_weights=False, delimiter=',',
                                                 use_genfromtxt=True)
                    # G = read_BIOGRID_network(path)
                    print('num edges in G_corrupt', len(G_corrupt.get_edges()))

                ### Set up reconstructor for G_corrupt
                print('!!!! CD_NDL_MCMC_iterations', CD_NDL_MCMC_iterations)
                reconstructor_corrupt = Network_Reconstructor(G=G_corrupt,  # NNetwork simple graph
                                                              n_components=n_components,  # num of dictionaries
                                                              MCMC_iterations=CD_NDL_MCMC_iterations,
                                                              # MCMC steps (macro, grow with size of ntwk)
                                                              loc_avg_depth=1,  # keep it at 1
                                                              sample_size=NDL_minibatch_size,
                                                              # number of patches in a single batch
                                                              batch_size=NDL_onmf_sub_minibatch_size,
                                                              # number of columns used to train dictionary
                                                              # within a single batch step (keep it)
                                                              sub_iterations=NDL_onmf_sub_iterations,
                                                              # number of iterations of the
                                                              # sub-batch learning (keep it)
                                                              k1=0, k2=k - 1,  # left and right arm lengths
                                                              alpha=0,
                                                              # regularizer for sparse coding, higher for stronger smoothing
                                                              if_wtd_network=if_wtd_network,
                                                              if_tensor_ntwk=if_tensor_ntwk,
                                                              sampling_alg = sampling_alg,
                                                              omit_folded_edges=omit_folded_edges,
                                                              # keep false to use Pivot chain for recons.
                                                              ONMF_subsample=NDL_onmf_subsample,
                                                              #save_path = save_folder + "/" + save_filename,
                                                              # whether use i.i.d. subsampling for each batch
                                                              Pivot_exact_MH_rule=Pivot_exact_MH_rule)
                ### Set up network dictionary
                if CD_custom_dict_path is not None:
                    # Transfer-denoising
                    result_dict = np.load(CD_custom_dict_path, allow_pickle=True).item()
                    reconstructor_corrupt.W = result_dict.get('Dictionary learned')
                    reconstructor_corrupt.At = result_dict.get('Code COV learned')
                    print('Dictionary loaded from:', str(CD_custom_dict_path))


                elif use_dict_from_corrupted_ntwk:
                    reconstructor_corrupt.W = reconstructor_corrupt.train_dict(update_dict_save=True,
                                                                               skip_folded_hom=NDL_skip_folded_hom)
                    # print('corrupted dictionary', reconstructor.W)
                    print('Dictionary learned from the corrupted network')

                else:
                    ### Use below for self-denoising from corrupt dictionary
                    reconstructor_corrupt.W = reconstructor.W

                save_filename = 'Network_dict' + '_' + str(network_name) + '_' + str(
                    k) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + '_' + str(n_components) + "_noisetype_" + noise_type

                ### Denoising
                # iter = np.floor(len(G.vertices) * np.log(len(G.vertices)) *4 )
                iter = CD_recons_iter
                W = reconstructor_corrupt.W.copy()
                #reconstructor_corrupt.train_dict(update_dict_save=True,
                #                                          iterations=0).copy()
                #W_rand = reconstructor_corrupt.W.copy()
                W_rand = np.random.rand(W.shape[0], W.shape[1])
                #W_list = [W, W, W_rand, W_rand]
                W_list = [W, W, W, W_rand, W_rand]
                #W_list = [W, W, W]
                #W_list = [reconstructor_corrupt.W] * len(CD_masking_params_list)
                G_recons_simple = reconstructor_corrupt.reconstruct_network_list(recons_iter=iter,
                                                                            W_list = W_list,
                                                                            jump_every=CD_recons_jump_every,
                                                                            ckpt_epoch=CD_ckpt_epoch,
                                                                            ### Keep true for denoising
                                                                            #masking_params_list=CD_masking_params_list,
                                                                            masking_params_list=[1,0.5,0,1,0],
                                                                            ### Keep true for denoising
                                                                            edges_added=edges_changed,
                                                                            save_path = save_folder + "/" + save_filename,
                                                                            if_keep_visit_statistics=CD_if_keep_visit_statistics)


                print('parameter', parameter)

                ### plot dictionary used for denoising
                title = 'Dictionary learned:' + str(network_name) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type

                """
                reconstructor_corrupt.display_dict(title='Dictionary used for denoising ' + str(network_name),
                                                   save_path = save_folder + '/Dict_used_for_denoising' + '_' + str(
                                                       network_name) + '_' + str(
                                                       k) + '_' + str(n_components),
                                                   make_first_atom_2by2=False,
                                                   show_importance=show_importance)
                """

                np.save(save_folder + "/full_result_" + str(network_name) + "_" + "Use_corrupt_dict_" + str(
                    use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type,
                        reconstructor_corrupt.result_dict)


                ### Start computing denoising accuracies
                save_file_name = save_folder + "/full_result_" + str(
                    network_name) + "_" + "Use_corrupt_dict_" + str(
                    use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                    iter) + "sampling_alg_" + str(sampling_alg)


                # np.save(save_file_name, reconstructor_corrupt.result_dict)

                recons_colored_edges = reconstructor_corrupt.result_dict.get('Colored edges in reconstruction')
                print('!!! All done with reconstruction -- # of reconstucted edges:', len(recons_colored_edges))
                recons_accuracy = reconstructor_corrupt.compute_recons_accuracy(G_recons_simple,
                                                                                if_baseline=True)

                if CD_if_compute_ROC_AUC:
                    ROC_file_name = network_name + "\n" + "ER_" + str(noise_nodes) + "_p_" + str(parameter).replace(
                        '.',
                        '') + "_n_edges_" + noise_sign + "_" + str(
                        len(edges_changed)) + "_iter_" + str(iter)
                    ROC_dict = compute_ROC_AUC_colored(G_original=G,
                                               path_corrupt=path_corrupt,
                                               recons_colored_edges=recons_colored_edges,
                                               #is_dict_edges=True,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               save_file_name=ROC_file_name,
                                               save_folder=save_folder,
                                               # flip_TF=not omit_chain_edges_denoising, (10/12/2021 for direct comparison of various methods)
                                               flip_TF=False,
                                               subtractive_noise=(noise_type == '-ER_edges'))

                    """
                    reconstructor_corrupt.result_dict.update(
                        {'False positive rate': ROC_dict.get('False positive rate')})
                    reconstructor_corrupt.result_dict.update({'True positive rate': ROC_dict.get('True positive rate')})
                    reconstructor_corrupt.result_dict.update({'AUC': ROC_dict.get('AUC')})
                    """

                    ROC_dict[0].update({'# edges of original ntwk': len(G.get_edges())})
                    ROC_dict[0].update({'Dictionary learned': reconstructor_corrupt.W})
                    ROC_dict[0].update({'Motif size': reconstructor_corrupt.k2 + 1})
                    ROC_dict[0].update({'Code learned': reconstructor_corrupt.code})
                    ROC_dict[0].update({'Network name': str(network_name)})
                    ROC_dict[0].update({'Use_corrupt_dict': str(use_dict_from_corrupted_ntwk)})
                    ROC_dict[0].update({'noise_nodes': str(noise_nodes)})
                    ROC_dict[0].update({'noise_type': noise_type})
                    ROC_dict[0].update({'n_corrupt_edges': edges_changed})
                    ROC_dict[0].update({'denoising_iter': str(iter)})
                    ROC_dict[0].update({'omit_chain_edges_denoising': str(omit_chain_edges_denoising)})
                    ROC_dict[0].update({'omit_folded_edges': str(omit_folded_edges)})


                    save_folder_sub = save_folder + "/" + str(network_name) + "/" + str(noise_type) + "/" + str(rate).replace('.', '')
                    Path(save_folder_sub).mkdir(parents=True, exist_ok=True)


                    if not use_dict_from_ER:
                        save_file_name = save_folder_sub + "/ROC_dict_" + str(
                            network_name) + "_" + "Use_corrupt_dict_" + str(
                            use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                            len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                            iter) + "sampling_alg_" + str(sampling_alg)
                    else:
                        save_file_name = save_folder_sub + "/ROC_dict_" + str(
                            network_name) + "_" + "Use_ER_dict_" + str(
                            use_dict_from_ER) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                            len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                            iter) + "sampling_alg_" + str(sampling_alg)

                    np.save(save_file_name, ROC_dict)

                    if CD_if_keep_visit_statistics:
                        denoising_dict = reconstructor_corrupt.result_dict.get("denoising_dict")
                        W_list_filtered = reconstructor_corrupt.result_dict.get("W_list_filtered")

                        save_path = save_folder + "/denoising_plot_" + str(
                            network_name) + "_" + "Use_corrupt_dict_" + str(
                            use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                            len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                            iter) + "sampling_alg_" + str(sampling_alg) + "mcmc_iter_" + str(CD_NDL_MCMC_iterations)

                        title = str(network_name) +  "_n_corrupt_edges_" + str(
                                        len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                                        iter)

                        denoising_dict = reconstructor_corrupt.result_dict.get("denoising_dict")
                        W_list_filtered = reconstructor_corrupt.result_dict.get("W_list_filtered")
                        display_denoising_stats_list_plot(denoising_dict = denoising_dict,
                                                     W_list = W_list,
                                                     At = reconstructor_corrupt.At,
                                                     W_list_filtered = W_list_filtered,
                                                     ROC_dict_list = ROC_dict,
                                                     save_path = save_path,
                                                     title = title,
                                                     fig_size = [15,8])


def Generate_all_dictionary():
    ### Generating all dictionaries
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"

    list_network_files = ['Caltech36.txt',
                          'UCLA26.txt',
                          'MIT8.txt',
                          'Harvard1.txt',
                          'COVID_PPI.txt',
                          'facebook_combined.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt',
                          'true_edgelist_for_SW_5000_k_50_p_0.05.txt',
                          'true_edgelist_for_SW_5000_k_50_p_0.1.txt',
                          'true_edgelist_for_ER_5000_mean_degree_100.txt',
                          'true_edgelist_for_ER_5000_mean_degree_50.txt',
                          'true_edgelist_for_BA_5000_m_50.txt',
                          'true_edgelist_for_BA_5000_m_25.txt',
                          'SBM1.txt',
                          'SBM2.txt']

    list_network_files = ['COVID_PPI.txt']
    #list_network_files = ['UCLA26.txt']

    list_k = [11]  # list of number of nodes in the chain motif -- scale parameter
    list_n_components = [9, 16, 36, 49, 64, 81, 100]  # number of latent motifs to be learned)

    #list_k = [51]  # list of number of nodes in the chain motif -- scale parameter
    #list_n_components = [25]  # number of latent motifs to be learned)


    run_NDL_NDR(  # ========================== Master parameters
        directory_network_files=directory_network_files,
        save_folder=save_folder,
        # -------------------------- loop parameters
        list_network_files=list_network_files,
        list_k=list_k,  # list of number of nodes in the chain motif -- scale parameter
        list_n_components=list_n_components,  # number of latent motifs to be learned)
        # -------------------------- chose functions
        if_learn_fresh=True,
        if_save_fig=True,
        if_recons=False,
        learn_from_reconstruction=False,
        if_corrupt_and_denoise=False,
        generate_corrupted_ntwk=False,
        use_dict_from_corrupted_ntwk=False,
        # --------------------------- Global parameters
        #is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        #is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        omit_folded_edges=True,
        omit_chain_edges_denoising=True,
        show_importance=True,
        # --------------------------- Iteration parameters
        NDL_MCMC_iterations=20,
        # ---------------------------- Reconstruction
        NR_recons_iter=5000,
        NR_if_save_history=True,
        NR_ckpt_epoch=10000,  # Not used if None
        NR_if_save_wtd_reconstruction=False,
        NR_edge_threshold=0.5,
    )

def Generate_corrupt_and_denoising_results():
    print('!! Generate & denoise experiment started..')
    ### Generating all dictionaries
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/denoising_3"

    list_network_files = ['COVID_PPI.txt',
                          'Caltech36.txt',
                          'facebook_combined.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt']

    list_network_files = ['COVID_PPI.txt']

    list_k = [21]  # list of number of nodes in the chain motif -- scale parameter
    list_n_components = [2]  # number of latent motifs to be learned)
    # ND_list_noise_type = ["-ER_edges", "ER_edges"]
    ND_list_noise_type = ["WS"]
    ND_list_noise_density = [0.5, 0.1]
    # ND_list_noise_density = [0.5]

    run_NDL_NDR(  # ========================== Master parameters
        directory_network_files=directory_network_files,
        save_folder=save_folder,
        # -------------------------- loop parameters
        list_network_files=list_network_files,
        list_k=list_k,  # list of number of nodes in the chain motif -- scale parameter
        list_n_components=list_n_components,  # number of latent motifs to be learned)
        ND_list_noise_type=ND_list_noise_type,
        ND_list_noise_density=ND_list_noise_density,
        # -------------------------- chose functions
        sampling_alg = 'pivot', # 'pivot' or 'glabuer' or 'idla' or 'pivot_inj'
        if_learn_fresh=False,
        if_save_fig=False,
        if_recons=False,
        learn_from_reconstruction=False,
        if_corrupt_and_denoise=True,
        generate_corrupted_ntwk=True,
        use_dict_from_corrupted_ntwk=True,
        # --------------------------- Global parameters
        #is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        #is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        omit_folded_edges=False,
        omit_chain_edges_denoising=True,
        show_importance=True,
        # --------------------------- Iteration parameters
        NDL_skip_folded_hom = True, # if true, only use injective homomorphisms during dictionary learning (denoising not affected)
        CD_NDL_MCMC_iterations=10,
        # CD_custom_dict_path = CD_custom_dict_path, # comment out if custom dictionary is not used
        CD_ckpt_epoch=5000,
        CD_recons_iter=50000,
        CD_if_compute_ROC_AUC=True,
        CD_if_keep_visit_statistics=False
    )

def recons_network(arglist):
  G1, name1, dictionary_name, nc, k2, full_output  = arglist


  reconstructor = Network_Reconstructor(G=G1,  #  simple graph
                                  n_components=nc,  # num of dictionaries
                                  MCMC_iterations=200,  # MCMC steps (macro, grow with size of ntwk)
                                  loc_avg_depth=1,  # keep it at 1
                                  sample_size=1000,  # number of patches in a single batch
                                  batch_size=20,  # number of columns used to train dictionary
                                  # within a single batch step (keep it)
                                  sub_iterations=100,  # number of iterations of the
                                  # sub-batch learning (keep it)
                                  k1=0, k2=k2,  # left and right arm lengths
                                  alpha=1,  # parameter for sparse coding, higher for stronger smoothing
                                  if_wtd_network=False,
                                  if_tensor_ntwk=False,
                                  #is_glauber_recons=False,  # keep false to use Pivot chain for recons.
                                  ONMF_subsample=True,  # whether use i.i.d. subsampling for each batch
                                  omit_folded_edges=True)

  number_of_nodes = k2 + 1
  save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
  reconstructor.result_dict = np.load(f'{save_folder}/full_result_{dictionary_name}_k_{number_of_nodes}_r_{nc}.npy', allow_pickle=True).item()
  reconstructor.W = reconstructor.result_dict.get('Dictionary learned')
  reconstructor.code = reconstructor.result_dict.get('Code learned')


  recons_iter = int(len(G1.vertices) * np.log(len(G1.vertices)))
  reconstructor.result_dict.update({'reconstruction iterations': iter})

  if full_output:
    G_recons_wtd = reconstructor.reconstruct_network(recons_iter=recons_iter,
                                             #if_construct_WtdNtwk=True,
                                             if_save_history=True,
                                             #use_checkpoint_refreshing=False,
                                             ckpt_epoch=10000,
                                             omit_chain_edges=False)
    G_recons_wtd.save_wtd_edgelist(save_folder, f"{name1}_recons_for_nc_{nc}_from_{dictionary_name}.txt")
  else:
    G_recons_simple = reconstructor.reconstruct_network(recons_iter=recons_iter,
                                             #if_construct_WtdNtwk=True,
                                             if_save_history=True,
                                             #use_checkpoint_refreshing=False,
                                             ckpt_epoch=10000,
                                             omit_chain_edges=False)

    G_recons_simple = G_recons_simple.threshold2simple(0.5)
    recons_metrics = reconstructor.compute_jaccard_recons_accuracy(G_recons_simple, output_full=True)
    np.save(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.npy", recons_metrics)

    #file = open(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.txt", "w+")
    #file.write(str(recons_accuracy))
    #file.close()



def accuracyScoreByTheta(G_original,G_recons,name):
    accuracies = []

    edges_recons = G_recons.get_edges()

    common_edges_recons_binned, uncommon_edges_recons_binned = [0]*101 , [0]*101
    for edge in edges_original:
        wt = G_recons.get_edge_weight(edge[0], edge[1])
        index = round(100*wt)
        if (wt is not None) and G_original.has_edge(edge[0], edge[1]):
            if index<0:
              common_edges_recons_binned[0] +=1
            elif index>=101:
              common_edges_recons_binned[100] +=1
            else:
              common_edges_recons_binned[index] +=1
        else:
            if index<0:
              uncommon_edges_recons_binned[0] +=1
            elif index>=101:
              uncommon_edges_recons_binned[100] +=1
            else:
              uncommon_edges_recons_binned[index] +=1

    common_edges_recons_binned_cum_sum = list(itertools.accumulate(reversed(common_edges_recons_binned)))
    uncommon_edges_recons_binned_cum_sum = list(itertools.accumulate(reversed(uncommon_edges_recons_binned)))
    G_original_num_edges = len(G_original.get_edges())


    for i in range(101):
      recons_accuracy = common_edges_recons_binned_cum_sum[i] / ( G_original_num_edges + uncommon_edges_recons_binned_cum_sum[i])
      accuracies.append(recons_accuracy)

    save_folder = "Network_dictionary/Figure3_Data"
    file = open(f"{save_folder}/self_recons_{name}_vary_threshold.txt", "w+")
    for recons_accuracy in reversed(accuracies):
      file.write(str(recons_accuracy)+'\n')
    file.close()


def accuracyScoreByTheta_old(G_original,G_recons,name):
    accuracies = []
    common_edges, uncommon_edges = G_recons.inter_and_outer_section(G_original)
    common_edges_binned, uncommon_edges_binned = [0]*101 , [0]*101

    for edge in common_edges:
      index = round(100*edge)
      if index<0:
        common_edges_binned[0] +=1
      elif index>=101:
        common_edges_binned[100] +=1
      else:
        common_edges_binned[index] +=1

    for edge in uncommon_edges:
      index = round(100*edge)
      if index<0:
        uncommon_edges_binned[0] +=1
      elif index>=101:
        uncommon_edges_binned[100] +=1
      else:
        uncommon_edges_binned[index] +=1

    common_edges_binned_cum_sum = list(itertools.accumulate(reversed(common_edges_binned)))
    uncommon_edges_binned_cum_sum = list(itertools.accumulate(reversed(uncommon_edges_binned)))
    G_original_num_edges = len(G_original.get_edges())


    for i in range(101):
      recons_accuracy = common_edges_binned_cum_sum[i] / ( G_original_num_edges + uncommon_edges_binned_cum_sum[i])
      accuracies.append(recons_accuracy)

    save_folder = "Network_dictionary/test"
    file = open(f"{save_folder}/self_recons_{name}_vary_threshold.txt", "w+")
    for recons_accuracy in reversed(accuracies):
      file.write(str(recons_accuracy)+'\n')
    file.close()


def recons_facebook_accuracy_parallel(network_to_recons_name,network_names_dict,pool_size, nc=None):
    directory_network_files = "Data/Networks_all_NDL/"
    if nc==None:
        nc_range = [i**2 for i in range(3,11)]
    else:
        nc_range = [nc]
        k2_range = [20]
        inputs = []
        edgelist = np.genfromtxt(f"{directory_network_files}{network_to_recons_name}.txt", delimiter=',', dtype=str)
        edgelist = edgelist.tolist()
        G = NNetwork()
        G.add_edges(edges=edgelist)

    for dictionary_name in network_names_dict:
        inputs += [(copy.copy(G),network_to_recons_name,dictionary_name,nc,k2, False) for nc in nc_range for k2 in k2_range]

    #if __name__ == '__main__':
    with Pool(pool_size) as p:
        p.map(recons_network,inputs)






def compute_all_recons_scores():
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/recons_plot_data"


    #Panel A: Self-Recons of Caltech, arXiv, Coranvirus PPI, Facebook, and Homo Sapiens PPI
    k2_range = [20]
    nc_range = [25]
    inputs = []
    network_names_recons = ['Caltech36']
    for network in network_names_recons:
        edgelist = np.genfromtxt(f"{directory_network_files}{network}.txt", delimiter=',', dtype=str)
        edgelist = edgelist.tolist()

        G = NNetwork()
        G.add_edges(edges=edgelist)

        inputs += [(copy.copy(G),network,network,nc,k2, True) for nc in nc_range for k2 in k2_range]

    network_names_recons = ["arxiv","COVID_PPI", "facebook_combined","node2vec_homosapiens_PPI"]

    for network in network_names_recons:
        edgelist = np.genfromtxt(f"{directory_network_files}{network}.txt", delimiter=',', dtype=str)
        edgelist = edgelist.tolist()

        G = NNetwork()
        G.add_edges(edges=edgelist)

        inputs += [(copy.copy(G),network,network,nc,k2,True) for nc in nc_range for k2 in k2_range]

    if __name__ == '__main__':
        print('!!! AAA')
        with Pool(5) as p:
            p.map(recons_network,inputs)




      #Panel A: Threshold Values from WTD Network
    """
    network_names_recons = ['Caltech36']
        for network in network_names_recons:
            edgelist = np.genfromtxt(f"{directory_network_files}/{network}.txt", delimiter=',', dtype=str)
            edgelist = edgelist.tolist()
            G_original = NNetwork()
            G_original.add_edges(edges=edgelist)
            recons_directory = save_folder
            network_file = f"{network}_recons_for_nc_25_from_{network}.txt.txt"
            G_recons = NNetwork()
            G_recons.load_add_wtd_edges(recons_directory+network_file)
            accuracyScoreByTheta(G_original,G_recons,network)
    network_names_recons = ["arxiv","COVID_PPI", "facebook_combined","node2vec_homosapiens_PPI"]
        for network in network_names_recons:
            edgelist = np.genfromtxt(f"{directory_network_files}/{network}.txt", delimiter=',', dtype=str)
            edgelist = edgelist.tolist()
            G_original = NNetwork()
            G_original.add_edges(edges=edgelist)
            recons_directory = save_folder
            network_file = f"{network}_recons_for_nc_25_from_{network}.txt.txt"
            G_recons = NNetwork()
            G_recons.load_add_wtd_edges(recons_directory+network_file)
            accuracyScoreByTheta(G_original,G_recons,network)


    # Panels B-E Cross-Recons Scores
    nodes = 5000
    p_values_ER = [50/(nodes-1), 100/(nodes-1)]
    ER = [ f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p*(nodes-1))}" for p in p_values_ER]
    p_values_SW = [0.05, 0.1]
    k_values_SW = [50]
    SW = [ f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p,2)).replace('.','')}" for k in k_values_SW for p in p_values_SW]
    m_values_BA = [25,50]
    BA = [ f"true_edgelist_for_BA_{nodes}_m_{m}"for m in m_values_BA]
    synth_network_file_names =   ER+SW+BA

    pool_size = [32, 32, 4, 8] #corresponds to network order below C M U H
    facebook_networks_file_names = ["Caltech36", "MIT8", "UCLA26", "Harvard1" ]
    network_names_dict  = synth_network_file_names +  facebook_networks_file_names

    for i in range(len(pool_size)):
        recons_facebook_accuracy_parallel(facebook_networks_file_names[i],network_names_dict,pool_size[i])
    """
