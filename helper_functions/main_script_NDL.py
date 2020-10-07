import numpy as np
from utils.ndl import Network_Reconstructor, Generate_corrupt_graph, compute_ROC_AUC
from utils.NNetwork import NNetwork, Wtd_NNetwork
import networkx as nx
import csv
import tracemalloc
import itertools


def read_BIOGRID_network(path, save_file_name):
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
    edgelist = []
    for i in np.arange(1, len(data)):
        if data[i][3] != '-' and data[i][4] != '-':
            edgelist.append([data[i][3], data[i][4]])
            print([data[i][3], data[i][4]])

    G = Wtd_NNetwork()
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
    G = Wtd_NNetwork()
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
        # -------------------------- Global parameters
        is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        Pivot_exact_MH_rule=False,  # If true, use exact Metropolis-Hastings rejection rule for Pivot chain
        omit_folded_edges=False,
        omit_chain_edges_denoising=True,
        show_importance=True,
        if_wtd_network=True,
        if_tensor_ntwk=False,
        # ========================= Parameters for Network Dictionary Learning (NDL)
        NDL_MCMC_iterations=100,
        NDL_minibatch_size=100,
        NDL_onmf_sub_minibatch_size=20,
        NDL_onmf_sub_iterations=100,
        NDL_alpha=1,  # L1 sparsity regularizer for sparse coding
        NDL_onmf_subsample=True,  # subsample from minibatches
        NDL_jump_every=10,
        # ========================= Parameters for Network Reconstruction (NR)
        NR_recons_iter=50000,
        NR_if_save_history=True,
        NR_ckpt_epoch=10000,  # not used if None
        NR_if_save_wtd_reconstruction=False,
        NR_edge_threshold=0.5,
        # ========================= Parameters for Corruption-Denoising experiments
        # ------------------------- ND-NDL from corrupted Network
        CD_NDL_MCMC_iterations=50,
        CD_NDL_corrupt_ntwk_path=None,
        # ------------------------- ND denoising paramters
        CD_recons_iter=50000,
        CD_custom_dict_path=None,
        CD_recons_jump_every=1000,
        CD_if_save_history=False,
        CD_ckpt_epoch=10000,  # not used if None
        CD_if_save_wtd_reconstruction=True,
        CD_if_keep_visit_statistics=False,
        CD_edge_threshold=0.3,
        # ---------------------------
        CD_if_compute_ROC_AUC=True
):
    for (k, ntwk, n_components) in itertools.product(list_k, list_network_files, list_n_components):
        print('!!! Network reconstructor initialized with (network, k, r)=', (ntwk, k, n_components))
        path = directory_network_files + ntwk
        network_name = ntwk.replace('.txt', '')
        network_name = network_name.replace('.', '')
        mcmc = "Glauber"
        if not is_glauber_dict:
            mcmc = "Pivot"

        G = Wtd_NNetwork()
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
                                              is_glauber_dict=is_glauber_dict,
                                              # keep true to use Glauber chain for dict. learning
                                              is_glauber_recons=is_glauber_recons,
                                              # keep false to use Pivot chain for recons.
                                              ONMF_subsample=NDL_onmf_subsample,
                                              # whether use i.i.d. subsampling for each batch
                                              omit_folded_edges=omit_folded_edges,
                                              Pivot_exact_MH_rule=Pivot_exact_MH_rule)

        reconstructor.result_dict.update({'Network name': network_name})
        reconstructor.result_dict.update({'# of nodes': len(G.vertices)})

        if if_learn_fresh:
            reconstructor.train_dict(jump_every=NDL_jump_every)
        elif if_save_fig:
            # network_name = 'UCLA26'
            reconstructor.result_dict = np.load('Network_dictionary/full_result_' + str(network_name) + '.npy',
                                                allow_pickle=True).item()
            reconstructor.W = reconstructor.result_dict.get('Dictionary learned')
            reconstructor.code = reconstructor.result_dict.get('Code COV learned')

        np.save(
            save_folder + "/full_result_" + str(network_name) + "_k_" + str(k) + "_r_" + str(n_components) + "_" + mcmc,
            reconstructor.result_dict)

        if if_save_fig:
            ### save dictionaytrain_dict figures

            reconstructor.display_dict(title='Latent motifs learned from ' + str(network_name),
                                       save_folder=save_folder,
                                       save_filename='Network_dict' + '_' + str(network_name) + '_' + str(
                                           k) + '_' + str(n_components) + "_" + mcmc + "_omit_folded_edges_" + str(
                                           omit_folded_edges),
                                       make_first_atom_2by2=False,
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
                                                  is_glauber_dict=is_glauber_dict,
                                                  # keep true to use Glauber chain for dict. learning
                                                  is_glauber_recons=is_glauber_recons,
                                                  # keep false to use Pivot chain for recons.
                                                  ONMF_subsample=NDL_onmf_subsample,
                                                  # whether use i.i.d. subsampling for each batch
                                                  omit_folded_edges=omit_folded_edges,
                                                  Pivot_exact_MH_rule=Pivot_exact_MH_rule)

            reconstructor.train_dict()
            ### save dictionaytrain_dict figures
            # reconstructor.display_dict(title, save_filename)
            reconstructor.display_dict(title='Dictionary learned from Recons. Network ' + str(network_name),
                                       save_folder=save_folder,
                                       save_filename='Network_dict_recons_Caltech_from_UCLA_chain_included' + '_' + str(
                                           network_name) + str(k),
                                       show_importance=show_importance)
            np.save(save_folder + "/full_result_" + str(network_name) + "_k_" + str(k) + "_r_" + str(n_components),
                    reconstructor.result_dict)

        if if_corrupt_and_denoise:
            print('!!!! corrupt and denoise')
            for (noise_type, rate) in itertools.product(ND_list_noise_type, ND_list_noise_density):

                n_edges = len(nx.Graph(G.edges).edges)
                print('!!! n_edges', n_edges)

                path_original = path
                p = np.floor(n_edges * rate)
                if (ntwk == 'COVID_PPI.txt') and (p == np.floor(n_edges * 0.5)) and (noise_type == "-ER_edges"):
                    p = np.floor(n_edges * 0.2)

                if generate_corrupted_ntwk:

                    noise_nodes = len(G.vertices)
                    # noise_nodes = 200
                    # noise_type = 'WS'
                    parameter = p.astype(int)
                    path_save = save_folder + "/" + network_name + "_noise_nodes_" + str(
                        noise_nodes) + "_noisetype_" + noise_type

                    noise_sign = "added"
                    if noise_type == '-ER_edges':
                        noise_sign = "deleted"
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
                    G_corrupt.load_add_wtd_edges(CD_NDL_corrupt_ntwk_path, increment_weights=False, delimiter=',',
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
                                                              is_glauber_dict=is_glauber_dict,
                                                              # keep true to use Glauber chain for dict. learning
                                                              is_glauber_recons=is_glauber_recons,
                                                              omit_folded_edges=omit_folded_edges,
                                                              # keep false to use Pivot chain for recons.
                                                              ONMF_subsample=NDL_onmf_subsample,
                                                              # whether use i.i.d. subsampling for each batch
                                                              Pivot_exact_MH_rule=Pivot_exact_MH_rule)
                ### Set up network dictionary
                if CD_custom_dict_path is not None:
                    # Transfer-denoising
                    result_dict = np.load(CD_custom_dict_path, allow_pickle=True).item()
                    reconstructor_corrupt.W = result_dict.get('Dictionary learned')
                    reconstructor_corrupt.code = result_dict.get('Code learned')
                    print('Dictionary loaded from:', str(CD_custom_dict_path))

                if use_dict_from_corrupted_ntwk:
                    reconstructor_corrupt.W = reconstructor_corrupt.train_dict(update_dict_save=True)
                    # print('corrupted dictionary', reconstructor.W)
                    print('Dictionary learned from the corrupted network')

                else:
                    ### Use below for self-denoising from corrupt dictionary
                    reconstructor_corrupt.W = reconstructor.W

                reconstructor_corrupt.display_dict(title='Dictionary used for denoising ' + str(network_name),
                                                   save_folder=save_folder,
                                                   save_filename='Dict_used_for_denoising' + '_' + str(
                                                       network_name) + '_' + str(
                                                       k) + '_' + str(n_components),
                                                   make_first_atom_2by2=False,
                                                   show_importance=show_importance)

                np.save(save_folder + "/full_result_" + str(network_name) + "_" + "Use_corrupt_dict_" + str(
                    use_dict_from_corrupted_ntwk) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type,
                        reconstructor_corrupt.result_dict)

                title = 'Dictionary learned:' + str(network_name) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type
                save_filename = 'Network_dict' + '_' + str(network_name) + '_' + str(
                    k) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + '_' + str(n_components) + "_noisetype_" + noise_type

                reconstructor_corrupt.display_dict(title=title,
                                                   save_folder=save_folder,
                                                   save_filename=save_filename,
                                                   make_first_atom_2by2=False,
                                                   show_importance=show_importance)

                ### Denoising
                # iter = np.floor(len(G.vertices) * np.log(len(G.vertices)) *4 )
                iter = CD_recons_iter

                G_recons_simple = reconstructor_corrupt.reconstruct_network(recons_iter=iter,
                                                                            jump_every=CD_recons_jump_every,
                                                                            if_save_history=CD_if_save_history,
                                                                            ckpt_epoch=CD_ckpt_epoch,
                                                                            if_save_wtd_reconstruction=CD_if_save_wtd_reconstruction,
                                                                            ### Keep true for denoising
                                                                            omit_chain_edges=omit_chain_edges_denoising,
                                                                            omit_folded_edges=omit_folded_edges,
                                                                            ### Keep true for denoising
                                                                            edge_threshold=CD_edge_threshold,
                                                                            edges_added=edges_changed,
                                                                            if_keep_visit_statistics=CD_if_keep_visit_statistics,
                                                                            save_folder=save_folder,
                                                                            save_filename=save_filename)

                print('parameter', parameter)

                save_file_name = save_folder + "/full_result_" + str(network_name) + "_" + str(
                    noise_nodes) + "_n_corrupt_edges_" + str(
                    len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                    iter) + "_Glauber_recons_" + str(is_glauber_recons)

                # np.save(save_file_name, reconstructor_corrupt.result_dict)

                recons_wtd_edgelist = reconstructor_corrupt.result_dict.get('Edges in weighted reconstruction')
                print('!!! All done with reconstruction -- # of reconstucted edges:', len(recons_wtd_edgelist))
                recons_accuracy = reconstructor_corrupt.compute_recons_accuracy(G_recons_simple,
                                                                                if_baseline=True)

                if CD_if_compute_ROC_AUC:
                    ROC_file_name = network_name + "\n" + "ER_" + str(noise_nodes) + "_p_" + str(parameter).replace(
                        '.',
                        '') + "_n_edges_" + noise_sign + "_" + str(
                        len(edges_changed)) + "_iter_" + str(iter)

                    ROC_dict = compute_ROC_AUC(G_original=G,
                                               path_corrupt=path_corrupt,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               is_dict_edges=True,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               save_file_name=ROC_file_name,
                                               save_folder=save_folder,
                                               flip_TF=not omit_chain_edges_denoising,
                                               subtractive_noise=(noise_type == '-ER_edges'))

                    reconstructor_corrupt.result_dict.update(
                        {'False positive rate': ROC_dict.get('False positive rate')})
                    reconstructor_corrupt.result_dict.update({'True positive rate': ROC_dict.get('True positive rate')})
                    reconstructor_corrupt.result_dict.update({'AUC': ROC_dict.get('AUC')})

                    ROC_dict.update({'Dictionary learned': reconstructor_corrupt.W})
                    ROC_dict.update({'Motif size': reconstructor_corrupt.k2 + 1})
                    ROC_dict.update({'Code learned': reconstructor_corrupt.code})

                    save_file_name = save_folder + "/ROC_dict_" + str(
                        network_name) + "_" + str(noise_nodes) + "_n_corrupt_edges_" + str(
                        len(edges_changed)) + "_noisetype_" + noise_type + "_iter_" + str(
                        iter) + "_Glauber_recons_" + str(is_glauber_recons)

                    np.save(save_file_name, ROC_dict)


def Generate_all_dictionary():
    ### Generating all dictionaries
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/test"

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
                          'true_edgelist_for_BA_5000_m_25.txt']

    list_k = [21]  # list of number of nodes in the chain motif -- scale parameter

    list_n_components = [9, 16, 25, 36, 49, 64, 81, 100]  # number of latent motifs to be learned)

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
        is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        omit_folded_edges=True,
        omit_chain_edges_denoising=True,
        show_importance=True,
        # --------------------------- Iteration parameters
        NDL_MCMC_iterations=100,
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
    save_folder = "Network_dictionary/test"

    list_network_files = ['COVID_PPI.txt',
                          'Caltech36.txt',
                          'facebook_combined.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt']

    list_k = [21]  # list of number of nodes in the chain motif -- scale parameter
    list_n_components = [25]  # number of latent motifs to be learned)
    ND_list_noise_type = ["-ER_edges", "ER_edges"]
    # ND_list_noise_type = ["-ER_edges"]
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
        if_learn_fresh=False,
        if_save_fig=False,
        if_recons=False,
        learn_from_reconstruction=False,
        if_corrupt_and_denoise=True,
        generate_corrupted_ntwk=True,
        use_dict_from_corrupted_ntwk=True,
        # --------------------------- Global parameters
        is_glauber_dict=False,  # Use Glauber chain MCMC sampling for dictionary learning (Use Pivot chain if False)
        is_glauber_recons=False,  # Use Glauber chain MCMC sampling for reconstruction
        omit_folded_edges=False,
        omit_chain_edges_denoising=True,
        show_importance=True,
        # --------------------------- Iteration parameters
        CD_NDL_MCMC_iterations=100,
        # CD_custom_dict_path = CD_custom_dict_path, # comment out if custom dictionary is not used
        CD_ckpt_epoch=20000,
        CD_recons_iter=200000,
        CD_if_compute_ROC_AUC=True,
        CD_if_keep_visit_statistics=False
    )
