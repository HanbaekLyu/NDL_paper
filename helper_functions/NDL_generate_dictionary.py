import numpy as np
from utils.ndl import Network_Reconstructor
from helper_functions.helper_functions import display_dict_and_graph
#from utils.NNetwork import NNetwork
from NNetwork.NNetwork import NNetwork
import networkx as nx
import matplotlib.pyplot as plt
import itertools

def Generate_all_dictionary(list_network_files=None,
                            k_list=[21],
                            r_list=[9, 16, 36, 49, 64, 81, 100] ,
                            omit_folded_edges=True,
                            show_importance=True,
                            NDL_MCMC_iterations=20):
    ### Generating all dictionaries
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
    if_save_fig = True

    if list_network_files is None:
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

    #k_list = [51]  # list of number of nodes in the chain motif -- scale parameter
    #r_list = [25]  # number of latent motifs to be learned)

    NDL_onmf_sub_minibatch_size=20
    NDL_onmf_sub_iterations=100
    NDL_alpha=1  # L1 sparsity regularizer for sparse coding
    NDL_onmf_subsample=True  # subsample from minibatches
    NDL_skip_folded_hom = True # if true, only use injective homomorphisms during dictionary learning (denoising not affected)
    NDL_jump_every=10

    for (k, ntwk, n_components) in itertools.product(k_list, list_network_files, r_list):
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
                                              sample_size=1000,  # number of patches in a single batch
                                              batch_size=NDL_onmf_sub_minibatch_size,
                                              # number of columns used to train dictionary
                                              # within a single batch step (keep it)
                                              sub_iterations=NDL_onmf_sub_iterations,  # number of iterations of the
                                              # sub-batch learning (keep it)
                                              k1=0, k2=k - 1,  # left and right arm lengths
                                              alpha=NDL_alpha,
                                              # parameter for sparse coding, higher for stronger smoothing
                                              sampling_alg = 'pivot',
                                              # keep false to use Pivot chain for recons.
                                              ONMF_subsample=NDL_onmf_subsample,
                                              # whether use i.i.d. subsampling for each batch
                                              omit_folded_edges=omit_folded_edges)

        reconstructor.result_dict.update({'Network name': network_name})
        reconstructor.result_dict.update({'# of nodes': len(G.vertices)})

        reconstructor.train_dict(jump_every=NDL_jump_every,
                                     skip_folded_hom=NDL_skip_folded_hom)

        np.save(
            save_folder + "/full_result_" + str(network_name) + "_k_" + str(k) + "_r_" + str(n_components),
            reconstructor.result_dict)

        if if_save_fig:
            ### save dictionaytrain_dict figures
            title = 'Latent motifs learned from ' + str(network_name)
            save_path = save_folder + '/Network_dict' + '_' + str(network_name) + '_' + str(k) + '_' + str(n_components)

            display_dict_and_graph(title=title,
                                         save_path=save_path,
                                         grid_shape=None,
                                         fig_size=[10,10],
                                         W = reconstructor.W,
                                         At = reconstructor.At,
                                         plot_graph_only=False,
                                         show_importance=False)

        print('Finished dictionary learning from network ' + str(ntwk))
