from helper_functions.final_plots_display import diplay_ROC_plots, all_dictionaries_display, top_dictionaries_display, all_dictionaries_display_rank, recons_display, recons_display_simple, few_dictionaries_display
import numpy as np
from utils.ndl import Network_Reconstructor, patch_masking
from helper_functions.helper_functions import Generate_corrupt_graph, compute_ROC_AUC_colored, display_denoising_stats_list_plot
from NNetwork.NNetwork import NNetwork
#from NNetwork import NNetwork as nn
import networkx as nx
import csv
import tracemalloc
import itertools
from multiprocessing import Pool
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.decomposition import SparseCoder

def recons_network(arglist):
  G1, name1, dictionary_name, nc, k2, full_output  = arglist
  print('!!! reconstructing network {}'.format(name1))

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
  load_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
  reconstructor.result_dict = np.load(f'{load_folder}/full_result_{dictionary_name}_k_{number_of_nodes}_r_{nc}.npy', allow_pickle=True).item()
  reconstructor.W = reconstructor.result_dict.get('Dictionary learned')
  reconstructor.code = reconstructor.result_dict.get('Code learned')


  recons_iter = int(len(G1.vertices) * np.log(len(G1.vertices)))
  if len(G1.vertices) < 1500:
      recons_iter *= 4
  reconstructor.result_dict.update({'reconstruction iterations': iter})

  save_folder = "Network_dictionary/recons_plot_data"
  if full_output:
    G_recons_wtd = reconstructor.reconstruct_network(recons_iter=recons_iter,
                                             #if_construct_WtdNtwk=True,
                                             if_save_history=False,
                                             if_keep_visit_statistics=False,
                                             #use_checkpoint_refreshing=False,
                                             ckpt_epoch=50000,
                                             omit_chain_edges=False)
    G_recons_wtd.save_wtd_edgelist(save_folder, f"{name1}_recons_for_nc_{nc}_from_{dictionary_name}.txt")
    print('file saved = {}'.format(name1))
  else:
    G_recons_simple = reconstructor.reconstruct_network(recons_iter=recons_iter,
                                             #if_construct_WtdNtwk=True,
                                             if_save_history=False,
                                             if_keep_visit_statistics=False,
                                             #use_checkpoint_refreshing=False,
                                             ckpt_epoch=50000,
                                             omit_chain_edges=False)

    G_recons_simple = G_recons_simple.threshold2simple(0.5)
    recons_accuracy = reconstructor.compute_recons_accuracy(G_recons_simple)
    recons_metrics = reconstructor.compute_recons_accuracy(G_recons_simple, output_full_metrics=True)
    print('nc = {}, recons_accuracy={}'.format(nc,  recons_metrics.get("Jaccard_recons_accuracy")))
    np.save(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.npy", recons_metrics)



def recons_network_nc_list(arglist):
  G1, name1, dictionary_name, nc_list, k, full_output  = arglist
  print('@@@ Reconstructing {} using dictionary learned from {}'.format(name1, dictionary_name))
  recons_iter = int(len(G1.vertices) * np.log(len(G1.vertices)))
  if len(G1.vertices) < 1500:
      recons_iter *= 4

  skip_folded_hom_recons = False

  reconstructor = Network_Reconstructor(G=G1,  #  simple graph
                                  n_components=nc_list[0],  # num of dictionaries
                                  MCMC_iterations=200,  # MCMC steps (macro, grow with size of ntwk)
                                  loc_avg_depth=1,  # keep it at 1
                                  sample_size=1000,  # number of patches in a single batch
                                  batch_size=20,  # number of columns used to train dictionary
                                  # within a single batch step (keep it)
                                  sub_iterations=100,  # number of iterations of the
                                  # sub-batch learning (keep it)
                                  k1=0, k2=k,  # left and right arm lengths
                                  alpha=1,  # parameter for sparse coding, higher for stronger smoothing
                                  if_wtd_network=False,
                                  if_tensor_ntwk=False,
                                  #is_glauber_recons=False,  # keep false to use Pivot chain for recons.
                                  ONMF_subsample=True,  # whether use i.i.d. subsampling for each batch
                                  omit_folded_edges=True)




  W_list = []
  reconstructor.result_dict.update({'reconstruction iterations': recons_iter})
  load_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
  for nc in nc_list:
      number_of_nodes = k + 1
      reconstructor.result_dict = np.load(f'{load_folder}/full_result_{dictionary_name}_k_{number_of_nodes}_r_{nc}.npy', allow_pickle=True).item()
      W = reconstructor.result_dict.get('Dictionary learned')
      #reconstructor.code = reconstructor.result_dict.get('Code learned')
      W_list.append(W.copy())

  #print("!!!! W_list", (W_list is None))

  save_folder = "Network_dictionary/recons_plot_data"
  if full_output:
    G_recons_colored = reconstructor.reconstruct_network_list(W_list = W_list,
                                 recons_iter=recons_iter,
                                 if_save_history=False,
                                 ckpt_epoch=30000,  # not used if None
                                 jump_every=1000,
                                 masking_params_list = [1]*len(W_list), # 1 for no masking, 0 for full masking of chain edges
                                 edges_added=None,
                                 test_edges=None,
                                 skip_folded_hom = skip_folded_hom_recons,
                                 if_keep_visit_statistics=False,
                                 save_path = None)

    recons_edges = G_recons_colored.get_edges()
    G_temp = NNetwork()
    G_temp.add_nodes(nodes=[v for v in G1.vertices])
    G_recons_list = [G_temp]*len(nc_list)
    for e in recons_edges:
        weight = G_recons_colored.get_colored_edge_weight(e[0], e[1])
        for i in np.arange(len(nc_list)):
            G_recons_list[i].add_edge(e, weight=weight[i], increment_weights=False)

    for i in np.arange(len(nc_list)):
        G_recons_list[i].save_wtd_edgelist(save_folder, f"{name1}_recons_for_nc_{nc}_from_{dictionary_name}.txt")

    for i in np.arange(len(nc_list)):
        G_recons_simple = G_recons_list[i].threshold2simple(0.4)
        nc = nc_list[i]
        recons_metrics = reconstructor.compute_recons_accuracy(G_recons_simple, output_full_metrics=True)
        print('nc = {}, recons_accuracy={}'.format(nc,  recons_metrics.get("Jaccard_recons_accuracy")))
        np.save(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.npy", recons_metrics)

    print('file saved = {}'.format(name1))
  else:
    G_recons_colored = reconstructor.reconstruct_network_list(W_list = W_list,
                                 #recons_iter=recons_iter,
                                 recons_iter=10000,
                                 if_save_history=False,
                                 ckpt_epoch=30000,  # not used if None
                                 jump_every=1000,
                                 masking_params_list = [1]*len(W_list), # 1 for no masking, 0 for full masking of chain edges
                                 edges_added=None,
                                 test_edges=None,
                                 skip_folded_hom = skip_folded_hom_recons,
                                 if_keep_visit_statistics=False,
                                 save_path = None)

    recons_edges = G_recons_colored.get_edges()
    G_temp = NNetwork()
    G_temp.add_nodes(nodes=[v for v in G1.vertices])
    #G_recons_list = [G_temp]*len(nc_list)
    G_recons_list = []
    for i in np.arange(len(nc_list)):
        G_recons_list.append(NNetwork())

    for e in recons_edges:
        weight = G_recons_colored.get_colored_edge_weight(e[0], e[1])
        #print('weight', weight)
        for i in np.arange(len(nc_list)):
            G_recons_list[i].add_edge(e, weight=weight[i], increment_weights=False)

    for i in np.arange(len(nc_list)):
        G_recons_simple = G_recons_list[i].threshold2simple(0.4)
        nc = nc_list[i]
        recons_metrics = reconstructor.compute_recons_accuracy(G_recons_simple, output_full_metrics=True)
        print('nc = {}, recons_accuracy={}'.format(nc,  recons_metrics.get("Jaccard_recons_accuracy")))
        np.save(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.npy", recons_metrics)

        #file = open(f"{save_folder}/{name1}_recons_score_for_nc_{nc}_from_{dictionary_name}.txt", "w+")
        #file.write(str(recons_accuracy))
        #file.close()


def accuracyScoreByTheta(G_original,G_recons,name):
    accuracies = []

    edges_recons = G_recons.get_edges()

    common_edges_recons_binned, uncommon_edges_recons_binned = [0]*101 , [0]*101
    for i in trange(len(edges_recons)):
        edge = edges_recons[i]
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

    save_folder = "Network_dictionary/recons_plot_data"
    file = open(f"{save_folder}/self_recons_{name}_vary_threshold.txt", "w+")
    for recons_accuracy in reversed(accuracies):
      file.write(str(recons_accuracy) + "\n")
      print('!!! recons_accuracy', recons_accuracy)
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
    inputs = []

    if nc==None:
        nc_range = [i**2 for i in range(3,11)]
        k2_range = [20]
    else:
        nc_range = [nc]
        k2_range = [20]

    edgelist = np.genfromtxt(f"{directory_network_files}{network_to_recons_name}.txt", delimiter=',', dtype=str)
    edgelist = edgelist.tolist()
    G = NNetwork()
    G.add_edges(edges=edgelist)

    for dictionary_name in network_names_dict:
        inputs += [(copy.copy(G),network_to_recons_name,dictionary_name,nc,k2, False) for nc in nc_range for k2 in k2_range]

    if __name__ == '__main__':
        with Pool(pool_size) as p:
            p.map(recons_network,inputs)




def compute_all_self_recons(compute_error_bd=False):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/recons_inj_test"


    #Panel A: Self-Recons of Caltech, arXiv, Coranvirus PPI, Facebook, and Homo Sapiens PPI
    k_list = [20]
    #nc_list = [25]
    nc_list = [i**2 for i in range(3,8)]
    inputs = []
    network_names_recons = ["arxiv","COVID_PPI", "facebook_combined","node2vec_homosapiens_PPI"]
    #network_names_recons = ["MIT8", "UCLA26", "Harvard1"]
    #network_names_recons = ["COVID_PPI"]

    """
    if __name__ == '__main__':
        print('!!! AAA')
        with Pool(1) as p:
            p.map(recons_network,inputs)
    """

    for network in network_names_recons:

        if network == "COVID_PPI":
            k2_range = [10]

        edgelist = np.genfromtxt(f"{directory_network_files}{network}.txt", delimiter=',', dtype=str)
        edgelist = edgelist.tolist()

        if compute_error_bd:
            G = NNetwork()
            G.add_edges(edges=edgelist)

            #inputs += [(copy.copy(G),network,network,[16, 16, 9],k,False) for k in k_list for nc in nc_list]
            inputs = [(copy.copy(G),network,network, nc_list ,k,False) for k in k_list]
            for input in inputs:
                compute_recons_error_bound(input)
        else:
            G = NNetwork()
            G.add_edges(edges=edgelist)
            inputs = [(copy.copy(G),network,network, nc_list ,k,False) for k in k_list]
            for input in inputs:
                recons_network_nc_list(input)


def compute_recons_error_bound(arglist):
    G1, name1, dictionary_name, nc_list, k, full_output  = arglist
    print('@@@ Computing recons. error bound for ntwk {} using dictionary learned from {}'.format(name1, dictionary_name))
    load_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
    save_folder = "Network_dictionary/recons_plot_data/"

    error_bd_dict = {}
    for i in trange(len(nc_list)):
        nc = nc_list[i]
        number_of_nodes = k + 1
        result_dict = np.load(f'{load_folder}/full_result_{dictionary_name}_k_{number_of_nodes}_r_{nc}.npy', allow_pickle=True).item()
        W = result_dict.get('Dictionary learned')
        #W = np.zeros(W.shape)
        atom_size, num_atoms = W.shape
        W_ext = np.empty((atom_size, 2 * num_atoms))
        W_ext[:, 0:num_atoms] = W[:, 0:num_atoms]
        W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(W[:, 0:num_atoms])

        error_list = []
        for i in range(10):
            X, embs = G1.get_patches(k=number_of_nodes, sample_size=1000, skip_folded_hom=True)

            coder = SparseCoder(dictionary=W_ext.T,  ### Use extended dictioanry
                                transform_n_nonzero_coefs=None,
                                transform_alpha=1,
                                transform_algorithm='lasso_lars',
                                positive_code=True)

            code = coder.transform(X.T)
            patch_recons = np.dot(W_ext, code.T)
            #print("np.min(patch_recons)", np.min(patch_recons) )
            #patch_recons_error = patch_masking(X, k=number_of_nodes, chain_edge_masking=0)
            #patch_recons_red = patch_masking(patch_recons, k=k, chain_edge_masking=0)
            #X_red = patch_masking(X, k=k, chain_edge_masking=0)
            error1 = np.linalg.norm(np.mean(X - patch_recons, axis=-1), ord=1)/(2*number_of_nodes)
            error = np.linalg.norm(np.mean(X - patch_recons, axis=-1), ord=2)/(2) # Cauchy-Schwarz bd

            error_list.append(error)
        error_bd_dict.update({str(nc) : np.mean(error_list)})
        print("L1 patch reconstruction error = ", error1)
        print(f"recons_error_bd_{name1}_from_{dictionary_name}_nc_{nc}_k_{k} = ", np.mean(error_list))
        np.save(f"{save_folder}/recons_error_bd_{name1}_from_{dictionary_name}_nc_{nc}_k_{k}.npy" ,error_bd_dict)


def compute_all_cross_recons(use_parallel_processing=False, compute_error_bd=True):
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/recons_plot_data/"

    # Panels B-E Cross-Recons Scores
    nodes = 5000
    #p_values_ER = [50/(nodes-1), 100/(nodes-1)]
    p_values_ER = [100/(nodes-1)]
    ER = [ f"true_edgelist_for_ER_{nodes}_mean_degree_{round(p*(nodes-1))}" for p in p_values_ER]

    #p_values_SW = [0.05, 0.1]
    p_values_SW = [0.1]
    k_values_SW = [50]
    SW = [ f"true_edgelist_for_SW_{nodes}_k_{k}_p_{str(round(p,2)).replace('.','')}" for k in k_values_SW for p in p_values_SW]

    m_values_BA = [25,50]
    m_values_BA = [50]
    BA = [ f"true_edgelist_for_BA_{nodes}_m_{m}"for m in m_values_BA]

    SBM = ["SBM1", "SBM2"]
    SBM = ["SBM2"]
    synth_network_file_names =   ER+SW+BA+SBM

    #pool_size = [32, 32, 4, 8] #corresponds to network order below C M U H
    pool_size = [8]
    nc_list = [i**2 for i in range(3,10)]
    #nc_list = [25]
    k_list = [20]
    #k_list = [5, 10, 20, 50]
    facebook_networks_file_names = ["MIT8", "UCLA26", "Caltech36", "Harvard1"]
    #ntwk = "COVID_PPI"
    #facebook_networks_file_names = [ntwk]
    network_names_dict  = ["MIT8", "UCLA26", "Caltech36", "Harvard1"] + synth_network_file_names
    #network_names_dict = synth_network_file_names
    #network_names_dict  = ["MIT8", "UCLA26", "Caltech36", "Harvard1"]
    #network_names_dict  = [ntwk]

    if use_parallel_processing:
        for i in range(len(pool_size)):
            recons_facebook_accuracy_parallel(facebook_networks_file_names[i],network_names_dict,pool_size[i])
    else:
        for network in facebook_networks_file_names:
            for network_dict in network_names_dict:
                edgelist = np.genfromtxt(f"{directory_network_files}{network}.txt", delimiter=',', dtype=str)
                edgelist = edgelist.tolist()

                if compute_error_bd:
                    G = NNetwork()
                    G.add_edges(edges=edgelist)

                    #inputs += [(copy.copy(G),network,network,[16, 16, 9],k,False) for k in k_list for nc in nc_list]
                    inputs = [(copy.copy(G),network,network_dict, nc_list ,k,False) for k in k_list]
                    for input in inputs:
                        compute_recons_error_bound(input)
                else:
                    G = NNetwork()
                    G.add_edges(edges=edgelist)
                    inputs = [(copy.copy(G),network,network_dict, nc_list ,k,False) for k in k_list]
                    for input in inputs:
                        recons_network_nc_list(input)



def compute_all_self_recons_threshold():
    #Panel A: Threshold Values from WTD Network
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/recons_plot_data/"
    network_names_recons = ['Caltech36']
    for network in network_names_recons:
        edgelist = np.genfromtxt(f"{directory_network_files}/{network}.txt", delimiter=',', dtype=str)
        edgelist = edgelist.tolist()
        G_original = NNetwork()
        G_original.add_edges(edges=edgelist)
        recons_directory = save_folder
        network_file = f"{network}_recons_for_nc_25_from_{network}.txt.txt"
        G_recons = NNetwork()
        G_recons.load_add_edges(recons_directory+network_file)
        accuracyScoreByTheta(G_original,G_recons,network)

    print('!!! 1')
    network_names_recons = ["arxiv","COVID_PPI", "facebook_combined","node2vec_homosapiens_PPI"]
    for network in network_names_recons:
        edgelist = np.genfromtxt(f"{directory_network_files}/{network}.txt", delimiter=',', dtype=str)
        print('!!! 2')
        edgelist = edgelist.tolist()
        G_original = NNetwork()
        G_original.add_edges(edges=edgelist)
        recons_directory = save_folder
        network_file = f"{network}_recons_for_nc_25_from_{network}.txt.txt"
        G_recons = NNetwork()
        G_recons.load_add_edges(recons_directory+network_file)
        print('!!! 3')
        accuracyScoreByTheta(G_original,G_recons,network)

#compute_all_self_recons()
#compute_all_cross_recons()
#compute_all_self_recons_threshold()
