import numpy as np
from utils.ndl import Network_Reconstructor
from NNetwork.NNetwork import NNetwork
from helper_functions.node2vec_helper import run_node2vec, run_DeepWalk, run_spectral
from helper_functions.NDR_denoising_helper import run_NDR_denoising, run_NDR_denoising_CV
from helper_functions.helper_functions import Generate_corrupt_graph, compute_ROC_AUC
import networkx as nx
import csv
import tracemalloc
import itertools
from multiprocessing import Pool
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
import os

import tqdm
from tqdm import trange


def preferential_attachment(G, test_edges):
    recons_wtd_edgelist = []
    for e in test_edges:
        wt = len(G.neighbors(e[0])) * len(G.neighbors(e[1]))
        wtd_edge = [e[0], e[1], wt]
        recons_wtd_edgelist.append(wtd_edge)
    return recons_wtd_edgelist


def triadic(G):

  e = list(G.edges())


  new_edges = []

  for i in e:
    a, b = i

    for j in e:
      x, y = j

      if i != j:
        if a == x and (b, y) not in e and (y, b) not in e:
          new_edges.append((b, y))
        if a == y and (b, x) not in e and (x, b) not in e:
          new_edges.append((b, x))
        if b == x and (a, y) not in e and (y, a) not in e:
          new_edges.append((a, y))
        if b == y and (a, x) not in e and (x, a) not in e:
          new_edges.append((a, x))

  return new_edges



def generate_corrupt_networks(
        directory_network_files="",
        save_folder="",
        # -------------------------- loop parameters
        list_network_files=[],
        ND_list_noise_type=[]
        ):

    for ntwk in list_network_files:

        path = directory_network_files + ntwk
        network_name = ntwk.replace('.txt', '')
        network_name = network_name.replace('.', '')


        G = NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)

        # take the largest component
        G_nx = nx.Graph(G.get_edges())
        G_nx = G_nx.subgraph(sorted(nx.connected_components(G_nx), key=len, reverse=True)[0])
        G = NNetwork()
        G.add_edges(list(G_nx.edges))

        for noise_type in ND_list_noise_type:
            path_save = save_folder + "/" + network_name + "_noisetype_" + noise_type + ".txt"
            n_edges = len(nx.Graph(G.get_edges()).edges)
            path_original = path
            rate = 0.1
            if noise_type in ['ER', '-ER']:
                parameter = 0.2
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

            elif noise_type == 'BA':
                #noise_nodes=len(G.vertices)
                noise_nodes = 500
                parameter = 50 # each new node introduced attaches to five existing nodes preferentially

            elif noise_type == '-BA':
                noise_nodes=len(G.vertices)
                parameter = rate

            elif noise_type == '-ER_walk':
                noise_nodes=len(G.vertices)
                parameter = 0.99

            elif noise_type == '-deg':
                noise_nodes=len(G.vertices)
                parameter = 0.5

            G_corrupt, edges_changed = Generate_corrupt_graph(path_load=path_original,
                                                              delimiter=' ',
                                                              G_original=G,
                                                              path_save=path_save,
                                                              noise_nodes=noise_nodes,
                                                              parameter=parameter,
                                                              noise_type=noise_type)

            print(path_save)
    return G_corrupt, edges_changed


def get_roc_score(edges_pos, edges_neg, score_matrix, adj_sparse):
    # Store positive edge predictions, actual values
    print("len(edges_pos)", len(edges_pos))
    print("len(edges_neg)", len(edges_neg))
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]]) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]]) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def edge_splits(G, G_corrupt, noise_sign = "deleted", k_mean=10):

    #nodes = list(nx.Graph(G.get_edges()).nodes)

    nodes = list(nx.Graph(G_corrupt.get_edges()).nodes)
    nodes_set = set(nodes)

    original_edges = set([(e[0],e[1]) for e in G.get_edges() if e[0] in nodes_set and e[1] in nodes_set])
    corrupt_edges = set([(e[0],e[1]) for e in G_corrupt.get_edges() if e[0] in nodes_set and e[1] in nodes_set])

    if noise_sign == "deleted": #(following node2vec paper setting) # but the resulting link prediction is too easy for sparse graphs
        deleted_edges = list(original_edges.difference(corrupt_edges))
        nonedges_false = []
        G_nx = nx.Graph(G_corrupt.get_edges())
        G_true_nx = nx.Graph(G.get_edges())

        # positive examples = deleted edges
        # netative exampels = non-edges in G_corrupt (observed graph)
        print('creating set of nonedges for negative examples...')
        #deg_seq = [d for n, d in G_nx.degree()]
        #dist = deg_seq/np.sum(deg_seq)

        V = G.nodes()


        """
        deleted_edges = []
        for i in np.arange(len(V)):
            for j in np.arange(i, len(V)):
                if not G_corrupt.has_edge(V[i], V[j]) and np.random.rand() < 0.1:

                    if G.has_edge(V[i], V[j]):
                        deleted_edges.append([V[i], V[j]])
                    else:
                        nonedges_false.append([V[i], V[j]])


        """
        with tqdm.tqdm(total=len(deleted_edges)) as pbar:
            while(len(nonedges_false) < len(deleted_edges)):
                #u = np.random.choice(nodes, 1, p = dist, replace=False)
                #v = np.random.choice(nodes, 1, p = dist, replace=False)
                #e = np.random.choice(nodes, 2, p = dist, replace=False)
                e = np.random.choice(nodes, 2, replace=False)
                e = (str(e[0]), str(e[1]))
                #e = (str(u), str(v))
                if e[0] >= e[1]:
                    continue
                if e in nonedges_false:
                    continue
                if e in original_edges or (e[1],e[0]) in original_edges:
                    continue
                nonedges_false.append(e)
                pbar.update(1)



            """
            while (len(nonedges_false) < len(deleted_edges)):
                #k = np.random.randint(false_edge_range[0], false_edge_range[1])
                k = np.maximum(np.random.geometric(p=1/k_mean), 5)
                X, embs = G.get_patches(k=k, emb=None, sample_size=10, skip_folded_hom=False)
                s = 0
                #while (s < X.shape[1]) and (len(nonedges_false) < len(deleted_edges)):
                for s in np.arange(X.shape[1]):
                    emb = embs[s]
                    H = G.subgraph(nodelist=emb)
                    for u in H.nodes():
                        for v in H.nodes():
                            if not H.has_edge(u,v):
                                e = (str(u), str(v))
                                #e = (str(u), str(v))
                                if e[0] >= e[1]:
                                    continue
                                if e in nonedges_false:
                                    continue
                                if e in original_edges or (e[1],e[0]) in original_edges:
                                    continue
                                nonedges_false.append(e)
                                s += 1
                                pbar.update(1)
                                #print('!! s', s)
                                if len(nonedges_false) < len(deleted_edges):
                                    break

                """




        # nonedges_false is a list of nonedges in G_corrupt that are also nonedges in G
        # and has the same size as deleted_edges
        X = deleted_edges + nonedges_false
        y = [1]*len(deleted_edges) + [0]*len(nonedges_false)
        print('splitting the examples into train/test sets...')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    else: # additive noise

        added_edges = list(corrupt_edges.difference(original_edges))

        edges_true = list(original_edges)
        # positive examples = original edges in G
        # netative exampels = added edges in G - G_corrupt

        #X = edges_true + added_edges
        X = list(corrupt_edges)
        y = [1]*len(edges_true) + [0]*len(added_edges)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    train_edges_true = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==1]
    train_edges_false = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==0]
    test_edges_true = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==1]
    test_edges_false = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==0]

    print('len(train_edges_false)', len(train_edges_false))
    print('len(test_edges_false)', len(test_edges_false))
    print('len(train_edges_true)', len(train_edges_true))
    print('len(test_edges_true)', len(test_edges_true))


    return train_edges_false, train_edges_true, test_edges_false, test_edges_true


def run_link_prediction_all(  # ========================== Master parameters
    directory_network_files="",
    save_folder="Network_dictionary/barplot1",
    # -------------------------- loop parameters
    list_network_files=[],
    ND_list_noise_type=[],
    k_mean = 10,
    method_names = ['jaccard', 'adamic_adar_index', 'preferential_attachment', 'DeepWalk', 'node2vec', 'NDL+NDR'],
):

    print("list_network_files", list_network_files)
    for ntwk in list_network_files:
        print("!!! network={}".format(ntwk))

        path = directory_network_files + ntwk
        network_name = ntwk.replace('.txt', '')
        network_name = network_name.replace('.', '')


        G = NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)

        # take the largest connected component
        G_nx = nx.Graph(G.get_edges())
        G_nx = G_nx.subgraph(sorted(nx.connected_components(G_nx), key=len, reverse=True)[0])
        G = NNetwork()
        G.add_edges(list(G_nx.edges))

        funcs = {}
        funcs.update({'jaccard':nx.jaccard_coefficient})
        funcs.update({'adamic_adar_index':nx.adamic_adar_index})
        funcs.update({'preferential_attachment':nx.preferential_attachment})

        ROC_dicts = []

        noise_nodes = len(G.vertices)

        for noise_type in ND_list_noise_type:
            output_dict = {}
            path_save = save_folder + "/" + network_name + "_noisetype_" + noise_type
            print("!!! path_save", path_save)
            if os.path.isfile(path_save + "_output_dict.npy"):
                output_dict = np.load(path_save + "_output_dict.npy", allow_pickle=True).item()

            n_edges = len(nx.Graph(G.get_edges()).edges)
            path_original = path
            #p = np.floor(n_edges * 0.5)
            #if (ntwk == 'COVID_PPI.txt') and (p == np.floor(n_edges * 0.5)) and (noise_type in ["-ER_edges", "-BA"]):
            #    p = np.floor(n_edges * 0.2)

            noise_sign = "added"
            if noise_type in ['-ER_edges','-ER','-BA', '-ER_walk']:
                noise_sign = "deleted"

            path_corrupt = path_save + ".txt"


            G_corrupt = NNetwork()
            G_corrupt.load_add_edges(path_corrupt, increment_weights=False, delimiter=',',
                                             use_genfromtxt=True)


            # get training and testing edges
            train_edges_false, train_edges_true, test_edges_false, test_edges_true =  edge_splits(G, G_corrupt,
                                                                                                  noise_sign = noise_sign,
                                                                                                  k_mean=k_mean)
            test_edges = test_edges_false + test_edges_true

            print('!!! num test edges: {}'.format(len(test_edges)))

            """
            recons_wtd_edgelist, denoising_dict = run_NDR_denoising_CV(G_corrupt, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G, path_corrupt, noise_type)

            ROC_dict = compute_ROC_AUC(G_original=G,
                                       path_corrupt=path_corrupt,
                                       recons_wtd_edgelist=recons_wtd_edgelist,
                                       #is_dict_edges=True,
                                       delimiter_original=',',
                                       delimiter_corrupt=',',
                                       test_edges = test_edges,
                                       save_file_name="Network_dictionary/ROC_file_test",
                                       save_folder=save_folder,
                                       flip_TF=False,
                                       subtractive_noise=(noise_sign == 'deleted'))

            """

            for a in np.arange(1):

                denoising_dict = {}
                for i in range(len(method_names)):

                    name = method_names[i]

                    print("Running link prediction with...", name)

                    if name == 'node2vec':
                        recons_wtd_edgelist = run_node2vec(G_corrupt, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G, path_corrupt, noise_type)

                    elif name == 'DeepWalk':
                        recons_wtd_edgelist = run_DeepWalk(G_corrupt, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G, path_corrupt, noise_type)

                    elif name == 'NDL+NDR':
                        #recons_wtd_edgelist = run_NDR_denoising(G_corrupt, test_edges, k=20, r=25)
                        recons_wtd_edgelist, denoising_dict = run_NDR_denoising_CV(G_corrupt, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G, path_corrupt, noise_type)

                    elif name == 'spectral':
                        recons_wtd_edgelist = run_spectral(G_corrupt, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G, path_corrupt, noise_type)

                    else:
                        f = funcs.get(name)
                        #recons_wtd_edgelist = list(f(nx.Graph(G_corrupt.get_edges()), test_edges + train_edges_false))
                        recons_wtd_edgelist = list(f(nx.Graph(G_corrupt.get_edges()), test_edges))


                    ROC_file_name = network_name + "\n" + name + "_noisetype_" + noise_type

                    ROC_dict = compute_ROC_AUC(G_original=G,
                                               path_corrupt=path_corrupt,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               save_file_name=ROC_file_name,
                                               subtractive_noise=(noise_sign == 'deleted'),
                                               save_folder=save_folder)

                    ROC_dicts.append(ROC_dict)

                    ROC_dict.update({"recons_wtd_edgelist":recons_wtd_edgelist})
                    ROC_dict.update({"denoising_dict":denoising_dict})
                    #print("!!! denoising_dict", denoising_dict)

                    e = a
                    while output_dict.get(name + "_trial_{}".format(int(e))) is not None:
                        e += 1

                    output_dict.update({name + "_trial_{}".format(int(e)) :ROC_dict})
                np.save(path_save + "_output_dict", output_dict)

        # print out results

        for i in range(len(method_names)):
            name = method_names[i]
            ROC_dict = ROC_dicts[i]
            print("method={}, ACC={}, AUC={}".format(name, np.mean(ROC_dict['Accuracy']), np.mean(ROC_dict['AUC'])))


    return ROC_dicts


def Generate_corrupt_and_denoising_results(
    #method_names = ['jaccard', 'adamic_adar_index', 'preferential_attachment', 'spectral', 'DeepWalk', 'node2vec', 'NDL+NDR'],
    #method_names = ['jaccard', 'preferential_attachment', 'spectral', 'DeepWalk', 'node2vec', 'NDL+NDR'],
    method_names = ['jaccard', 'preferential_attachment', 'spectral', 'DeepWalk', 'NDL+NDR'],
    #method_names = ['jaccard', 'preferential_attachment'],
    #method_names = ['NDL+NDR'],
    save_folder = "",
    ):
    print('!! Generate & denoise experiment started..')
    ### Generating all dictionaries
    directory_network_files = "Data/Networks_all_NDL/"
    save_folder = "Network_dictionary/barplot1"

    list_network_files = ['COVID_PPI.txt',
                          'Caltech36.txt',
                          'arxiv.txt',
                          'node2vec_homosapiens_PPI.txt',
                          'facebook_combined.txt']


    list_network_files = ['Caltech36.txt']

    #ND_list_noise_type = ["-ER", "ER", "BA", "-BA"]
    #ND_list_noise_type = ["-ER_walk"]
    ND_list_noise_type = ["-ER"]
    #ND_list_noise_type = ["-ER"]

    #list_network_files = ['COVID_PPI.txt']
    #list_network_files = ['facebook_combined.txt']
    #list_network_files = ['Caltech36.txt']
    #list_network_files = ['Caltech36.txt',
    #                      'facebook_combined.txt',
    #                      'arxiv.txt']
    #list_network_files = ['facebook_combined.txt']
    #list_network_files = ['Caltech36.txt']
    #list_network_files = ['arxiv.txt']
    #list_network_files = ['node2vec_homosapiens_PPI.txt']
    #list_network_files = ['COVID_PPI.txt']

    generate_corrupt_networks(
        directory_network_files=directory_network_files,
        save_folder=save_folder,
        list_network_files=list_network_files,
        ND_list_noise_type=ND_list_noise_type)


    run_link_prediction_all(
        directory_network_files=directory_network_files,
        save_folder=save_folder,
        list_network_files=list_network_files,
        ND_list_noise_type=ND_list_noise_type,
        method_names = method_names,
        k_mean=10)

#method_names=['jaccard', 'preferential_attachment', 'spectral', 'DeepWalk', 'node2vec', 'NDL+NDR']

#Generate_corrupt_and_denoising_results(method_names=method_names)
