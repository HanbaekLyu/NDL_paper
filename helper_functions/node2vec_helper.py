#Imports
import argparse
import tqdm
import numpy as np
import networkx as nx
from NNetwork.NNetwork import NNetwork
from helper_functions import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

from helper_functions.helper_functions import compute_ROC_AUC
from sklearn.model_selection import train_test_split


def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks, args, verbose=False):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks_str = []
    for walk in walks:
        walks_str.append([str(x) for x in walk])

    print("Beginning word2vec...")

    model = Word2Vec(sentences=walks_str,
                     vector_size=args.get("emb_dim"), # new gensim uses argument 'vector_size' instaed of 'size'
                     window=args.get("window_size"),
                     min_count=0, sg=1,
                     workers=args.get("workers"),
                     epochs=args.get("SGD_iter")) # 'formerlly 'iter''
    model.wv.save_word2vec_format(args.get("output"))

    print("Finished word2vec...")

    # Store just the words + their trained embeddings as a numpy dictionary
    word_vectors = model.wv
    my_dict = dict({})
    keys = []
    for idx, key in enumerate(model.wv.key_to_index):
        my_dict[key] = model.wv[key]

    if verbose:
        # print("motifs embedded : {}".format(my_dict.keys()))
        for word in my_dict.keys():
            print("node \n {} ===> emb {}".format(word, my_dict.get(word)))
    # self.sort_dict(my_dict)
    return my_dict

def compute_recons_edges_dot(my_dict, option="dot_product"):
    # From learned node embedding, compute reconstructed edge weights
    # Use NNetwork graph class
    G_recons = NNetwork()
    node_list = [v for v in my_dict.keys()]
    for u in node_list:
        for v in node_list:
            u_emb = my_dict.get(u)
            v_emb = my_dict.get(v)
            weight = 0
            if option == "dot_product":
                weight = u_emb.T @ v_emb

            G_recons.add_edge(edge=[u,v], weight=weight)
    return G_recons

def compute_edge_embeds(edges, my_dict, option="dot_product"):
    # From learned node embedding, compute edge embeddings
    print("Computing edge embeddings...")

    edge_embeds = []
    for edge in tqdm.tqdm(edges):
        u_emb = my_dict.get(edge[0])
        v_emb = my_dict.get(edge[1])
        #embed = np.multiply(u_emb, v_emb)
        embed = u_emb * v_emb
        edge_embeds.append(embed)

    return edge_embeds

def train(train_embeds, train_embeds_labels):

    classifier = LogisticRegression(random_state=0)
    classifier.fit(train_embeds, train_embeds_labels)
    return classifier

def setup_args():

    args = {}
    args.update({"weighted" : False})
    args.update({"directed" : False})
    args.update({"emb_dim" : 128})
    args.update({"output" : "Network_dictionary/node2vec_output0/test_node2vec.txt"})
    args.update({"walk_length" : 80})
    args.update({"num_walks" : 10})
    args.update({"window_size" : 10})
    args.update({"SGD_iter" : 1})
    args.update({"workers" : 8})
    args.update({"p" : 1}) # return hyperparameter
    args.update({"q" : 1}) # inout hyperparameter

    return args

def run_node2vec(G, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G_original, path_corrupt, noise_type):

    nx_G = nx.Graph()
    for edge in G.get_edges():
        nx_G.add_edge(edge[0], edge[1], weight=1)

    args = setup_args()

    X = train_edges_true + train_edges_false
    y = [1]*len(train_edges_true) + [0]*len(train_edges_false)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    train_edges_true_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==1]
    train_edges_false_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==0]
    val_edges_true = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==1]
    val_edges_false = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==0]

    aucs = []
    accs = []
    ps = []
    qs = []
    dicts = []
    ROC_dicts = []

    for p in [0.25, 0.5, 1, 2]:
        for q in [0.25, 0.5, 1, 2]:

    #for p in [1]:
    #    for q in [1]:
            print('p={}, q={}'.format(np.round(p,3), np.round(q,3)))
            node2vec_object = node2vec.Graph(nx_G, is_directed=False, p=p, q=q)
            node2vec_object.preprocess_transition_probs()
            walks = node2vec_object.simulate_walks(args.get("num_walks"), args.get("walk_length"))
            my_dict = learn_embeddings(walks, args)

            # Compute edge embeddings
            train_edge_embeds_false_hyper = compute_edge_embeds(train_edges_false_hyper, my_dict)
            train_edge_embeds_true_hyper = compute_edge_embeds(train_edges_true_hyper, my_dict)
            val_edge_embeds_false = compute_edge_embeds(val_edges_false, my_dict)
            val_edge_embeds_true = compute_edge_embeds(val_edges_true, my_dict)

            train_edge_embeds_hyper = np.concatenate([train_edge_embeds_true_hyper, train_edge_embeds_false_hyper])
            train_edge_labels_hyper = np.concatenate([np.ones(len(train_edges_true_hyper)), np.zeros(len(train_edges_false_hyper))])

            val_edges = np.concatenate([val_edges_true, val_edges_false])
            val_edge_embeds = np.concatenate([val_edge_embeds_true, val_edge_embeds_false])
            val_edge_labels = np.concatenate([np.ones(len(val_edges_true)), np.zeros(len(val_edges_false))])

            classifier = train(train_edge_embeds_hyper, train_edge_labels_hyper)
            weights = classifier.predict_proba(val_edge_embeds)[:, 1]

            recons_wtd_edgelist = [[val_edges[i][0],val_edges[i][1],weights[i]] for i in range(len(weights))]

            ROC_dict = compute_ROC_AUC(G_original=G_original,
                                               path_corrupt=path_corrupt,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               subtractive_noise=(noise_type == '-ER_edges'))

            aucs.append(ROC_dict['AUC'])
            accs.append(np.mean(ROC_dict['Accuracy']))

            ps.append(p)
            qs.append(q)
            #dicts.append(my_dict)
            dicts.append(my_dict)

    print('!!! aucs', aucs)
    j = np.argmax(aucs)
    print('j', j)

    p = ps[j]
    q = qs[j]
    my_dict = dicts[j]

    train_edge_embeds_false_hyper = compute_edge_embeds(train_edges_false_hyper, my_dict)
    train_edge_embeds_true_hyper = compute_edge_embeds(train_edges_true_hyper, my_dict)
    test_edge_embeds_false = compute_edge_embeds(test_edges_false, my_dict)
    test_edge_embeds_true = compute_edge_embeds(test_edges_true, my_dict)

    train_edge_embeds_hyper = np.concatenate([train_edge_embeds_true_hyper, train_edge_embeds_false_hyper])
    train_edge_labels_hyper = np.concatenate([np.ones(len(train_edges_true_hyper)), np.zeros(len(train_edges_false_hyper))])

    test_edges = np.concatenate([test_edges_true, test_edges_false])
    test_edge_embeds = np.concatenate([test_edge_embeds_true, test_edge_embeds_false])
    test_edge_labels = np.concatenate([np.ones(len(test_edges_true)), np.zeros(len(test_edges_false))])

    classifier = train(train_edge_embeds_hyper, train_edge_labels_hyper)
    weights = classifier.predict_proba(test_edge_embeds)[:, 1]

    recons_wtd_edgelist = [[test_edges[i][0],test_edges[i][1],weights[i]] for i in range(len(weights))]

    return recons_wtd_edgelist


def run_DeepWalk(G, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G_original, path_corrupt, noise_type):

    nx_G = nx.Graph()
    for edge in G.get_edges():
        nx_G.add_edge(edge[0], edge[1], weight=1)

    args = setup_args()

    X = train_edges_true + train_edges_false
    y = [1]*len(train_edges_true) + [0]*len(train_edges_false)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    train_edges_true_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==1]
    train_edges_false_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==0]
    val_edges_true = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==1]
    val_edges_false = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==0]

    aucs = []
    accs = []
    ps = []
    qs = []
    dicts = []
    ROC_dicts = []

    for p in [1]:
        for q in [1]:

    #for p in [1]:
    #    for q in [1]:
            print('p={}, q={}'.format(np.round(p,3), np.round(q,3)))
            node2vec_object = node2vec.Graph(nx_G, is_directed=False, p=p, q=q)
            node2vec_object.preprocess_transition_probs()
            walks = node2vec_object.simulate_walks(args.get("num_walks"), args.get("walk_length"))
            my_dict = learn_embeddings(walks, args)

            # Compute edge embeddings
            train_edge_embeds_false_hyper = compute_edge_embeds(train_edges_false_hyper, my_dict)
            train_edge_embeds_true_hyper = compute_edge_embeds(train_edges_true_hyper, my_dict)
            val_edge_embeds_false = compute_edge_embeds(val_edges_false, my_dict)
            val_edge_embeds_true = compute_edge_embeds(val_edges_true, my_dict)

            train_edge_embeds_hyper = np.concatenate([train_edge_embeds_true_hyper, train_edge_embeds_false_hyper])
            train_edge_labels_hyper = np.concatenate([np.ones(len(train_edges_true_hyper)), np.zeros(len(train_edges_false_hyper))])

            val_edges = np.concatenate([val_edges_true, val_edges_false])
            val_edge_embeds = np.concatenate([val_edge_embeds_true, val_edge_embeds_false])
            val_edge_labels = np.concatenate([np.ones(len(val_edges_true)), np.zeros(len(val_edges_false))])

            classifier = train(train_edge_embeds_hyper, train_edge_labels_hyper)
            weights = classifier.predict_proba(val_edge_embeds)[:, 1]

            recons_wtd_edgelist = [[val_edges[i][0],val_edges[i][1],weights[i]] for i in range(len(weights))]

            ROC_dict = compute_ROC_AUC(G_original=G_original,
                                               path_corrupt=path_corrupt,
                                               recons_wtd_edgelist=recons_wtd_edgelist,
                                               delimiter_original=',',
                                               delimiter_corrupt=',',
                                               subtractive_noise=(noise_type == '-ER_edges'))

            aucs.append(ROC_dict['AUC'])
            accs.append(np.mean(ROC_dict['Accuracy']))

            ps.append(p)
            qs.append(q)
            #dicts.append(my_dict)
            dicts.append(my_dict)

    print('!!! aucs', aucs)
    j = np.argmax(aucs)
    print('j', j)

    p = ps[j]
    q = qs[j]
    my_dict = dicts[j]

    train_edge_embeds_false_hyper = compute_edge_embeds(train_edges_false_hyper, my_dict)
    train_edge_embeds_true_hyper = compute_edge_embeds(train_edges_true_hyper, my_dict)
    test_edge_embeds_false = compute_edge_embeds(test_edges_false, my_dict)
    test_edge_embeds_true = compute_edge_embeds(test_edges_true, my_dict)

    train_edge_embeds_hyper = np.concatenate([train_edge_embeds_true_hyper, train_edge_embeds_false_hyper])
    train_edge_labels_hyper = np.concatenate([np.ones(len(train_edges_true_hyper)), np.zeros(len(train_edges_false_hyper))])

    test_edges = np.concatenate([test_edges_true, test_edges_false])
    test_edge_embeds = np.concatenate([test_edge_embeds_true, test_edge_embeds_false])
    test_edge_labels = np.concatenate([np.ones(len(test_edges_true)), np.zeros(len(test_edges_false))])

    classifier = train(train_edge_embeds_hyper, train_edge_labels_hyper)
    weights = classifier.predict_proba(test_edge_embeds)[:, 1]

    recons_wtd_edgelist = [[test_edges[i][0],test_edges[i][1],weights[i]] for i in range(len(weights))]

    return recons_wtd_edgelist



def spectral_embedding(G, d=128):
    nodes = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodes).todense()
    e, v = np.linalg.eigh(L)
    emb = np.asarray(v)[:,:d]
    my_dict = dict({})
    for i in np.arange(len(nodes)):
        u = nodes[i]
        my_dict.update({u: emb[i,:d]})
    return my_dict


def run_spectral(G, train_edges_false, train_edges_true, test_edges_false, test_edges_true, G_original, path_corrupt, noise_type):

    nx_G = nx.Graph()
    for edge in G.get_edges():
        nx_G.add_edge(edge[0], edge[1], weight=1)

    args = setup_args()

    X = train_edges_true + train_edges_false
    y = [1]*len(train_edges_true) + [0]*len(train_edges_false)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=37)

    train_edges_true_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==1]
    train_edges_false_hyper = [X_train[i] for i in np.arange(len(X_train)) if y_train[i]==0]
    val_edges_true = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==1]
    val_edges_false = [X_test[i] for i in np.arange(len(X_test)) if y_test[i]==0]

    my_dict = spectral_embedding(nx_G, args.get("emb_dim"))

    # Compute edge embeddings
    train_edge_embeds_false_hyper = compute_edge_embeds(train_edges_false_hyper, my_dict)
    train_edge_embeds_true_hyper = compute_edge_embeds(train_edges_true_hyper, my_dict)
    test_edge_embeds_false = compute_edge_embeds(test_edges_false, my_dict)
    test_edge_embeds_true = compute_edge_embeds(test_edges_true, my_dict)

    train_edge_embeds_hyper = np.concatenate([train_edge_embeds_true_hyper, train_edge_embeds_false_hyper])
    train_edge_labels_hyper = np.concatenate([np.ones(len(train_edges_true_hyper)), np.zeros(len(train_edges_false_hyper))])

    test_edges = np.concatenate([test_edges_true, test_edges_false])
    test_edge_embeds = np.concatenate([test_edge_embeds_true, test_edge_embeds_false])
    test_edge_labels = np.concatenate([np.ones(len(test_edges_true)), np.zeros(len(test_edges_false))])

    classifier = train(train_edge_embeds_hyper, train_edge_labels_hyper)
    weights = classifier.predict_proba(test_edge_embeds)[:, 1]

    recons_wtd_edgelist = [[test_edges[i][0],test_edges[i][1],weights[i]] for i in range(len(weights))]

    return recons_wtd_edgelist
