#Imports
import numpy as np
from NNetwork import NNetwork as nn
import networkx as nx
#import utils.NNetwork as nn
import utils.ndl as ndl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from scipy.spatial import ConvexHull
from tqdm import trange
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
import tqdm


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





#### helper functions

def display_graphs(title,
                     save_path,
                     grid_shape=[2,3],
                     fig_size=[10,10],
                     data = None, # [X, embs]
                     show_importance=False):

        # columns of X = vectorized k x k adjacency matrices
        # corresponding list in embs = sequence of nodes (may overalp)
        X, embs = data
        print('X.shape', X.shape)

        rows = grid_shape[0]
        cols = grid_shape[1]

        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        # make outer gridspec

        idx = np.arange(X.shape[1])
        outer_grid = gridspec.GridSpec(nrows=rows, ncols=cols, wspace=0.02, hspace=0.05)

        # make nested gridspecs
        for i in range(rows * cols):
            emb = embs[idx[i]]

            a = i // cols
            b = i % cols

            Ndict_wspace = 0.05
            Ndict_hspace = 0.05

            # display graphs
            inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)

            # get rid of duplicate nodes
            A = X[:,idx[i]]
            A = X[:,idx[i]].reshape(int(np.sqrt(X.shape[0])), -1)
            H = nn.NNetwork()
            H.read_adj(A, embs[idx[i]])
            A_sub = H.get_adjacency_matrix()

            # read in as a nx graph for plotting

            G1 = nx.Graph()
            for a in np.arange(len(emb)):
                G1.add_node(emb[a])

            for a in np.arange(len(emb)):
                for b in np.arange(len(emb)):
                    u = emb[a]
                    v = emb[b]

                    if H.has_edge(u,v):
                        if np.abs(a-b) == 1:
                            G1.add_edge(u,v, color='r', weight=2)
                        elif not G1.has_edge(u,v):
                            G1.add_edge(u,v, color='b', weight=0.5)


            ax = fig.add_subplot(inner_grid[0, 0])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            colors = [G1[u][v]['color'] for u,v in edges]

            weights = [10*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=20, ax=ax,
                    width=weights, edge_color=colors, label='Graph')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')

def display_adj_graph(title,
                     save_path,
                     grid_shape=[2,3],
                     fig_size=[10,10],
                     data = None, # [X, embs]
                     show_importance=False):

        X, embs = data
        k = int(np.sqrt(X.shape[0]))
        print('X.shape', X.shape)

        rows = grid_shape[0]
        cols = grid_shape[1]
        idx = np.arange(X.shape[1])

        Ndict_wspace = 0.05
        Ndict_hspace = 0.05

        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        outer_grid = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.02, hspace=0.05)
        for t in np.arange(2):
            # make nested gridspecs

            if t == 0:
                ### Make gridspec
                inner_grid = outer_grid[t].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    ax.imshow(X[:,idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                    # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])

            if t == 1:
                inner_grid = outer_grid[t].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols

                    G1 = nx.from_numpy_matrix(X[:,idx[i]].reshape(k,k))
                    ax = fig.add_subplot(inner_grid[a, b])
                    pos = nx.spring_layout(G1)
                    edges = G1.edges()
                    weights = [1*G1[u][v]['weight'] for u,v in edges]
                    nx.draw(G1, with_labels=False, node_size=50, ax=ax, width=weights, label='Graph')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.suptitle(title, fontsize=40)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')

def display_dict_and_graph(title=None,
                             save_path=None,
                             grid_shape=None,
                             fig_size=[10,10],
                             W = None,
                             At = None,
                             plot_graph_only=False,
                             show_importance=False,
                             edge_thickness=10):

        n_components = W.shape[1]
        k = int(np.sqrt(W.shape[0]))

        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if grid_shape is not None:
            rows = grid_shape[0]
            cols = grid_shape[1]
        else:
            if rows ** 2 == n_components:
                cols = rows
            else:
                cols = rows + 1

        if At is None:
            idx = np.arange(W.shape[1])
        else:
            importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
            # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
            idx = np.argsort(importance)
            idx = np.flip(idx)

        Ndict_wspace = 0.05
        Ndict_hspace = 0.05

        fig = plt.figure(figsize=fig_size, constrained_layout=False)

        ncols = 2
        if plot_graph_only:
            ncols =1

        outer_grid = gridspec.GridSpec(nrows=1, ncols=ncols, wspace=0.02, hspace=0.05)
        for t in np.arange(ncols):
            # make nested gridspecs

            if t == 1:
                ### Make gridspec
                inner_grid = outer_grid[1].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                    # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])
            if t == 0:
                inner_grid = outer_grid[0].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax = fig.add_subplot(inner_grid[a, b])

                    k = int(np.sqrt(W.shape[0]))
                    A_sub = W[:,idx[i]].reshape(k,k)
                    H = nx.from_numpy_matrix(A_sub)
                    G1 = nx.Graph()
                    for a in np.arange(k):
                        for b in np.arange(k):
                            u = list(H.nodes())[a]
                            v = list(H.nodes())[b]
                            if H.has_edge(u,v):
                                if np.abs(a-b) == 1:
                                    G1.add_edge(u,v, color='r', weight=A_sub[a,b])
                                elif not G1.has_edge(u,v):
                                    G1.add_edge(u,v, color='b', weight=A_sub[a,b])

                    pos = nx.spring_layout(G1)
                    edges = G1.edges()
                    colors = [G1[u][v]['color'] for u,v in edges]
                    weights = [edge_thickness*G1[u][v]['weight'] for u,v in edges]

                    nx.draw(G1, with_labels=False, node_size=20, ax=ax, width=weights, edge_color=colors, label='Graph')

                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

                    ax.set_xticks([])
                    ax.set_yticks([])

        if title is not None:
            plt.suptitle(title, fontsize=25)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')


def display_graphs_dict_graph(data, #[X, embs, W, At] = [subgraph patches, embeddings, dictionary, code]
                             title,
                             subtitle,
                             save_path,
                             grid_shape_subg=None,
                             grid_shape_dict=None,
                             width_ratios=[1,1,1],
                             fig_size=[10,10],
                             show_importance=False):

        X, embs, W, At = data

        n_components = W.shape[1]
        k = int(np.sqrt(W.shape[0]))

        Ndict_wspace = 0.05
        Ndict_hspace = 0.05

        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        outer_grid = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.02, hspace=0.05, width_ratios=width_ratios)
        for t in np.arange(3):
            # make nested gridspecs

            if t == 0:
                ax = plt.Subplot(fig, outer_grid[t])
                ax.set_title(subtitle, fontsize=15)
                ax.axis('off')
                fig.add_subplot(ax)

                idx = np.arange(X.shape[1])
                rows_subgraphs = grid_shape_subg[0]
                cols_subgraphs = grid_shape_subg[1]

                inner_grid = outer_grid[t].subgridspec(rows_subgraphs, cols_subgraphs, wspace=Ndict_wspace, hspace=Ndict_hspace)

                # make nested gridspecs
                for i in range(rows_subgraphs * cols_subgraphs):
                    a0 = i // cols_subgraphs
                    b0 = i % cols_subgraphs

                    Ndict_wspace = 0.05
                    Ndict_hspace = 0.05

                    # get rid of duplicate nodes
                    A = X[:,idx[i]]
                    A = X[:,idx[i]].reshape(int(np.sqrt(X.shape[0])), -1)
                    H = nn.NNetwork()
                    H.read_adj(A, embs[idx[i]])
                    A_sub = H.get_adjacency_matrix()

                    # read in as a nx graph for plotting
                    #G1 = nx.from_numpy_matrix(A_sub)

                    G1 = nx.Graph()
                    emb = embs[idx[i]]
                    for a in np.arange(len(emb)):
                        G1.add_node(emb[a])

                    for a in np.arange(len(emb)):
                        for b in np.arange(len(emb)):
                            u = emb[a]
                            v = emb[b]

                            if H.has_edge(u,v):
                                G1.add_edge(u,v, color='b', weight=1)


                    for a in np.arange(len(emb)):
                        for b in np.arange(len(emb)):
                            u = emb[a]
                            v = emb[b]

                            if H.has_edge(u,v) and np.abs(a-b) == 1:
                                G1.add_edge(u,v, color='r', weight=2)


                    ax = fig.add_subplot(inner_grid[a0, b0])
                    pos = nx.spring_layout(G1)
                    edges = G1.edges()
                    colors = [G1[u][v]['color'] for u,v in edges]
                    weights = [1*G1[u][v]['weight'] for u,v in edges]
                    nx.draw(G1, with_labels=False, node_size=20, ax=ax, edge_color=colors,
                            width=weights, label='Graph')

                    ax.set_xticks([])
                    ax.set_yticks([])

            if t == 1:
                ax = plt.Subplot(fig, outer_grid[t])
                ax.set_title('Latent motifs in matrices', fontsize=15)
                ax.axis('off')
                fig.add_subplot(ax)


                rows = np.round(np.sqrt(n_components))
                rows = rows.astype(int)
                if grid_shape_dict is not None:
                    rows = grid_shape_dict[0]
                    cols = grid_shape_dict[1]
                else:
                    if rows ** 2 == n_components:
                        cols = rows
                    else:
                        cols = rows + 1

                if show_importance:
                    importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
                    # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
                    idx = np.argsort(importance)
                    idx = np.flip(idx)
                else:
                    idx = np.arange(W.shape[1])



                inner_grid = outer_grid[t].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                    # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])

            if t == 2:
                ax = plt.Subplot(fig, outer_grid[t])
                ax.set_title('Latent motifs in graphs', fontsize=15)
                ax.axis('off')
                fig.add_subplot(ax)

                inner_grid = outer_grid[t].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)

                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols

                    G1 = nx.from_numpy_matrix(W[:,idx[i]].reshape(int(np.sqrt(W.shape[0])),-1))
                    ax = fig.add_subplot(inner_grid[a, b])
                    pos = nx.spring_layout(G1)
                    edges = G1.edges()
                    weights = [5*G1[u][v]['weight'] for u,v in edges]
                    nx.draw(G1, with_labels=False, node_size=10, ax=ax, width=weights, label='Graph')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylabel(title, fontsize=20)

        #plt.suptitle(title, fontsize=20)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')


def motif_sample_display_list(list_graphs, k=20, sample_size=100,
                              subtitle = None,
                              skip_folded_hom=True, grid_shape=[5,3],
                              fig_size=[10,3],
                              save_path=None):
    # list of graphs in NNetwork format
    X_list = []
    embs_list = []
    for G in list_graphs:
        X, embs = G.get_patches(k=k, sample_size=sample_size, skip_folded_hom=skip_folded_hom)
        X_list.append(X)
        embs_list.append(embs)


    fig = plt.figure(figsize=fig_size, constrained_layout=False)
    # make outer gridspec

    outer_grid = gridspec.GridSpec(nrows=1, ncols=len(list_graphs), wspace=0.2, hspace=0.2)

    for i in np.arange(len(list_graphs)):
        if subtitle is not None:
            ax = plt.Subplot(fig, outer_grid[i])
            ax.set_title(subtitle[i], fontsize=15)
            ax.axis('off')
            fig.add_subplot(ax)

        Ndict_wspace = 0.05
        Ndict_hspace = 0.05

        rows = grid_shape[0]
        cols = grid_shape[1]

        inner_grid = outer_grid[i].subgridspec(rows, cols,
                                               wspace=Ndict_wspace, hspace=Ndict_hspace)

        idx = np.arange(X.shape[1])
        X = X_list[i]
        embs = embs_list[i]

        for j in range(rows * cols):
            emb = embs[idx[j]]

            a = j // cols
            b = j % cols

            # get rid of duplicate nodes
            A = X[:,idx[j]]
            A = X[:,idx[j]].reshape(int(np.sqrt(X.shape[0])), -1)
            H = nn.NNetwork()
            H.read_adj(A, embs[idx[j]])
            A_sub = H.get_adjacency_matrix()

            # read in as a nx graph for plotting

            G1 = nx.Graph()
            for w in np.arange(len(emb)):
                G1.add_node(emb[w])

            for p in np.arange(len(emb)):
                for q in np.arange(len(emb)):
                    u = emb[p]
                    v = emb[q]

                    if H.has_edge(u,v):
                        if np.abs(p-q) == 1:
                            G1.add_edge(u,v, color='r', weight=2)
                        else:
                            G1.add_edge(u,v, color='b', weight=0.5)



            ax = fig.add_subplot(inner_grid[a, b])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            colors = [G1[u][v]['color'] for u,v in edges]

            weights = [1*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=10, ax=ax,
                    width=weights, edge_color=colors, label='Graph')

            ax.set_xticks([])
            ax.set_yticks([])

        #plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')




def compute_community_stats(ntwk_list, k=20, num_subgraphs=10000, save_path=None):
    subgraphs_community_list = []
    latentmotifs_community_list = []

    for i in trange(len(ntwk_list)):
        ntwk = ntwk_list[i]
        if ntwk == "COVID_PPI.txt":
            k = 10

        print(f"network={ntwk}, k={k}")
        sampling_alg = 'pivot'
        #save_folder = 'Network_dictionary/NDL_rev1/'
        ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
        path = "Data/Networks_all_NDL/" + str(ntwk)
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        print('ntwk={}. num_nodes={}, num_edges={}'.format(ntwk_nonumber, len(G.nodes()), len(G.get_edges()) ) )

        X = []
        embs = []
        for j in trange(10):
            X0, embs0 = G.get_patches(k=k, sample_size=num_subgraphs, skip_folded_hom=True)
            X.append(X0)
            embs.append(embs0)
            #print('X0.shape', X0.shape)

        X = np.asarray(X).reshape(k**2,-1)
        print('X.shape', X.shape)

        a = compute_avg_community(X, embs)
        subgraphs_community_list.append(a.copy())

        NDL = ndl.Network_Reconstructor(G,
                                         n_components=25,
                                         MCMC_iterations=100,
                                         sub_iterations=100,
                                         sample_size=100,
                                         batch_size=10,
                                         k1=0,
                                         k2=k,
                                         sampling_alg='pivot',
                                         if_wtd_network=False)

        #W0 = NDL.train_dict(skip_folded_hom=True)
        load_folder = "Network_dictionary/NDL_inj_dictionary_k_all1"
        result_dict = np.load(f'{load_folder}/full_result_{"COVID_PPI"}_k_{str(k+1)}_r_{25}.npy', allow_pickle=True).item()
        W0 = result_dict.get('Dictionary learned')


        At0 = NDL.At
        b = compute_avg_community(X=W0, embs=None, weights=np.diag(At0)*100, wtd_duplicates=True)
        latentmotifs_community_list.append(b)

    if save_path is not None:
        np.save(save_path, [subgraphs_community_list, latentmotifs_community_list])

    return subgraphs_community_list, latentmotifs_community_list


from networkx.algorithms import community

def compute_avg_community(X, embs, weights=None, wtd_duplicates=False):
    top_community_list = []
    wt_list = []
    #print('embs', len(embs))
    if embs is None:
        k = int(np.sqrt(X.shape[0]))
        embs = [np.arange(k)]*X.shape[1]


    for j in trange(len(embs)):
        emb = embs[j]
        A = X[:,j]
        A = X[:,j].reshape(int(np.sqrt(X.shape[0])), -1)
        H = nn.NNetwork()
        nodelist = np.arange(A.shape[0])
        for i0 in np.arange(len(nodelist)):
            nb_list = H.indices(A[:, i0], lambda x: x > 0)
            for j0 in nb_list:
                H.add_edge(edge=[nodelist[i0],nodelist[j0]], weight=float(A[i0,j0]),
                           increment_weights=False, is_dict=False)

        #H.read_adj(A, embs[j])
        A_sub = H.get_adjacency_matrix()

        H_nx = nx.from_numpy_matrix(A_sub)

        communities_generator =  community.louvain_communities(H_nx)

        wt = 1
        if weights is not None:
            wt = weights[j]/np.sum(weights) * len(embs)

        a = [len(communities_generator[i]) for i in range(len(communities_generator)) ]
        if wtd_duplicates:
            a = a*int(weights[j])

        #top_community_list.append(max(a))
        #top_community_list.append(a)
        top_community_list = top_community_list + a
        wt_list = wt_list + [wt]*len(communities_generator)
        #print(max(a))

    if not wtd_duplicates:
        avg = np.average(top_community_list, weights=wt_list)
        var = np.average((top_community_list-avg)**2, weights=wt_list)
        std = np.sqrt(var)
    else:
        avg = np.mean(top_community_list)
        std = np.std(top_community_list)

    print('mean community size={}, std={}'.format(avg, std) )

    return top_community_list



def show_array(arr):
    ### Plots heatmap of an array
    ### Used to plot network adjcency matrix
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.xaxis.set_ticks_position('top')
    ax.imshow(arr, cmap='viridis', interpolation='nearest')  ### Without 'nearest', 1's look white not yellow.
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.tick_params(axis='x', which='minor', labelsize=8)
    plt.show()


def compute_max_component_stats(path):
    G = nx.Graph()
    edgelist = np.genfromtxt(path, delimiter=',', dtype=str)
    for e in edgelist:
        G.add_edge(e[0], e[1], weight=1)
    Gc = max(nx.connected_components(G), key=len)
    G_conn = G.subgraph(Gc)
    print('num_nodes in max_comp of G=', len(G_conn.nodes))
    print('num_edges in max_comp of G=', len(G_conn.edges))




def Generate_corrupt_graph(path_load, path_save,
                           G_original=None,
                           delimiter=',',
                           noise_nodes=200,
                           parameter=0.1,
                           noise_type='-ER_walk'):
    ### noise_type = 'ER' (Erdos-Renyi), 'WS' (Watts-Strongatz), 'BA' (Barabasi-Albert), '-ER_edges' (Delete ER edges)
    ### '-ER_walk' (Delete ER edges along k-walk MCMC sampler (not globally))

    if G_original is not None:
        G = nx.Graph()
        edgelist = [e for e in G_original.get_edges()]
        for e in G_original.get_edges():
            G.add_edge(e[0], e[1], weight=1)
        edgelist = list(G.edges())

    else:
        # load original graph from path
        edgelist = np.genfromtxt(path_load, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G = nx.Graph()
        for e in edgelist:
            G.add_edge(e[0], e[1], weight=1)

    node_list = [v for v in G.nodes]
    # randomly sample nodes from original graph
    sample = np.random.choice([n for n in G.nodes], noise_nodes, replace=False)
    d = {n: sample[n] for n in range(0, noise_nodes)}  ### set operation
    G_noise = nx.Graph()
    # Generate corrupt network
    # SW = nx.watts_strogatz_graph(70,50,0.05)


    if noise_type in ['ER', 'ER_edges', 'WS', 'BA', 'lattice']:
        # edges are added
        edges_added = []
        if noise_type in ['ER', '+ER']:
            #G_noise = nx.erdos_renyi_graph(noise_nodes, parameter)
            G_noise = nx.dense_gnm_random_graph(noise_nodes, int(parameter*len(edgelist)))  # num edges for second input
        elif noise_type == 'ER_edges':
            G_noise = nx.gnm_random_graph(noise_nodes, int(parameter*len(edgelist)))

        elif noise_type == 'WS':
            # number of edges in WS(n, d, p) = (d/2) * n, want this to be "parameter".
            G_noise = nx.watts_strogatz_graph(noise_nodes, 2 * parameter // noise_nodes, 0.3)
            print('!!! # edges in WS', len(G_noise.edges))
            # G_noise = nx.watts_strogatz_graph(100, 50, 0.4)
        elif noise_type == 'BA':
            G_noise = nx.barabasi_albert_graph(noise_nodes, parameter)
        elif noise_type == 'lattice':
            G_noise = nx.generators.lattice.grid_2d_graph(noise_nodes, noise_nodes)


        edges = list(G_noise.edges)
        # print(len(edges))

        # Overlay corrupt edges onto graph
        for edge in edges:

            # for lattice graphs
            # ------------------------------------
            # edge1 = edge[0][0] * 40 + edge[0][1]
            # edge2 = edge[1][0] * 40 + edge[1][1]

            # if not (G.has_edge(d[edge1], d[edge2])):
            #    edges_added.append([d[edge1], d[edge2]])
            #    G.add_edge(d[edge1], d[edge2], weight=1)
            # ---------------------------------------
            if not (G.has_edge(d[edge[0]], d[edge[1]])):
                edges_added.append([d[edge[0]], d[edge[1]]])
                G.add_edge(d[edge[0]], d[edge[1]], weight=1)

        edgelist_permuted = np.random.permutation(G.edges)
        G_new = nx.Graph()
        # G_new.add_nodes(G.nodes)
        for e in edgelist_permuted:
            G_new.add_edge(e[0], e[1], weight=1)

        edges_changed = edges_added
        print('edges added = {}'.format(len(edges_added)))


    elif noise_type in ['-BA']:
        # preferntially delete edges
        print('!!!@@ noise type', noise_type)
        G_new = nx.Graph()
        G_new = nx.Graph(G.edges)
        G_new.subgraph(sorted(nx.connected_components(G_new), key=len, reverse=True)[0])

        nodes = list(G.nodes())
        print('nodes', len(nodes))
        #deg_seq = [d for n, d in G.degree()]

        #dist = deg_seq/np.sum(deg_seq)
        #print('dist', len(dist))
        deleted_edges = []

        mst = nx.minimum_spanning_edges(G, data=False)
        mst_edgelist = list(mst)  # MST edges
        print('!!! len(mst_edgelist)', len(mst_edgelist))

        abundant_edges = [e for e in G.edges if e not in mst_edgelist]

        if noise_type == '-BA':
            preferential_seq = [G.degree(e[0])*G.degree(e[1]) for e in abundant_edges]
            dist = preferential_seq/np.sum(preferential_seq)


        print('parameter = {}'.format(parameter))
        with tqdm.tqdm(total=len(deleted_edges)) as pbar:
            while len(deleted_edges) < len(abundant_edges)*parameter:

                idx = np.arange(len(abundant_edges))

                if noise_type == '-BA':
                    i = np.random.choice(idx, p=dist)
                else:
                    i = np.random.choice(idx)
                u,v = abundant_edges[i]

                if [u,v] in deleted_edges or [v,u] in deleted_edges:
                    continue
                if not G_new.has_edge(u,v):
                    continue
                G_new.remove_edge(u,v)
                deleted_edges.append([u,v])
                pbar.update(1)

        print('!! edges deleted = {} / target={}'.format(len(deleted_edges), len(abundant_edges)*parameter))
        edges_changed = deleted_edges


    elif noise_type in ['-ER_edges', '-ER']:
        ### take a minimum spanning tree and add back edges except ones to be deleted
        noise_sign = "deleted"
        full_edge_list = G_original.get_edges()
        G_diminished = nx.Graph(full_edge_list)
        Gc = max(nx.connected_components(G_diminished), key=len)
        G_diminished = G_diminished.subgraph(Gc).copy()
        full_edge_list = [e for e in G_diminished.edges]

        #print('!!! G_diminished.nodes', len(G_diminished.nodes()))
        #print('!!! G_diminished.edges', len(G_diminished.edges()))

        G_new = nx.Graph()
        mst = nx.minimum_spanning_edges(G_diminished, data=False)
        mst_edgelist = list(mst)  # MST edges
        print('!!! len(mst_edgelist)', len(mst_edgelist))
        G_new = nx.Graph(mst_edgelist)
        G_new.add_nodes_from(G_diminished.nodes())

        edges_non_mst = []
        for edge in full_edge_list:
            if edge not in mst_edgelist:
                edges_non_mst.append(edge)
        print('!!! len(edges_non_mst)', len(edges_non_mst))

        idx_array = np.random.choice(range(len(edges_non_mst)), int(len(edges_non_mst)*parameter), replace=False)
        #edges_deleted = [full_edge_list[i] for i in idx_array]
        edges_deleted = [edges_non_mst[i] for i in idx_array]
        print('!!! len(edges_deleted)', len(edges_deleted))
        for i in range(len(edges_non_mst)):
            if i not in idx_array:
                edge = edges_non_mst[i]
                G_new.add_edge(edge[0], edge[1])

        print('!! edges deleted = {} / target={}'.format(len(edges_deleted), int(len(edges_non_mst)*parameter)))
        edges_changed = edges_deleted

    elif noise_type in ['-ER_walk']:
        ### take a minimum spanning tree and add back edges except ones to be deleted
        full_edge_list = G_original.get_edges()
        G_nn = nn.NNetwork()
        G_nn.add_edges(full_edge_list)

        noise_sign = "deleted"

        G_new = nx.Graph(full_edge_list)
        Gc = max(nx.connected_components(G_new), key=len)
        G_new = G_new.subgraph(Gc).copy()
        full_edge_list = [e for e in G_new.edges]

        edges_deleted = []

        k=100
        #target_n_delete = int(len(full_edge_list)*parameter)
        target_n_delete = 100
        while len(edges_deleted) < target_n_delete:

            X, embs = G_nn.get_patches(k=k, emb=None, sample_size=1, skip_folded_hom=False)
            s = 0
            while (s < X.shape[1]) and (len(edges_deleted) < target_n_delete):
                emb = embs[s]
                H = G_nn.subgraph(nodelist=emb)
                H_edges = H.get_edges()
                H_chain_edges = [[emb[i],emb[i+1]] for i in np.arange(k-1)]
                H_offchain_edges = [edge for edge in H_edges if edge not in H_chain_edges]

                #print('!!! len(H_offchain_edges)', len(H_offchain_edges))

                idx_array = np.random.choice(range(len(H_offchain_edges)), int(len(H_offchain_edges)*parameter), replace=False)

                for j in list(idx_array):
                    edge = H_offchain_edges[j]
                    if G_new.has_edge(edge[0], edge[1]):
                        G_new.remove_edge(edge[0], edge[1])
                        edges_deleted.append(edge)

                s += 1
        print('!! edges deleted = {} / target={}'.format(len(edges_deleted), int(len(full_edge_list)*parameter)))

        edges_changed = edges_deleted

    nx.write_edgelist(G_new, path_save, data=False, delimiter=',')
    print('num undirected edges right after corruption:', len(nx.Graph(G_new.edges).edges))

    ### Output network as NNetwork class
    G_out = nn.NNetwork()
    G_out.load_add_edges(path_save, delimiter=',', increment_weights=False, use_genfromtxt=True)

    return G_out, edges_changed


def permute_nodes(path_load, path_save):
    # Randomly permute node labels of a given graph
    edgelist = np.genfromtxt(path_load, delimiter=',', dtype=int)
    edgelist = edgelist.tolist()
    G = nx.Graph()
    for e in edgelist:
        G.add_edge(e[0], e[1], weight=1)

    node_list = [v for v in G.nodes]
    permutation = np.random.permutation(np.arange(1, len(node_list) + 1))

    G_new = nx.Graph()
    for e in edgelist:
        G_new.add_edge(permutation[e[0] - 1], permutation[e[1] - 1], weight=1)
        # print('new edge', permutation[e[0]-1], permutation[e[1]-1])

    nx.write_edgelist(G, path_save, data=False, delimiter=',')

    return G_new




def rocch(fpr0, tpr0):
    """
    @author: Dr. Fayyaz Minhas (http://faculty.pieas.edu.pk/fayyaz/)
    Construct the convex hull of a Receiver Operating Characteristic (ROC) curve
        Input:
            fpr0: List of false positive rates in range [0,1]
            tpr0: List of true positive rates in range [0,1]
                fpr0,tpr0 can be obtained from sklearn.metrics.roc_curve or
                    any other packages such as pyml
        Return:
            F: list of false positive rates on the convex hull
            T: list of true positive rates on the convex hull
                plt.plot(F,T) will plot the convex hull
            auc: Area under the ROC Convex hull
    """
    fpr = np.array([0] + list(fpr0) + [1.0, 1, 0])
    tpr = np.array([0] + list(tpr0) + [1.0, 0, 0])
    hull = ConvexHull(np.vstack((fpr, tpr)).T)
    vert = hull.vertices
    vert = vert[np.argsort(fpr[vert])]
    F = [0]
    T = [0]
    for v in vert:
        ft = (fpr[v], tpr[v])
        if ft == (0, 0) or ft == (1, 1) or ft == (1, 0):
            continue
        F += [fpr[v]]
        T += [tpr[v]]
    F += [1]
    T += [1]
    auc = np.trapz(T, F)
    return F, T, auc


def calculate_AUC(x, y):
    total = 0
    for i in range(len(x) - 1):
        total += np.abs((y[i] + y[i + 1]) * (x[i] - x[i + 1]) / 2)

    return total

def compute_ROC_AUC(G_original=None,
                    recons_wtd_edgelist=[],
                    path_original=None,
                    path_corrupt=None,
                    G_corrupted=None,
                    delimiter_original=',',
                    delimiter_corrupt=',',
                    save_file_name=None,
                    test_edges=None,
                    save_folder=None,
                    flip_TF=False,
                    subtractive_noise=False,
                    show_PR=False,
                    test_size=0.5):
    ### set motif arm lengths
    # recon = "Caltech_permuted_recon_corrupt_ER_30k_wtd.txt"
    # full = "Caltech36_corrupted_ER.txt"
    # original = "Caltech36_node_permuted.txt"
    # if subtractive_noise: Edges are deleted from the original, and the classification is among the nonedges in the corrupted ntwk
    # Reveal true labels for "test_size" fraction of all edges, determine optimal threshold, and apply it to the rest
    print('!!!! ROC computation begins..')
    result_dict = {}
    ### read in networks
    G_recons =nn.NNetwork()
    # G_recons.load_add_edges(path_recons, increment_weights=False, delimiter=delimiter_recons, use_genfromtxt=True)
    G_recons.add_edges(recons_wtd_edgelist, increment_weights=False)
    print('num edges in G_recons', len(G_recons.get_wtd_edgelist()))
    # print('wtd edges in G_recons', G_recons.get_wtd_edgelist())

    if G_original is None:
        G_original =nn.NNetwork()
        G_original.load_add_edges(path_original, increment_weights=False, delimiter=delimiter_corrupt,
                                      use_genfromtxt=True)
    edgelist_original = G_original.get_edges()
    # else G_corrupted is given as a Wtd_netwk form
    print('num edges in G_original', len(G_original.get_wtd_edgelist()))

    if G_corrupted is None:
        G_corrupted =nn.NNetwork()
        G_corrupted.load_add_edges(path_corrupt, increment_weights=False, delimiter=delimiter_corrupt,
                                       use_genfromtxt=True)
    edgelist_full = G_corrupted.get_edges()
    # else G_corrupted is given as a Wtd_netwk form
    print('num edges in G_corrupted', len(G_corrupted.get_wtd_edgelist()))

    # G_original =nn.NNetwork()
    # G_original.load_add_edges(path_original, increment_weights=False, delimiter=delimiter_original,
    #                              use_genfromtxt=True)

    y_true = []
    y_pred = []

    print("~~~!!subtractive_noise", subtractive_noise)

    if not subtractive_noise:
        """
        # classify all observed edges
        for e in tqdm.tqdm(edgelist_full):
            pred = G_recons.get_edge_weight(e[0], e[1])
            if pred is None: # means MCMC sampler never visited this edge e --- low probability of being true edge
                pred = 0

            if not flip_TF:
                y_pred.append(pred)
            else:
                y_pred.append(1 - pred)

            if G_original.has_edge(e[0], e[1]):
                y_true.append(1)
            else:
                y_true.append(0)
        """
        # classify all edges in G_recons (assuming it only contains test edges)
        edges = G_recons.get_edges()
        for e in tqdm.tqdm(edges):
            pred = G_recons.get_edge_weight(e[0], e[1])
            if not flip_TF:
                y_pred.append(pred)
            else:
                y_pred.append(1 - pred)

            if G_original.has_edge(e[0], e[1]):
                y_true.append(1)
            else:
                y_true.append(0)

    else:
        # classify all observed non-edges O(n^2)
        # usually use subsample of true non-edges
        # G_recons contains only those subsampled edges

        edges = G_recons.get_edges()
        for e in tqdm.tqdm(edges):
            pred = G_recons.get_edge_weight(e[0], e[1])
            if not flip_TF:
                y_pred.append(pred)
            else:
                y_pred.append(1 - pred)

            if G_original.has_edge(e[0], e[1]):
                y_true.append(1)
            else:
                y_true.append(0)



        """
        V = G_original.nodes()

        for i in np.arange(len(V)):
            for j in np.arange(i, len(V)):
                if not G_corrupted.has_edge(V[i], V[j]) and np.random.rand(1) < 0.1:
                    pred = G_recons.get_edge_weight(V[i], V[j])
                    if pred == None:
                        y_pred.append(0)
                    else:
                        if not flip_TF:
                            y_pred.append(pred)
                        else:
                            y_pred.append(1 - pred)

                    if G_original.has_edge(V[i], V[j]):
                        y_true.append(1)
                    else:
                        y_true.append(0)


        """
        """
                if test_edges is not None:
                    V = G_original.nodes()

                    for i in np.arange(len(V)):
                        for j in np.arange(i, len(V)):
                            if not G_corrupted.has_edge(V[i], V[j]) and np.random.rand(1) < 0.1:
                                pred = G_recons.get_edge_weight(V[i], V[j])
                                if pred == None:
                                    y_pred.append(0)
                                else:
                                    if not flip_TF:
                                        y_pred.append(pred)
                                    else:
                                        y_pred.append(1 - pred)

                                if G_original.has_edge(V[i], V[j]):
                                    y_true.append(1)
                                else:
                                    y_true.append(0)
                else:
                    print("~~~!!! test edges ROC")
                    for e in tqdm.tqdm(test_edges):
                        pred = G_recons.get_edge_weight(e[0], e[1])
                        if pred is None:
                            pred = 0
                        y_pred.append(pred)

                        if G_original.has_edge(e[0], e[1]):
                            y_true.append(1)
                        else:
                            y_true.append(0)
        """

    X_train, X_test, y_train, y_test = train_test_split(y_pred, y_true, test_size=test_size)

    # ROC and PR plots
    # fpr, tpr, thresholds = roc_curve(y_train, X_train) # y_true, y_score
    # precision, recall, thresholds_PR = precision_recall_curve(y_train, X_train) # y_true, y_score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred) # y_true, y_score
    precision, recall, thresholds_PR = precision_recall_curve(y_true, y_pred) # y_true, y_score


    F, T, ac = rocch(fpr, tpr)
    auc_PR = calculate_AUC(recall, precision)

    print("AUC with convex hull: ", ac)
    print("AUC without convex hull: ", calculate_AUC(fpr, tpr))
    print("PR Accuracy without convex hull: ", auc_PR)

    result_dict = compute_binary_classification_metrics(y_pred, y_true, test_size=test_size)

    result_dict.update({'AUC': ac})

    print("Training AUC: ", np.mean(result_dict.get('Training_AUC')))
    print("Training threshold: ", np.mean(result_dict.get('Training_threshold')))
    mcm = np.mean(result_dict.get('Confusion_mx'), axis=0)
    sum_of_rows = mcm.sum(axis=1)
    mcm /= sum_of_rows[:, np.newaxis]
    print("Test confusion_mx: \n", mcm)
    print("Test accuracy: ", np.mean(result_dict.get('Accuracy')))
    print("Test precision: ", np.mean(result_dict.get('Precision')))
    print("Test recall: ", np.mean(result_dict.get('Recall')))
    print("Test F-score: ", np.mean(result_dict.get('F_score')))

    return result_dict

def compute_ROC_AUC_colored(G_original=None,
                    recons_colored_edges={},
                    path_original=None,
                    path_corrupt=None,
                    G_corrupted=None,
                    delimiter_original=',',
                    delimiter_corrupt=',',
                    save_file_name=None,
                    save_folder=None,
                    flip_TF=False,
                    subtractive_noise=False,
                    show_PR=False,
                    test_size=0.5):

    ### set motif arm lengths
    # recon = "Caltech_permuted_recon_corrupt_ER_30k_wtd.txt"
    # full = "Caltech36_corrupted_ER.txt"
    # original = "Caltech36_node_permuted.txt"
    # if subtractive_noise: Edges are deleted from the original, and the classification is among the nonedges in the corrupted ntwk
    # Reveal true labels for "test_size" fraction of all edges, determine optimal threshold, and apply it to the rest
    print('!!!! ROC computation begins..')
    ### read in networks
    result_dict_list = []
    G_recons = nn.NNetwork()
    # G_recons.load_add_edges(path_recons, increment_weights=False, delimiter=delimiter_recons, use_genfromtxt=True)
    G_recons.add_colored_edges(recons_colored_edges)
    print('num colored edges in G_recons', len(G_recons.get_wtd_edgelist()))
    edge0 = eval(list(recons_colored_edges.keys())[0])
    n_layers = len(G_recons.get_colored_edge_weight(edge0[0], edge0[1]))

    if G_original is None:
        G_original =nn.NNetwork()
        G_original.load_add_edges(path_original, increment_weights=False, delimiter=delimiter_corrupt,
                                      use_genfromtxt=True)
    edgelist_original = G_original.get_edges()
    # else G_corrupted is given as a Wtd_netwk form
    print('num edges in G_original', len(G_original.get_wtd_edgelist()))

    if G_corrupted is None:
        G_corrupted = nn.NNetwork()
        G_corrupted.load_add_edges(path_corrupt, increment_weights=False, delimiter=delimiter_corrupt,
                                       use_genfromtxt=True)
    edgelist_full = G_corrupted.get_edges()
    # else G_corrupted is given as a Wtd_netwk form
    print('num edges in G_corrupted', len(G_corrupted.get_wtd_edgelist()))

    # G_original =nn.NNetwork()
    # G_original.load_add_edges(path_original, increment_weights=False, delimiter=delimiter_original,
    #                              use_genfromtxt=True)

    y_true = []

    j = 0
    for a in range(n_layers):
        y_true = []
        y_pred = []
        result_dict = {}
        if not subtractive_noise:
            for edge in edgelist_full:
                pred = G_recons.get_colored_edge_weight(edge[0], edge[1])

                if pred == None:
                    y_pred.append(0)
                else:
                    if not flip_TF:
                        y_pred.append(pred[a])
                    else:
                        y_pred.append(1 - pred[a])

                if edge in edgelist_original:
                    y_true.append(1)
                else:
                    y_true.append(0)
        else:
            V = G_original.nodes()
            print('!! len(G_original.edges())', G_original.edges[0])
            print('!! len(G_corrupted.edges())', G_corrupted.edges[0])

            for i in np.arange(len(V)):
                for j in np.arange(i, len(V)):
                    if not G_corrupted.has_edge(V[i], V[j]) and np.random.rand(1) < 0.1:
                        # for classifying nonedges ~O(n^2), subsample 10% and compute classification metrics
                        pred = G_recons.get_colored_edge_weight(V[i], V[j])
                        if pred == None:
                            y_pred.append(0)
                        else:
                            if not flip_TF:
                                y_pred.append(pred[a])
                            else:
                                y_pred.append(1 - pred[a])

                        if G_original.has_edge(V[i], V[j]):
                            y_true.append(1)
                        else:
                            y_true.append(0)

        X_train, X_test, y_train, y_test = train_test_split(y_pred, y_true, test_size=test_size)

        # ROC and PR plots
        fpr, tpr, thresholds = roc_curve(y_train, X_train) # y_true, y_score
        precision, recall, thresholds_PR = precision_recall_curve(y_train, X_train) # y_true, y_score
        #fpr, tpr, thresholds = roc_curve(y_true, y_pred) # y_true, y_score
        #precision, recall, thresholds_PR = precision_recall_curve(y_true, y_pred) # y_true, y_score


        F, T, ac = rocch(fpr, tpr)
        auc_PR = calculate_AUC(recall, precision)

        print("AUC with convex hull: ", ac)
        print("AUC without convex hull: ", calculate_AUC(fpr, tpr))
        print("PR Accuracy without convex hull: ", auc_PR)

        result_dict = compute_binary_classification_metrics(y_pred, y_true, test_size=test_size)

        print("Training AUC: ", np.mean(result_dict.get('Training_AUC')))
        print("Training threshold: ", np.mean(result_dict.get('Training_threshold')))
        mcm = np.mean(result_dict.get('Confusion_mx'), axis=0)
        sum_of_rows = mcm.sum(axis=1)
        mcm /= sum_of_rows[:, np.newaxis]
        print("Test confusion_mx: \n", mcm)
        print("Test accuracy: ", np.mean(result_dict.get('Accuracy')))
        print("Test precision: ", np.mean(result_dict.get('Precision')))
        print("Test recall: ", np.mean(result_dict.get('Recall')))
        print("Test F-score: ", np.mean(result_dict.get('F_score')))

        result_dict_list.append(result_dict)

    return result_dict_list


def compute_binary_classification_metrics(x, y, test_size=0.2, num_splits=10):
    tn_list = []
    tp_list = []
    fn_list = []
    fp_list = []
    auc_list = []
    thresholds_list = []
    confusion_mx_list = []
    for i in np.arange(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        fpr, tpr, thresholds = roc_curve(y_train, X_train, pos_label=None)
        mythre = thresholds[np.argmax(tpr - fpr)]  # classification threshold from train set
        myauc = calculate_AUC(fpr, tpr)
        auc_list.append(myauc)
        thresholds_list.append(mythre)
        #print('*** Train AUC', myauc)
        #print('*** Train Threshold', mythre)


        fpr1, tpr1, thresholds1 = roc_curve(y_test, X_test, pos_label=None)
        myauc1 = calculate_AUC(fpr1, tpr1)
        #
        #print('*** Test AUC', myauc1)

        # Compute classification statistics
        Y_pred = np.asarray(X_test.copy())
        Y_pred[np.asarray(X_test) < mythre] = 0
        Y_pred[np.asarray(X_test) >= mythre] = 1

        mcm = confusion_matrix(y_test, Y_pred)
        tn = mcm[0, 0]
        tp = mcm[1, 1]
        fn = mcm[1, 0]
        fp = mcm[0, 1]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        #print('*** Test Accuracy', accuracy)

        tn_list.append(tn)
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)
        confusion_mx_list.append(mcm)



    auc = np.asarray(auc_list)
    thresholds = np.asarray(thresholds_list)
    tn = np.asarray(tn_list)
    tp = np.asarray(tp_list)
    fn = np.asarray(fn_list)
    fp = np.asarray(fp_list)
    confusion_mx = np.asarray(fp_list)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    misclassification = 1 - accuracy
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fall_out = fp / (fp + tn)
    miss_rate = fn / (fn + tp)
    F_score = 2 * precision * recall / ( precision + recall )

    #precision_list, recall_list, thresholds_PR = precision_recall_curve(y_train, X_train) # y_true, y_score
    #auc_PR = calculate_AUC(recall_list, precision_list)

    # Save results
    result_dict = {}
    result_dict.update({'Y_test_full': y})
    result_dict.update({'Y_pred_full': x})
    result_dict.update({'Y_test': y_test})
    result_dict.update({'Y_pred': Y_pred})
    result_dict.update({'Training_AUC': auc})
    result_dict.update({'Training_threshold': thresholds})
    result_dict.update({'Accuracy': accuracy})
    result_dict.update({'Misclassification': misclassification})
    result_dict.update({'Precision': precision})
    result_dict.update({'Recall': recall})
    result_dict.update({'Sensitivity': sensitivity})
    result_dict.update({'Specificity': specificity})
    result_dict.update({'F_score': F_score})
    result_dict.update({'Fall_out': fall_out})
    result_dict.update({'Miss_rate': miss_rate})
    result_dict.update({'Confusion_mx': confusion_mx_list})
    #result_dict.update({'AUC_PR': auc_PR})
    #result_dict.update({'Thresholds_PR': thresholds_PR})

    return result_dict



def display_denoising_stats_list_plot(denoising_dict,
                                 W_list_filtered,
                                 ROC_dict_list,
                                 save_path,
                                 title,
                                 At = None,
                                 fig_size = [15,10.5]):

    Ndict_wspace = 0.05
    Ndict_hspace = 0.05
    #fig_size = [5,15]

    fig = plt.figure(figsize=fig_size, constrained_layout=False)
    outer_grid = gridspec.GridSpec(nrows=2, ncols=1+len(W_list_filtered), wspace=0.1, hspace=0.4)

    ### first row = Network Dictionary Graphs, MCMC path sampler visit histogram
    W = W_list_filtered[0]
    n_components = W.shape[1]
    k = int(np.sqrt(W.shape[0]))

    rows = np.round(np.sqrt(n_components))
    rows = rows.astype(int)

    if rows ** 2 == n_components:
        cols = rows
    else:
        cols = rows + 1

    if At is not None:
        importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
        # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)
    else:
        idx = np.arange(W.shape[1])


    # visit count histograms
    visit_counts_false = denoising_dict.get("visit_counts_false")
    visit_counts_true = denoising_dict.get("visit_counts_true")

    ax = plt.Subplot(fig, outer_grid[0,0])
    ax.set_title(r"$\#$ $k$-walk MCMC" +  "\n sampler visits")
    #ax.set_title('$\\textsc{\\texttt{BA}}_{2}$')

    ax.axis('off')
    fig.add_subplot(ax)

    inner_grid = outer_grid[0,0].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
    ax = fig.add_subplot(inner_grid[0, 0])
    ax.hist(visit_counts_true, bins='auto', alpha=0.3, label='true edges')
    ax.hist(visit_counts_false, bins='auto', alpha=0.3, label='false edges')
    ax.legend()

        # on-chain lengths upon visits
    avg_dist_on_chain_false = denoising_dict.get("avg_dist_on_chain_false")
    avg_dist_on_chain_true = denoising_dict.get("avg_dist_on_chain_true")

    ax = plt.Subplot(fig, outer_grid[1,0])
    ax.set_title(r"Mean dist. along $k$-walks")
    ax.axis('off')
    fig.add_subplot(ax)

    inner_grid = outer_grid[1,0].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
    ax = fig.add_subplot(inner_grid[0, 0])


    ax.hist(avg_dist_on_chain_true, bins='auto', alpha=0.3, label='true edges')
    ax.hist(avg_dist_on_chain_false, bins='auto', alpha=0.3, label='false edges')
    ax.legend()

    ### subsequent rows = dictionary-specific results:
    ### network dictionary, recons weights, ROC & PR curves, confusion matrix
    recons_weights_false_full = denoising_dict.get("recons_colored_weights_false")
    recons_weights_true_full = denoising_dict.get("recons_colored_weights_true")

    ax = plt.Subplot(fig, outer_grid[0,1:len(W_list_filtered)+1])
    #ax.set_title("Edge weights in reconstruction", y=1.3)
    ax.axis('off')
    fig.add_subplot(ax)

    for t in np.arange(len(W_list_filtered)):
        # network dictionary


        # network dictionary in matrix form with filter applied
        ax = plt.Subplot(fig, outer_grid[1, t+1])
        if t == 0:
            ax.set_title("Latent motifs \n (learned from " + r"$G$)")
        elif t == 1:
            ax.set_title("Latent motifs \n (on-chain thinning: 0.5)")
        elif t == 2:
            ax.set_title("Latent motifs \n (on-chain thinning: 0)")
        elif t == 3:
            ax.set_title("Random latent motifs")
        elif t == 4:
            ax.set_title("Random latent motifs \n (on-chain thinning: 0)")

        ax.axis('off')
        fig.add_subplot(ax)

        inner_grid = outer_grid[1, t+1].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
        #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)
        W = W_list_filtered[t]
        for i in range(rows * cols):
            a = i // cols
            b = i % cols
            ax = fig.add_subplot(inner_grid[a, b])
            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.set_xticks([])
            ax.set_yticks([])



        ### reconstructed edge weights
        recons_weights_false = [x[t] for x in recons_weights_false_full]
        recons_weights_true = [x[t] for x in recons_weights_true_full]

        inner_grid = outer_grid[0,t+1].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
        ax = fig.add_subplot(inner_grid[0, 0])
        ax.hist(recons_weights_true, bins='auto', alpha=0.5, label='true edges')
        ax.hist(recons_weights_false, bins='auto', alpha=0.5, label='false edges')
        ax.set_xlim(xmax=1)
        ax.legend()
        ax.set_yticks([])


        ### ROC and PR plots
        y_full = ROC_dict_list[t].get("Y_test_full")
        x_full = ROC_dict_list[t].get("Y_pred_full")

        fpr, tpr, thresholds = roc_curve(y_full, x_full) # y_true, y_score
        precision, recall, thresholds_PR = precision_recall_curve(y_full, x_full) # y_true, y_score

        F, T, ac = rocch(fpr, tpr)
        auc_PR = calculate_AUC(recall, precision)

        ACC_list = ROC_dict_list[t].get("Accuracy") # list of confusion matrix for various train/test split
        ACC = np.mean(ACC_list, axis=0)

        PRE_list = ROC_dict_list[t].get("Precision") # list of confusion matrix for various train/test split
        PRE = np.mean(PRE_list, axis=0)

        subtitle_acc = r"ACC $\approx$ {:.3f}".format(np.round(ACC, 3))+"\n"+ r"PRE $\approx$ {:.3f}".format(np.round(PRE, 3)) + "\n" + r"AUC $\approx$ {:.3f}".format(np.round(ac, 3))
        ax.set_title(subtitle_acc)

    #plt.suptitle(title, fontsize=15)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
    fig.savefig(save_path, bbox_inches='tight')




def MACC(G, k_list=range(2,20)):
    # Matrix of averaged clustering coefficient
    MACC_list = []
    density_dict = {}
    k_hop_edge_density = {}
    for j in trange(len(k_list)):
        A_list = []
        k = k_list[j]
        n_off_chain_edges=[]
        for r in np.arange(10):
            X, embs = G.get_patches(k=k, sample_size=100, skip_folded_hom=False)
            k = int(np.sqrt(X.shape[0]))
            for i in np.arange(X.shape[1]):
                A = (X[:,i].reshape(-1, k))
                #A1 = A - np.eye(k, k=1) - np.eye(k, k=-1)
                #n_off_chain_edges.append(np.sum(A1))
                #print('np.sum(A1)', np.sum(A1))
                A_list.append(A)
                #print('A_list.shape', np.asarray(A_list).shape)
        A_list = np.asarray(A_list)
        A_mean = np.mean(A_list, axis=0)
        MACC_list.append(A_mean)

    return MACC_list


def k_hop_edge_density(MACC_list):
    # compute MACC_list = MACC(G, k_list=range(2,20))

    k_hop_edge_density_dict = {}
    for i in np.arange(len(MACC_list)):
        A = MACC_list[i]
        k = A.shape[0]
        k_hop_edge_density_dict.update({str(k): A[0,-1]})

    return k_hop_edge_density_dict


def mean_off_chain_density(MACC_list):
    # compute MACC_list = MACC(G, k_list=range(2,20))
    mean_off_chain_density_dict = {}
    for i in np.arange(len(MACC_list)):
        A = MACC_list[i]
        k = A.shape[0]
        x = np.sum(A - np.eye(k, k=-1) - np.eye(k, k=1))/(2*k)
        mean_off_chain_density_dict.update({str(k): x})
    return mean_off_chain_density_dict
