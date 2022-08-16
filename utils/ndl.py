# from utils.onmf.onmf import Online_NMF
from utils.onmf import Online_NMF
#from utils.NNetwork import NNetwork
from NNetwork.NNetwork import NNetwork
import numpy as np
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import networkx as nx
import os
import psutil
from tqdm import trange
import matplotlib.gridspec as gridspec
from time import sleep
import sys
import random
import pandas as pd
import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import ast
import gc

DEBUG = False


class Network_Reconstructor():
    def __init__(self,
                 G,
                 n_components=100,
                 MCMC_iterations=500,
                 sub_iterations=100,
                 loc_avg_depth=1,
                 sample_size=1000,
                 batch_size=10,
                 k1=0,
                 k2=21,
                 patches_file='',
                 alpha=None,
                 is_glauber_dict=True,
                 sampling_alg='pivot', # 'pivot' or  'idla' or 'pivot_inj'
                 Pivot_exact_MH_rule=False,
                 ONMF_subsample=True,
                 if_wtd_network=False,
                 if_tensor_ntwk=False,
                 omit_folded_edges=False):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.G = G  ### Full netowrk -- could have positive or negagtive edge weights (as a NNetwork or NNetwork class)
        if if_tensor_ntwk:
            self.G.set_clrd_edges_signs()
            ### Each edge with weight w is assigned with tensor weight [+(w), -(w)] stored in the field colored_edge_weight

        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.loc_avg_depth = loc_avg_depth
        self.k1 = k1
        self.k2 = k2
        self.patches_file = patches_file
        self.if_tensor_ntwk = if_tensor_ntwk  # if True, input data is a 3d array
        self.omit_folded_edges = omit_folded_edges  # if True, get induced k by k patch without off-chain edges appearing
        ### due to folding of the underlying motif (e.g., completely folded k-chain --> no checkerboard pattern)
        self.W = np.random.rand((k1 + k2 + 1) ** 2, n_components)
        if if_tensor_ntwk:
            self.W = np.random.rand(G.color_dim * (k1 + k2 + 1) ** 2, n_components)

        print('n_components', n_components)
        print('sample_size', sample_size)
        self.code = np.zeros(shape=(n_components, sample_size))
        self.code_recons = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.sampling_alg = sampling_alg ### subgraph sampling algorithm: ['glauber', 'pivot', 'idla']
        self.Pivot_exact_MH_rule = Pivot_exact_MH_rule
        self.edges_deleted = []
        self.ONMF_subsample = ONMF_subsample
        self.At = np.random.rand(n_components, n_components)
        self.result_dict = {}
        self.if_wtd_network = if_wtd_network

    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def path_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        if k1 == 0 or k2 == 0:
            k3 = max(k1, k2)
            A = np.eye(k3 + 1, k=1, dtype=int)
        else:
            A = np.eye(k1 + k2 + 1, k=1, dtype=int)
            A[k1, k1 + 1] = 0
            A[0, k1 + 1] = 1
        return A

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        # (!!! Also finds self-loop)
        return min(j)

    def tree_sample(self, B, x=None):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node
        if x is None:
            x = np.random.choice(self.G.nodes())
        N = self.G
        k = np.shape(B)[0]
        emb = np.array([x], dtype='<U32')  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N.num_nodes(), size=(1, k - 1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                nbs_j = np.asarray(list(N.neighbors(emb[j])))
                if len(nbs_j) > 0:
                    y = np.random.choice(nbs_j)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))
        # print('emb', emb)
        return emb

    def glauber_gen_update(self, B, emb):
        N = self.G
        k = np.shape(B)[0]

        if k == 1:

            emb[0] = self.walk(emb[0], 1)
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            time_a = time()
            cmn_nbs = N.nodes(is_set=True)
            time_1 = time()
            time_neighbor = 0

            if not self.if_wtd_network:
                for r in nbh_in:
                    time_neighb = time()
                    nbs_r = N.neighbors(emb[r])
                    end_neighb = time()
                    time_neighbor += end_neighb - time_neighb
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                for r in nbh_out:
                    nbs_r = N.neighbors(emb[r])
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                cmn_nbs = list(cmn_nbs)
                if len(cmn_nbs) > 0:
                    y = np.random.choice(np.asarray(cmn_nbs))
                    emb[j] = y
                else:
                    emb[j] = np.random.choice(N.nodes())
                    print('Glauber move rejected')  # Won't happen once a valid embedding is established

            else:  ### Now need to use edge weights for Glauber chain update as well
                # build distribution for resampling emb[j] and resample emb[j]
                cmn_nbs = [i for i in N.nodes()]
                for r in nbh_in:
                    # print('emb[r]:',emb[r])
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
                for r in nbh_out:
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))

                if len(cmn_nbs) > 0:

                    ### Compute distribution on cmn_nbs
                    dist = np.ones(len(cmn_nbs))
                    for v in np.arange(len(cmn_nbs)):
                        for r in nbh_in:
                            dist[v] = dist[v] * abs(N.get_edge_weight(emb[r], cmn_nbs[v]))
                        for r in nbh_out:
                            dist[v] = dist[v] * abs(N.get_edge_weight(cmn_nbs[v], emb[r]))
                            ### As of now (05/15/2020) NNetwork class has weighted edges without orientation,
                            ### so there is no distinction between in- and out-neighbors
                            ### Use abs since edge weights could be negative
                    dist = dist / np.sum(dist)
                    # idx = np.random.choice(np.arange(len(cmn_nbs)), p=dist)
                    ### 7/25/2020: If just use np.random.choice(cmn_nbs, p=dist), then it somehow only selects first six-digit and causes key erros
                    idx = np.random.choice(np.arange(len(cmn_nbs)), p=dist)

                    emb[j] = cmn_nbs[idx]
                    # if len(emb[j]) == 7:
                    #    print('y len 7')

                else:
                    emb[j] = np.random.choice(np.asarray([i for i in self.G.nodes]))
                    print('Glauber move rejected')  # Won't happen once valid embedding is established

        return emb

    def Pivot_update(self, emb, if_inj = False):
        # G = underlying simple graph
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule
        # if_inj = injective sampling of subsequent nodes -- repeat RW until k distinct nodes are collected

        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)  # new location of the pivot
        B = self.path_adj(0, len(emb)-1)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding
        if not if_inj:
            emb_new = self.tree_sample(B, x0)  # new path embedding
        else:
            H = None
            while H is None:
                H = self.G.k_node_IDLA_subgraph(k=len(emb), center=x0)
                x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)
            emb_new = H.nodes()
        return emb_new

    def RW_update(self, x, Pivot_exact_MH_rule=False):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform
        # Pivot_exact_MH_rule = True --> RW is updated so that the Pivot chain is sampled from the exact conditional distribution
        # otherwise the pivot of the Pivot chain performs random walk with uniform distribution as its stationary distribution

        N = self.G
        length = self.k1 + self.k2  # number of edges in the chain motif
        nbs_x = np.asarray(list(N.neighbors(x)))  # array of neighbors of x in G

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # choose a uniform element in nbs_x
            # x ---> y move generated
            # Use MH-rule to accept or reject the move
            # stationary distribution = Uniform(nodes)
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            nbs_y = np.asarray(list(N.neighbors(y)))
            prob_accept = min(1, len(nbs_x) / len(nbs_y))

            if Pivot_exact_MH_rule:
                a = N.count_k_step_walks(y, radius=length)
                b = N.count_k_step_walks(x, radius=length)
                print('!!!! MHrule a', a)
                print('!!!! MHrule b', b)

                prob_accept = min(1, a * len(nbs_x) / b * len(nbs_y))

            if np.random.rand() > prob_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray(N.nodes()))
        return y

    def update_hom_get_meso_patch(self,
                                  B,
                                  emb,
                                  iterations=1,
                                  sampling_alg='glauber', # 'pivot' or  'idla' or 'pivot_inj'
                                  verbose=0,
                                  omit_folded_edges=False):
        # computes a mesoscale patch of the input network G using Glauber chain to evolve embedding of B in to G
        # also update the homomorphism once
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif
        start = time()

        N = self.G
        emb2 = emb
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding

        hom_mx2 = np.zeros([k, k])
        if self.if_tensor_ntwk:
            hom_mx2 = np.zeros([k, k, N.color_dim])

        nofolding_ind_mx = np.zeros([k, k])

        for i in range(iterations):
            start_iter = time()
            if sampling_alg == 'glauber':
                emb2 = self.glauber_gen_update(B, emb2)
            elif sampling_alg == 'pivot':
                emb2 = self.Pivot_update(emb2, if_inj = False)
            elif sampling_alg == 'idla':
                # IDLA sampling: centered around a uniformly chosen node, sample
                # a k-node subgraph (no repeated nodes)
                H = None
                while H is None:
                    H = N.k_node_IDLA_subgraph(k=k, center=None)
                emb2 = H.nodes()
            elif sampling_alg == 'pivot_inj':
                emb2 = self.Pivot_update(emb2, if_inj = True)

            end_update = time()

            # start = time.time()

            ### Form induced graph = homomorphic copy of the motif given by emb2 (may have < k nodes)
            ### e.g., a k-chain can be embedded onto a K2, then H = K2.
            H = NNetwork()
            for q in np.arange(k):
                for r in np.arange(k):
                    edge = [emb2[q], emb2[r]]  ### "edge" may repeat for distinct pairs of [q,r]
                    if B[q, r] > 0:  ### means [q,r] is an edge in the motif with adj mx B
                        H.add_edge(edge=edge, weight=1)

            if not self.if_tensor_ntwk:
                # full adjacency matrix or induced weight matrix over the path motif
                a2 = np.zeros([k, k])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if not self.if_wtd_network or N.has_edge(emb2[q], emb2[r]) == 0:
                            if not omit_folded_edges:
                                a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
                                nofolding_ind_mx[q, r] = 1

                        else:
                            if not omit_folded_edges:
                                a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])
                                nofolding_ind_mx[q, r] = 1

                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            else:  # full induced weight tensor over the path motif (each slice of colored edge gives a weight matrix)
                a2 = np.zeros([k, k, N.color_dim])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if N.has_edge(emb2[q], emb2[r]) == 0:
                            if not omit_folded_edges:
                                a2[q, r, :] = np.zeros(N.color_dim)
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r, :] = np.zeros(N.color_dim)
                                nofolding_ind_mx[q, r] = 1
                        else:
                            if not omit_folded_edges:
                                a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
                            elif not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
                                a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
                                nofolding_ind_mx[q, r] = 1
                                # print('np.sum(a2[q, r, :])', np.sum(a2[q, r, :]))
                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            if (verbose):
                print([int(i) for i in emb2])

        if omit_folded_edges:
            return hom_mx2, emb2, nofolding_ind_mx
        else:
            return hom_mx2, emb2

    def get_patches(self, B, emb,
                    skip_folded_hom=False,
                    sample_size=1,
                    omit_folded_edges=False,
                    sampling_alg='pivot'):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        X = np.zeros((k ** 2, 1))
        if self.if_tensor_ntwk:
            X = np.zeros((k ** 2, self.G.color_dim, 1))


        num_hom_sampled = 0
        X = []
        count = 0
        while (num_hom_sampled < sample_size) and (count < 10000 * sample_size):
            meso_patch = self.update_hom_get_meso_patch(B, emb,
                                                        iterations=1,
                                                        sampling_alg = sampling_alg,
                                                        omit_folded_edges=omit_folded_edges)
            Y = meso_patch[0]
            emb = meso_patch[1]
            if not (skip_folded_hom and len(set(emb))<k):
                # skip adding the sampled patch if the nodes are not distinct
                if not self.if_tensor_ntwk:
                    Y = Y.reshape(k ** 2, -1)
                else:
                    Y = Y.reshape(k ** 2, self.G.color_dim, -1)
                X.append(Y)
                    # now X.shape = (k**2, sample_size) or (k**2, color_dim, sample_size)
                num_hom_sampled += 1
            count += 1
        if len(X) > 0:
            X = np.asarray(X)[..., 0].T
        else:
            X = None

        if not omit_folded_edges:
            return X, emb
        else:
            return X, emb, meso_patch[2]  # last output is the nofolding indicator mx


    def get_single_patch(self, B, emb, omit_folded_edges=False, sampling_alg='pivot', skip_folded_hom=False):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        num_hom_sampled = 0
        while num_hom_sampled < 1:
            meso_patch = self.update_hom_get_meso_patch(B,
                                                        emb, iterations=1,
                                                        sampling_alg=sampling_alg,
                                                        omit_folded_edges=omit_folded_edges)

            Y = meso_patch[0]
            emb = meso_patch[1]
            if not (skip_folded_hom and len(set(emb))<k):
                num_hom_sampled += 1

        if not self.if_tensor_ntwk:
            X = Y.reshape(k ** 2, -1)
        else:
            X = Y.reshape(k ** 2, self.G.get_edge_color_dim(), -1)

        if not omit_folded_edges:
            return X, emb
        else:
            return X, emb, meso_patch[2]  # last output is the nofolding indicator mx

    def glauber_walk(self, x0, length, iters=1, verbose=0):

        N = self.G
        B = self.path_adj(0, length)
        # x0 = 2
        # x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        k = B.shape[0]

        emb, _ = self.update_hom_get_meso_patch(B, emb, iterations=iters, verbose=0)

        return [int(i) for i in emb]

    def walk(self, node, iters=10):
        for i in range(iters):
            node = np.random.choice(self.G.neighbors(node))

        return node

    def compute_overlap_stat(self, k, iterations=100):
        G = self.G
        B = self.path_adj(0, k-1)
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb_p = self.tree_sample(B, x0)
        emb_g = self.tree_sample(B, x0)
        node_overlap_list_pivot = []
        node_overlap_list_glauber = []
        for t in trange(iterations):
            X, emb_p = self.get_patches(B, emb_p, sampling_alg = 'pivot', skip_folded_hom=False)
            X, emb_g = self.get_patches(B, emb_g, sampling_alg = 'glauber', skip_folded_hom=False)
            node_overlap_list_pivot.append(len(list(set(emb_p))))
            node_overlap_list_glauber.append(len(list(set(emb_g))))
        return node_overlap_list_pivot, node_overlap_list_glauber

    def train_dict(self,
                   jump_every=20,
                   update_dict_save=True, show_error=False,
                   skip_folded_hom=False,
                   iterations=None):
        # emb = initial embedding of the motif into the network
        print('training dictionaries from patches...')
        print('skip_folded_hom=', skip_folded_hom)
        '''
        Trains dictionary based on patches.
        '''

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        node_overlap_list = []
        W = self.W
        print('W.shape', W.shape)
        errors = []
        code = self.code
        iter0 = iterations
        if iterations is None:
            iter0 = self.MCMC_iterations
        for t in trange(iter0):
            X, emb = self.get_patches(B, emb, sample_size = self.sample_size,
                                      skip_folded_hom=skip_folded_hom,
                                      sampling_alg = self.sampling_alg)
            # print('X.shape', X.shape)  ## X.shape = (k**2, sample_size)
            node_overlap_list.append(len(list(set(emb)))) # number of distinct nodes in the image of emb
            # print('# of distinct nodes sampled : ', len(list(set(emb))))
            ### resample the embedding for faster mixing of the Glauber chain
            if X is not None: # None means no injective homomorphism is sampled
                if t % jump_every == 0:
                    x0 = np.random.choice(np.asarray([i for i in G.vertices]))
                    emb = self.tree_sample(B, x0)
                    print('homomorphism resampled')

                if not self.if_tensor_ntwk:
                    X = np.expand_dims(X, axis=1)  ### X.shape = (k**2, 1, sample_size)
                if t == 0:
                    self.ntf = Online_NMF(X, self.n_components,
                                          iterations=self.sub_iterations,
                                          batch_size=self.batch_size,
                                          alpha=self.alpha,
                                          mode=2,
                                          learn_joint_dict=True,
                                          subsample=self.ONMF_subsample)  # max number of possible patches
                    self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                    self.H = code
                else:
                    self.ntf = Online_NMF(X, self.n_components,
                                          iterations=self.sub_iterations,
                                          batch_size=self.batch_size,
                                          ini_dict=self.W,
                                          ini_A=self.At,
                                          ini_B=self.Bt,
                                          ini_C=self.Ct,
                                          alpha=self.alpha,
                                          history=self.ntf.history,
                                          subsample=self.ONMF_subsample,
                                          mode=2,
                                          learn_joint_dict=True)
                    # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                    # for "iterations" iterations
                    self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                    H_temp = np.zeros(shape=[code.shape[0], np.maximum(code.shape[1], self.H.shape[1])])
                    H_temp[:,:code.shape[1]] += code
                    H_temp[:,:self.H.shape[1]] += self.H
                    code = H_temp
                    if show_error:
                        error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                        print('error', error)
                        errors.append(error)
                #  progress status
                # if 100 * t / self.MCMC_iterations % 1 == 0:
                #    print(t / self.MCMC_iterations * 100)
                # print('Current iteration %i out of %i' % (t, self.MCMC_iterations))
        self.code = code
        print('!!!number of distinct nodes in homomorhpisms : avg {} std {:.3f}'.format(np.mean(node_overlap_list), np.std(node_overlap_list)))
        if update_dict_save:
            self.result_dict.update({'Dictionary learned': self.W})
            self.result_dict.update({'Motif size': self.k2 + 1})
            self.result_dict.update({'Code learned': self.code})
            self.result_dict.update({'Code COV learned': self.At})
        # print(self.W)
        return self.W

    def display_dict(self,
                     title,
                     save_path = None,
                     make_first_atom_2by2=False,
                     show_importance=False):
        #  display learned dictionary
        W = self.W
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1

        if save_path is None:
            save_path = "Network_dictionary/test"

        # cols=3
        # rows=6
        k = self.k1 + self.k2 + 1

        ### Use the code covariance matrix At to compute importance
        importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
        # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        if make_first_atom_2by2:
            ### Make gridspec
            fig = plt.figure(figsize=(3, 6), constrained_layout=False)
            gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.1, hspace=0.1)

            for i in range(rows * cols - 3):
                if i == 0:
                    ax = fig.add_subplot(gs1[:2, :2])
                elif i < 2 * cols - 3:  ### first two rows of the dictionary plot
                    if i < cols - 1:
                        ax = fig.add_subplot(gs1[0, i + 1])
                    else:
                        ax = fig.add_subplot(gs1[1, i - (cols - 1) + 2])
                else:
                    i1 = i + 3
                    a = i1 // cols
                    b = i1 % cols
                    ax = fig.add_subplot(gs1[a, b])

                ax.imshow(self.W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                ax.set_xticks([])
                ax.set_yticks([])

            plt.suptitle(title)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            fig.savefig(save_path)
            # plt.show()

        else:
            if not self.if_tensor_ntwk:
                figsize = (5, 5)
                if show_importance:
                    figsize = (5, 6)

                fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                                        subplot_kw={'xticks': [], 'yticks': []})

                k = self.k1 + self.k2 + 1  # number of nodes in the motif F
                for ax, j in zip(axs.flat, range(n_components)):
                    ax.imshow(self.W.T[idx[j]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    # use gray_r to make black = 1 and white = 0

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
                fig.savefig(save_path)
            else:
                W = W.reshape(k ** 2, self.G.color_dim, self.n_components)
                for c in range(self.G.color_dim):
                    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                            subplot_kw={'xticks': [], 'yticks': []})

                    for ax, j in zip(axs.flat, range(n_components)):
                        ax.imshow(W[:, c, :].T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        # use gray_r to make black = 1 and white = 0
                        if show_importance:
                            ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                            ax.xaxis.set_label_coords(0.5,
                                                      -0.05)  # adjust location of importance appearing beneath patches

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
                fig.savefig(save_path + '_color_' + str(c))

        # plt.show()

    def display_graphs(self,
                     title,
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
            a = i // cols
            b = i % rows

            Ndict_wspace = 0.05
            Ndict_hspace = 0.05

            # display graphs
            inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)

            # get rid of duplicate nodes
            A = X[:,idx[i]]
            A = X[:,idx[i]].reshape(int(np.sqrt(X.shape[0])), -1)
            H = NNetwork()
            H.read_adj(A)
            nodes = list(set(embs[idx[i]]))
            H_sub = H.subgraph(nodes)
            A_sub = H_sub.get_adjacency_matrix()

            # read in as a nx graph for plotting
            G1 = nx.from_numpy_matrix(A_sub)
            ax = fig.add_subplot(inner_grid[0, 0])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            weights = [5*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=10, ax=ax, width=weights, label='Graph')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')


    def display_dict_graph(self,
                             title,
                             save_path,
                             grid_shape=None,
                             fig_size=[10,10],
                             W = None,
                             show_importance=False):

        if W is None:
            W = self.W
        n_components = W.shape[1]
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


        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        # make outer gridspec

        if show_importance:
            importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
            # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            idx = np.arange(W.shape[1])

        outer_grid = gridspec.GridSpec(nrows=rows, ncols=cols, wspace=0.02, hspace=0.05)

        # make nested gridspecs
        for i in range(rows * cols):
            a = i // cols
            b = i % rows

            Ndict_wspace = 0.05
            Ndict_hspace = 0.05

            # display graphs
            inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
            G1 = nx.from_numpy_matrix(W[:,idx[i]].reshape(int(np.sqrt(W.shape[0])),-1))
            ax = fig.add_subplot(inner_grid[0, 0])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            weights = [5*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=10, ax=ax, width=weights, label='Graph')
            if show_importance:
                ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')


    def display_dict_and_graph(self,
                             title,
                             save_path,
                             grid_shape=None,
                             fig_size=[10,10],
                             W = None,
                             show_importance=False):
        if W is None:
            W = self.W
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

        if show_importance:
            importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
            # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            idx = np.arange(W.shape[1])

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
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
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

        plt.suptitle(title, fontsize=25)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')


    def display_dict_gridspec(self,
                              title,
                              save_filename,
                              save_folder):
        ### Initial setup
        W = self.W
        n_components = W.shape[1]
        nrows = np.round(np.sqrt(n_components))
        nrows = nrows.astype(int)
        if nrows ** 2 == n_components:
            ncols = nrows
        else:
            ncols = nrows + 1

        if save_folder is None:
            save_folder = "Network_dictionary"

        k = int(np.sqrt(W.shape[0]))

        importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        ### Make gridspec
        fig1 = plt.figure(figsize=(6, 6), constrained_layout=False)
        gs1 = fig1.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.05, hspace=0.05)

        for i in range(nrows * ncols - 3):
            if i == 0:
                ax = fig1.add_subplot(gs1[:2, :2])
            elif i < 2 * ncols - 3:  ### first two rows of the dictionary plot
                if i < ncols - 1:
                    ax = fig1.add_subplot(gs1[0, i + 1])
                else:
                    ax = fig1.add_subplot(gs1[1, i - (ncols - 1) + 2])
            else:
                i1 = i + 3
                a = i1 // ncols
                b = i1 % ncols
                ax = fig1.add_subplot(gs1[a, b])

            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title)
        fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig1.savefig(save_folder + '/' + save_filename)
        plt.show()



# instantiating the decorator
    def reconstruct_network(self,
                            recons_iter=100,
                            if_save_history=False,
                            ckpt_epoch=10000,  # not used if None
                            jump_every=None,
                            omit_chain_edges=False,  ### Turn this on for denoising
                            omit_folded_edges=True,
                            patch_masking_ratio = 0,
                            W = None,
                            edge_threshold=None, ### if a neumeric, threshold the weighted reconstructed edges at the end
                            edges_added=None,
                            if_keep_visit_statistics=False,
                            if_save_wtd_reconstruction=True,
                            print_patches=False,
                            # sampling_alg = 'pivot', # or 'glauber' or 'idla'
                            save_path = None,
                            use_refreshing_random_dict=False,
                            show_memory_states=False):
        print('reconstructing given network...')
        print('!!! sampling alg', self.sampling_alg)
        '''
        NNetwork version of the reconstruction algorithm (custom Neighborhood Network package for scalable Glauber chain sampling)
        Using large "ckpt_epoch" improves reconstruction accuracy but uses more memory
        edges_added = list of false edges added to the original network to be denoised by reconstruction
        use_refreshing_random_dict = If true, resample the dictionary matrix W randomly every iteration
        '''
        if save_path is None:
            save_path = "Network_dictionary/test"

        if W is None:
            W = self.W

        G = self.G
        self.G_recons = NNetwork()
        self.G_recons_baseline = NNetwork()  ## reconstruct only the edges used by the Glauber chain
        self.G_overlap_count = NNetwork()
        self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
        self.G_recons.add_nodes(nodes=[v for v in G.vertices])
        self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        self.result_dict.update({'NDR iterations': recons_iter})
        self.result_dict.update({'omit_chain_edges for NDR': omit_chain_edges})

        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        emb_history = emb.copy()
        code_history = np.zeros(2 * self.n_components)

        ### Extend the learned dictionary for the flip-symmetry of the path embedding
        atom_size, num_atoms = W.shape
        W_ext = np.empty((atom_size, 2 * num_atoms))
        W_ext[:, 0:num_atoms] = W[:, 0:num_atoms]
        W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(W[:, 0:num_atoms])

        W_ext_reduced = W_ext

        ### Set up paths and folders
        path_recons = save_path + "_wtd_edgelist_recons" + '.pickle'
        path_recons_baseline = save_path + "_baseline_recons" + '.pickle'
        path_overlap_count = save_path + "_overlap_count" + '.pickle'
        path_overlap_colored_count = save_path + "_overlap_colored_count" + '.pickle'

        t0 = time()

        ### omit all chain edges from the extended dictionary
        W_ext_reduced = patch_masking(W_ext,
                                      k = self.k1 + self.k2 + 1,  # size of the network patch
                                      chain_edge_masking=1-int(omit_chain_edges))

        has_saved_checkpoint = False
        for t in trange(recons_iter):
            meso_patch = self.get_patches(B, emb,
                                          omit_folded_edges=omit_folded_edges,
                                          sampling_alg=self.sampling_alg)
            patch = meso_patch[0]
            emb = meso_patch[1]
            if (jump_every is not None) and (t % jump_every == 0):
                x0 = np.random.choice(np.asarray([i for i in G.vertices]))
                emb = self.tree_sample(B, x0)
                # print('homomorphism resampled')

            # meso_patch[2] = nofolding_indicator matrix

            # print('patch', patch.reshape(k, k))


            ### omit all chain edges from the patches matrix
            patch_reduced = patch_masking(patch,
                                          k = self.k1 + self.k2 + 1,  # size of the network patch
                                          chain_edge_masking=1-int(omit_chain_edges))
            ratio_edges_removed = (np.sum(patch) - np.sum(patch_reduced))/np.sum(patch)

            if use_refreshing_random_dict:
                #self.W = np.random.rand(self.W.shape[0],self.W.shape[1])
                W1 = np.ones(shape=W.shape)
                atom_size, num_atoms = W1.shape
                W_ext = np.empty((atom_size, 2 * num_atoms))
                W_ext[:, 0:num_atoms] = W1[:, 0:num_atoms]
                W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(W1[:, 0:num_atoms])
                W_ext_reduced = W_ext
                W = W1
                #print('!!! avg entry', np.sum(self.W)/np.prod(self.W.shape))

            coder = SparseCoder(dictionary=W_ext_reduced.T,  ### Use extended dictioanry
                                transform_n_nonzero_coefs=None,
                                transform_alpha=self.alpha,
                                transform_algorithm='lasso_lars',
                                positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch_reduced.T)
            #code = np.random.rand(W_ext_reduced.shape[1], patch_reduced.shape[1]).T
            #code = np.ones(shape=[W_ext_reduced.shape[1], patch_reduced.shape[1]]).T

            if print_patches and edges_added is not None:
                P = patch.reshape(k, k)
                for x in itertools.product(np.arange(k), repeat=2):
                    a = emb[x[0]]
                    b = emb[x[1]]
                    if [a, b] in edges_added:
                        P[x[0], x[1]] = -P[x[0], x[1]]
                print('!!!!! Current sampled patch:\n', P.astype(int))

            if if_save_history:
                emb_history = np.vstack((emb_history, emb))
                code_history = np.vstack((code_history, code))

            patch_recons = np.dot(W_ext, code.T).T
            patch_recons = patch_recons.reshape(k, k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                # edge = [str(a), str(b)] ### Use this when nodes are saved as strings, e.g., '154' as in DNA networks
                edge = [a, b]  ### Use this when nodes are saved as integers, e.g., 154 as in FB networks

                # print('!!!! meso_patch[2]', meso_patch[2])

                if not (omit_folded_edges and meso_patch[2][x[0], x[1]] == 0):
                    #    print('!!!!!!!!! reconstruction masked')
                    #    print('!!!!! meso_patch[2]', meso_patch[2])
                    if self.G_overlap_count.has_edge(a, b) == True:
                        # print(G_recons.edges)
                        # print('ind', ind)
                        j = self.G_overlap_count.get_edge_weight(a, b)
                    else:
                        j = 0


                    if if_keep_visit_statistics:
                        if not self.G_overlap_count.has_colored_edge(edge[0], edge[1]):
                            self.G_overlap_count.add_colored_edge([edge[0], edge[1], [np.abs(x[0] - x[1])]])
                            ### np.abs(x[0]-x_{1}) = distance on the chain motif
                        else:
                            colored_edge_weight = float(self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0])
                            self.G_overlap_count.add_colored_edge(
                                [edge[0], edge[1], [(j * colored_edge_weight + np.abs(x[0] - x[1])) / (j + 1)]])

                    if self.G_recons.has_edge(a, b) == True:
                        new_edge_weight = (j * self.G_recons.get_edge_weight(a, b) + patch_recons[x[0], x[1]]) / (j + 1)
                    else:
                        new_edge_weight = patch_recons[x[0], x[1]]

                    # if j>0 and new_edge_weight==0:
                    #    print('!!!overlap count %i, new edge weight %.2f' % (j, new_edge_weight))

                    if np.abs(x[0] - x[1]) == 1:
                        # print('baseline edge added!!')
                        self.G_recons_baseline.add_edge(edge, weight=1, increment_weights=False)

                    if not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):

                        self.G_overlap_count.add_edge(edge, weight=j + 1, increment_weights=False)
                        #if omit_chain_edges and np.abs(x[0] - x[1]) == 1:
                        #    print('!!!!! Chain edges counted')

                        if new_edge_weight > 0:
                            self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                            ### Add the same edge to the baseline reconstruction
                            ### if x[0] and x[1] are adjacent in the chain motif

            # print progress status and memory use
            if t % 50000 == 0:
                self.result_dict.update({'homomorphisms_history': emb_history})
                self.result_dict.update({'code_history': code_history})
                self.compute_recons_accuracy(G_recons = self.G_recons_baseline)
                # print('iteration %i out of %i' % (t, recons_iter))
                # self.G_recons.get_min_max_edge_weights()
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                print('memory use:', memoryUse)

                gc.collect()




            # refreshing memory at checkpoints
            if (ckpt_epoch is not None) and (t % ckpt_epoch == 0) and (t>0):
                # print out current memory usage
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                print('memory use:', memoryUse)

                # print('num edges in G_count', len(self.G_overlap_count.get_edges()))

                ### Load and combine with the saved edges and reconstruction counts
                if has_saved_checkpoint:
                    print('!!! merging with previous checkpoint...')
                    self.G_recons_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True,
                                                              is_pickle=True)

                    G_overlap_count_new = NNetwork()
                    G_overlap_count_new.add_wtd_edges(edges=self.G_overlap_count.wtd_edges)

                    G_overlap_count_old = NNetwork()
                    G_overlap_count_old.load_add_wtd_edges(path=path_overlap_count, increment_weights=False,
                                                           is_pickle=True)


                    if if_keep_visit_statistics:
                        G_overlap_count_new.add_colored_edges(colored_edges=self.G_overlap_count.colored_edges)
                        G_overlap_count_old.load_add_colored_edges(path=path_overlap_colored_count)

                        wtd_edges = G_overlap_count_old.wtd_edges
                        colored_edges = G_overlap_count_old.colored_edges
                        #print('!!! check if wtd and colored edges are the same:', wtd_edges.keys() == colored_edges.keys())

                    G_recons_new = NNetwork()
                    G_recons_new.add_wtd_edges(edges=self.G_recons.wtd_edges)
                    # G_recons_new.get_min_max_edge_weights()

                    self.G_recons = NNetwork()
                    self.G_recons.load_add_wtd_edges(path=path_recons, increment_weights=False,
                                                     is_pickle=True)
                    # self.G_recons.get_min_max_edge_weights()

                    for edge in G_recons_new.wtd_edges.keys():
                        edge = eval(edge)
                        count_old = G_overlap_count_old.get_edge_weight(edge[0], edge[1])
                        count_new = G_overlap_count_new.get_edge_weight(edge[0], edge[1])

                        old_edge_weight = self.G_recons.get_edge_weight(edge[0], edge[1])
                        new_edge_weight = G_recons_new.get_edge_weight(edge[0], edge[1])

                        if count_old is None:
                            count_old = 0
                        if count_new is None:
                            count_old = 0
                        if old_edge_weight is None:
                            old_edge_weight = 0
                        if new_edge_weight is None:
                            new_edge_weight = 0

                        if count_old + count_new > 0:
                            new_edge_weight = (count_old / (count_old + count_new)) * old_edge_weight + (
                                        count_new / (count_old + count_new)) * new_edge_weight

                        self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                        G_overlap_count_old.add_edge(edge=edge, weight=count_new, increment_weights=True)

                        if if_keep_visit_statistics:
                            old_edge_color = G_overlap_count_old.get_colored_edge_weight(edge[0], edge[1])
                            new_edge_color = G_overlap_count_new.get_colored_edge_weight(edge[0], edge[1])
                            #old_edge_color = old_edge_color)
                            #new_edge_color = int(new_edge_color)

                            if old_edge_color is not None and new_edge_color is not None:
                                new_edge_color = (count_old / (count_old + count_new)) * old_edge_color[0] + (
                                            count_new / (count_old + count_new)) * new_edge_color[0]
                            elif old_edge_color is not None: # then count_new is None
                                new_edge_color = old_edge_color[0]
                            elif new_edge_color is not None: # then count_old is None
                                new_edge_color = new_edge_color[0]

                            G_overlap_count_old.add_colored_edge(colored_edge=[edge[0], edge[1], [new_edge_color]])

                    # print('!!!! max new weight', max(new_edge_wts))
                    # self.G_recons = G_recons_old
                    self.G_overlap_count = G_overlap_count_old
                    del G_overlap_count_old
                    del G_overlap_count_new
                    del G_recons_new

                print('!!! num edges in G_recons', len(self.G_recons.get_edges()))
                print('!!! num edges in G_overlap_count', len(self.G_overlap_count.get_edges()))
                if if_keep_visit_statistics:
                    print('!!! num colored edges in G_overlap_count', len(self.G_overlap_count.colored_edges))
                print('!!! num edges in G_recons_baseline', len(self.G_recons_baseline.get_edges()))
                # self.G_recons.get_min_max_edge_weights()
                # self.G_overlap_count.get_min_max_edge_weights()

                ### Save current graphs
                self.G_recons.save_wtd_edges(path_recons)
                self.G_overlap_count.save_wtd_edges(path_overlap_count)
                if if_keep_visit_statistics:
                    self.G_overlap_count.save_colored_edges(path_overlap_colored_count)
                self.G_recons_baseline.save_wtd_edges(path_recons_baseline)

                has_saved_checkpoint = True

                print('sys.getsizeof(self.G_recons)!', sys.getsizeof(self.G_recons))
                print('sys.getsizeof(self.G_recons_baseline)!', sys.getsizeof(self.G_recons_baseline))
                print('sys.getsizeof(self.G_overlap_count)!', sys.getsizeof(self.G_overlap_count))

                ### Clear up the edges of the current graphs
                self.G_recons = NNetwork()
                self.G_recons_baseline = NNetwork()
                self.G_overlap_count = NNetwork()
                self.G_recons.add_nodes(nodes=[v for v in G.vertices])
                self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
                self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])
                G_overlap_count_new = NNetwork()
                G_overlap_count_old = NNetwork()
                G_recons_new = NNetwork()

                print('!!! num edges in G_recons after refreshing', len(self.G_recons.get_edges()))
                print('!!! num edges in G_overlap_count after refreshing', len(self.G_overlap_count.get_edges()))
                if if_keep_visit_statistics:
                    print('!!! num colored edges in G_overlap_count after refreshing', len(self.G_overlap_count.colored_edges))
                print('!!! num edges in G_recons_baseline after refreshing', len(self.G_recons_baseline.get_edges()))

                gc.collect()

                if show_memory_states:
                    # For memory usage debugging purposes
                    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                         key= lambda x: -x[1])[:10]:
                        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

                    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
                        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

        # Load up saved reconstruction files
        if (ckpt_epoch is not None) and (recons_iter>ckpt_epoch):
            self.G_recons = NNetwork()
            self.G_recons.load_add_wtd_edges(path=path_recons, increment_weights=False, is_pickle=True)
            self.G_overlap_count = NNetwork()
            self.G_overlap_count.load_add_wtd_edges(path=path_overlap_count, increment_weights=False, is_pickle=True)
            if if_keep_visit_statistics:
                self.G_overlap_count.load_add_colored_edges(path=path_overlap_colored_count)

        ### Save weigthed reconstruction into full results dictionary
        if if_save_wtd_reconstruction:
            self.result_dict.update({'Edges in weighted reconstruction': self.G_recons.wtd_edges})

        if if_keep_visit_statistics:
            denoising_dict = compute_denoising_stats(G=self.G,
                                                      edges_added=edges_added,
                                                      G_recons=self.G_recons,
                                                      G_overlap_count=self.G_overlap_count,
                                                      save_path=save_path)
            self.result_dict.update({'denoising_dict': denoising_dict})

        #print('ratio of masked edges in sampled patches: (mean, std)=({:.4f}, {:.4f})'.format(np.mean(chain_edge_stat), np.std(chain_edge_stat)))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        # self.compute_recons_accuracy(G_recons = self.G_recons_baseline)
        print('Number of edges in the baseline recons. : ', len(self.G_recons_baseline.get_edges()))
        #print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})

        # Return reconstruction
        if edge_threshold is not None:
            ### Finalize the simplified reconstruction graph
            G_recons_final = self.G_recons.threshold2simple(threshold=edge_threshold)
            return G_recons_final
        else:
            return self.G_recons


    def reconstruct_network_list(self,
                                W_list = None,
                                recons_iter=100,
                                if_save_history=False,
                                ckpt_epoch=10000,  # not used if None
                                jump_every=None,
                                masking_params_list = [0], # 1 for no masking, 0 for full masking of chain edges
                                edges_added=None,
                                test_edges=None,
                                if_keep_visit_statistics=False,
                                skip_folded_hom = False,
                                save_path = None):

        print('reconstructing given network...')
        print('!!! sampling alg = ', self.sampling_alg)
        print('!!! masking_params_list = ', masking_params_list)
        '''
        NNetwork version of the reconstruction algorithm (custom Neighborhood Network package for scalable Glauber chain sampling)
        edges_added = list of false edges added to the original network to be denoised by reconstruction
        Uses a list of network dictionary to creat multiple versions of reconstructed network at the same time
        Stores as colored edges (edges weighted by a list)
        If test_edges is given, initialize around these edges to make sure reconstruction contains these edges.
        '''
        if save_path is None:
            save_path = "Network_dictionary/test"

        G = self.G
        self.G_recons = NNetwork()
        self.G_recons_baseline = NNetwork()  ## reconstruct only the edges used by the Glauber chain
        self.G_overlap_count = NNetwork()
        self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
        self.G_recons.add_nodes(nodes=[v for v in G.vertices])
        self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        self.result_dict.update({'NDR iterations': recons_iter})
        self.result_dict.update({'masking_params_list': masking_params_list})

        if test_edges is not None:
            test_nodes0 = [e[0] for e in test_edges]
            test_nodes1 = [e[1] for e in test_edges]
            test_nodes = list(set(test_nodes0 + test_nodes1))

        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        if test_edges is None:
            x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        else:
            x0 = np.random.choice(np.asarray([i for i in test_nodes]))

        emb = self.tree_sample(B, x0)
        emb_history = emb.copy()
        code_history = np.zeros(2 * self.n_components)

        ### Set up paths and folders
        path_recons = save_path + "_wtd_edgelist_recons" + '.pickle'
        path_recons_baseline = save_path + "_baseline_recons" + '.pickle'
        path_overlap_count = save_path + "_overlap_count" + '.pickle'
        path_overlap_colored_count = save_path + "_overlap_colored_count" + '.pickle'


        ### Extend the learned dictionary for the flip-symmetry of the path embedding
        W_list_filtered = []
        W_list_ext = []
        for i in range(len(W_list)):
            atom_size, num_atoms = W_list[i].shape # num_atoms may change in i # there was an error of taking num_stoms only from the first dictionaryx
            W = W_list[i]
            print('masking_params_list applied:', masking_params_list[i])
            W1 = patch_masking(W,
                              k = self.k1 + self.k2 + 1,  # size of the network patch
                              chain_edge_masking=masking_params_list[i])
            W_list_filtered.append(W1)

            W_ext = np.empty((atom_size, 2 * num_atoms))
            W_ext[:, 0:num_atoms] = W1[:, 0:num_atoms]
            W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(W1[:, 0:num_atoms])
            W_list_ext.append(W_ext)

        self.result_dict.update({'W_list_filtered': W_list_filtered})

        t0 = time()
        chain_edge_stat = []

        has_saved_checkpoint = False
        for t in trange(recons_iter):
            meso_patch = self.get_patches(B, emb,
                                          sampling_alg=self.sampling_alg,
                                          skip_folded_hom=skip_folded_hom)

            patch = meso_patch[0]
            emb = meso_patch[1]
            if (jump_every is not None) and (t % jump_every == 0):
                if test_edges is None:
                    x0 = np.random.choice(np.asarray([i for i in G.vertices]))
                else:
                    x0 = np.random.choice(np.asarray([i for i in test_nodes]))

                emb = self.tree_sample(B, x0)
                print('homomorphism resampled during reconstruction')

            ### local reconstruction
            patch_recons_list=[]
            for i in range(len(W_list_ext)):
                ### omit all chain edges from the patches matrix

                patch_reduced = patch_masking(patch,
                                              k = self.k1 + self.k2 + 1,  # size of the network patch
                                              chain_edge_masking=masking_params_list[i])

                coder = SparseCoder(dictionary=W_list_ext[i].T,  ### Use extended dictioanry
                                    transform_n_nonzero_coefs=None,
                                    transform_alpha=self.alpha,
                                    transform_algorithm='lasso_lars',
                                    positive_code=True)
                code = coder.transform(patch_reduced.T)

                if if_save_history:
                    emb_history = np.vstack((emb_history, emb))
                    code_history = np.vstack((code_history, code))

                patch_recons = np.dot(W_list_ext[i], code.T).T
                patch_recons = patch_recons.reshape(k, k)
                patch_recons_list.append(patch_recons)


            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                # edge = [str(a), str(b)] ### Use this when nodes are saved as strings, e.g., '154' as in DNA networks
                edge = [a, b]  ### Use this when nodes are saved as integers, e.g., 154 as in FB networks

                # print('!!!! meso_patch[2]', meso_patch[2])

                #if not (omit_folded_edges and meso_patch[2][x[0], x[1]] == 0):
                #    print('!!!!!!!!! reconstruction masked')
                #    print('!!!!! meso_patch[2]', meso_patch[2])
                if self.G_overlap_count.has_edge(a, b) == True:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = self.G_overlap_count.get_edge_weight(a, b)
                else:
                    j = 0


                if if_keep_visit_statistics:
                    if not self.G_overlap_count.has_colored_edge(edge[0], edge[1]):
                        self.G_overlap_count.add_colored_edge([edge[0], edge[1], [np.abs(x[0] - x[1])]])
                        ### np.abs(x[0]-x_{1}) = distance on the chain motif
                    else:
                        colored_edge_weight = float(self.G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0])
                        self.G_overlap_count.add_colored_edge(
                            [edge[0], edge[1], [(j * colored_edge_weight + np.abs(x[0] - x[1])) / (j + 1)]])

                ### update colored edge weights in G_recons
                for i in range(len(patch_recons_list)):
                    patch_recons = patch_recons_list[i]
                    old_edge_weight = 0
                    if self.G_recons.has_colored_edge(a, b) == True:
                        old_edge_weight = self.G_recons.get_colored_edge_weight(a, b)[i]

                    new_edge_weight = (j * old_edge_weight + patch_recons[x[0], x[1]]) / (j + 1)

                    if (new_edge_weight > 0):

                        colored_edge_weight = self.G_recons.get_colored_edge_weight(edge[0], edge[1]) # a list
                        if colored_edge_weight is None:
                            colored_edge_weight = [0] * len(patch_recons_list)
                            colored_edge_weight[i] = new_edge_weight
                        else:
                            colored_edge_weight[i] = new_edge_weight

                        if not (masking_params_list[i]<1 and np.abs(x[0] - x[1]) == 1):
                            self.G_recons.add_colored_edge(colored_edge=[edge[0], edge[1], colored_edge_weight])

                # if j>0 and new_edge_weight==0:
                #    print('!!!overlap count %i, new edge weight %.2f' % (j, new_edge_weight))

                if np.abs(x[0] - x[1]) == 1:
                    # print('baseline edge added!!')
                    self.G_recons_baseline.add_edge(edge, weight=1, increment_weights=False)

                #if not (omit_chain_edges and np.abs(x[0] - x[1]) == 1):

                self.G_overlap_count.add_edge(edge, weight=j + 1, increment_weights=False)
                #if omit_chain_edges and np.abs(x[0] - x[1]) == 1:
                #    print('!!!!! Chain edges counted')


            # print progress status and memory use
            if t % 50000 == 0:
                self.result_dict.update({'homomorphisms_history': emb_history})
                self.result_dict.update({'code_history': code_history})
                self.compute_recons_accuracy(G_recons = self.G_recons_baseline)
                # print('iteration %i out of %i' % (t, recons_iter))
                # self.G_recons.get_min_max_edge_weights()
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                print('memory use:', memoryUse)

                gc.collect()

            # refreshing memory at checkpoints
            if (ckpt_epoch is not None) and (t % ckpt_epoch == 0) and (t>0):
                # print out current memory usage
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                print('memory use:', memoryUse)

                # print('num edges in G_count', len(self.G_overlap_count.get_edges()))

                ### Load and combine with the saved edges and reconstruction counts
                if has_saved_checkpoint:
                    print('!!! merging with previous checkpoint...')
                    self.G_recons_baseline.load_add_wtd_edges(path=path_recons_baseline, increment_weights=True,
                                                              is_pickle=True)

                    G_overlap_count_new = NNetwork()
                    G_overlap_count_new.add_wtd_edges(edges=self.G_overlap_count.wtd_edges)

                    G_overlap_count_old = NNetwork()
                    G_overlap_count_old.load_add_wtd_edges(path=path_overlap_count, increment_weights=False,
                                                           is_pickle=True)


                    if if_keep_visit_statistics:
                        G_overlap_count_new.add_colored_edges(colored_edges=self.G_overlap_count.colored_edges)
                        G_overlap_count_old.load_add_colored_edges(path=path_overlap_colored_count)

                        wtd_edges = G_overlap_count_old.wtd_edges
                        colored_edges = G_overlap_count_old.colored_edges
                        #print('!!! check if wtd and colored edges are the same:', wtd_edges.keys() == colored_edges.keys())



                    G_recons_new = NNetwork()
                    G_recons_new.add_colored_edges(colored_edges=self.G_recons.colored_edges)
                    # G_recons_new.get_min_max_edge_weights()

                    self.G_recons = NNetwork()
                    self.G_recons.load_add_colored_edges(path=path_recons)
                    # self.G_recons.get_min_max_edge_weights()

                    for edge in G_recons_new.wtd_edges.keys():
                        edge = eval(edge)
                        count_old = G_overlap_count_old.get_edge_weight(edge[0], edge[1])
                        count_new = G_overlap_count_new.get_edge_weight(edge[0], edge[1])

                        old_edge_weight = self.G_recons.get_colored_edge_weight(edge[0], edge[1]) # a list
                        new_edge_weight = G_recons_new.get_colored_edge_weight(edge[0], edge[1]) # a list

                        if count_old is None:
                            count_old = 0
                        if count_new is None:
                            count_old = 0
                        if old_edge_weight is None:
                            old_edge_weight = [0] * len(W_list)
                        if new_edge_weight is None:
                            new_edge_weight = [0] * len(W_list)

                        if count_old + count_new > 0:
                            new_edge_weight = (count_old / (count_old + count_new)) * np.asarray(old_edge_weight) + (
                                        count_new / (count_old + count_new)) * np.asarray(new_edge_weight)
                            new_edge_weight = list(new_edge_weight)

                        self.G_recons.add_colored_edge([edge[0], edge[1], new_edge_weight])
                        G_overlap_count_old.add_edge(edge=edge, weight=count_new, increment_weights=True)

                        if if_keep_visit_statistics:
                            old_edge_color = G_overlap_count_old.get_colored_edge_weight(edge[0], edge[1])
                            new_edge_color = G_overlap_count_new.get_colored_edge_weight(edge[0], edge[1])
                            #old_edge_color = old_edge_color)
                            #new_edge_color = int(new_edge_color)

                            if old_edge_color is not None and new_edge_color is not None:
                                new_edge_color = (count_old / (count_old + count_new)) * old_edge_color[0] + (
                                            count_new / (count_old + count_new)) * new_edge_color[0]
                            elif old_edge_color is not None: # then count_new is None
                                new_edge_color = old_edge_color[0]
                            elif new_edge_color is not None: # then count_old is None
                                new_edge_color = new_edge_color[0]

                            G_overlap_count_old.add_colored_edge(colored_edge=[edge[0], edge[1], [new_edge_color]])

                    # print('!!!! max new weight', max(new_edge_wts))
                    # self.G_recons = G_recons_old
                    self.G_overlap_count = G_overlap_count_old


                print('!!! num edges in G_recons', len(self.G_recons.get_edges()))
                print('!!! num edges in G_overlap_count', len(self.G_overlap_count.get_edges()))
                if if_keep_visit_statistics:
                    print('!!! num colored edges in G_overlap_count', len(self.G_overlap_count.colored_edges))
                print('!!! num edges in G_recons_baseline', len(self.G_recons_baseline.get_edges()))
                # self.G_recons.get_min_max_edge_weights()
                # self.G_overlap_count.get_min_max_edge_weights()

                ### Save current graphs
                self.G_recons.save_colored_edges(path_recons)
                self.G_overlap_count.save_wtd_edges(path_overlap_count)
                if if_keep_visit_statistics:
                    self.G_overlap_count.save_colored_edges(path_overlap_colored_count)
                self.G_recons_baseline.save_wtd_edges(path_recons_baseline)

                has_saved_checkpoint = True

                ### Clear up the edges of the current graphs
                self.G_recons = NNetwork()
                self.G_recons_baseline = NNetwork()
                self.G_overlap_count = NNetwork()
                self.G_recons.add_nodes(nodes=[v for v in G.vertices])
                self.G_recons_baseline.add_nodes(nodes=[v for v in G.vertices])
                self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])
                G_overlap_count_new = NNetwork()
                G_overlap_count_old = NNetwork()
                G_recons_new = NNetwork()

        if (ckpt_epoch is not None) and (recons_iter>ckpt_epoch):
            self.G_recons = NNetwork()
            self.G_recons.load_add_colored_edges(path=path_recons)
            self.G_overlap_count = NNetwork()
            self.G_overlap_count.load_add_wtd_edges(path=path_overlap_count, increment_weights=False, is_pickle=True)
            if if_keep_visit_statistics:
                self.G_overlap_count.load_add_colored_edges(path=path_overlap_colored_count)

        ### Save weigthed reconstruction into full results dictionary
        self.result_dict.update({'Colored edges in reconstruction': self.G_recons.colored_edges})

        if if_keep_visit_statistics:
            #print("@@@ edges_added", edges_added)
            denoising_dict = compute_denoising_stats(G=self.G,
                                                      edges_added=edges_added,
                                                      G_recons=self.G_recons,
                                                      G_overlap_count=self.G_overlap_count,
                                                      save_path=save_path)
            self.result_dict.update({'denoising_dict': denoising_dict})

        #print('ratio of masked edges in sampled patches: (mean, std)=({:.4f}, {:.4f})'.format(np.mean(chain_edge_stat), np.std(chain_edge_stat)))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        # self.compute_recons_accuracy(G_recons = self.G_recons_baseline)
        print('Number of edges in the baseline recons. : ', len(self.G_recons_baseline.get_edges()))
        #print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})

        # Return reconstruction
        return self.G_recons



    def compute_recons_accuracy(self, G_recons, if_baseline=False, edges_added=None, output_full_metrics=False):
        ### Compute reconstruction error
        G = self.G
        G_recons.add_nodes(G.vertices) # unnecessary if the node sets are already common
        G_nx = nx.Graph(G.get_edges())
        G_nx = G_nx.subgraph(sorted(nx.connected_components(G_nx), key=len, reverse=True)[0])
        G_recons_nx = nx.Graph(G_recons.get_edges())
        G_recons_nx = G_recons_nx.subgraph(sorted(nx.connected_components(G_recons_nx), key=len, reverse=True)[0])
        recons_metrics = {}


        # Jaccard metric
        common_edges = G.intersection(G_recons)
        recons_accuracy = len(common_edges) / (len(G.get_edges()) + len(G_recons.get_edges()) - len(common_edges))
        print('# edges of original ntwk=', len(G.get_edges()))
        self.result_dict.update({'# edges of original ntwk': len(G.get_edges())})

        print('# edges of reconstructed ntwk=', len(G_recons.get_edges()))
        print('Jaccard reconstruction accuracy=', recons_accuracy)
        self.result_dict.update({'# edges of reconstructed ntwk=': len(G_recons.get_edges())})
        self.result_dict.update({'reconstruction accuracy=': recons_accuracy})
        recons_metrics.update({'Jaccard_recons_accuracy': recons_accuracy})

        # Degree_distribution
        deg_seq = sorted((d for n, d in G_nx.degree()), reverse=True)
        deg_seq_recons = sorted((d for n, d in G_recons_nx.degree()), reverse=True)
        self.result_dict.update({'degree_sequence=': deg_seq})
        self.result_dict.update({'degree_sequence_recons=': deg_seq_recons})
        recons_metrics.update({'degree_sequence': deg_seq})
        recons_metrics.update({'degree_sequence_recons=': deg_seq_recons})

        # Average Shortest Path length
        if len(G.nodes()) < 1000:
            avg_path_len = nx.average_shortest_path_length(G_nx)
            avg_path_len_recons = nx.average_shortest_path_length(G_recons_nx)
            self.result_dict.update({'avg_shortest_path_length=': avg_path_len})
            self.result_dict.update({'avg_shortest_path_length_recons=': avg_path_len_recons})
            recons_metrics.update({'avg_shortest_path_length=': avg_path_len})
            recons_metrics.update({'avg_shortest_path_length_recons=': avg_path_len_recons})

        # Average Clustering Coefficient
        avg_clustering_coeff = nx.average_clustering(G_nx)
        avg_clustering_coeff_recons = nx.average_clustering(G_recons_nx)
        self.result_dict.update({'avg_clustering_coeff=': avg_clustering_coeff})
        self.result_dict.update({'avg_clustering_coeff_recons=': avg_clustering_coeff_recons})
        recons_metrics.update({'avg_clustering_coeff=': avg_clustering_coeff})
        recons_metrics.update({'avg_clustering_coeff_recons=': avg_clustering_coeff_recons})

        if if_baseline:
            print('# edges of reconstructed baseline ntwk=', len(self.G_recons_baseline.get_edges()))
            common_edges_baseline = G.intersection(self.G_recons_baseline)
            recons_accuracy_baseline = len(common_edges_baseline) / (
                    len(G.get_edges()) + len(self.G_recons_baseline.get_edges()) - len(common_edges_baseline))
            print('reconstruction accuracy for baseline=', recons_accuracy_baseline)
            self.result_dict.update(
                {'# edges of reconstructed baseline ntwk=': len(self.G_recons_baseline.get_edges())})
            self.result_dict.update({'reconstruction accuracy for baseline=': recons_accuracy_baseline})

        if output_full_metrics:
            return recons_metrics
        else:
            return recons_accuracy

    def compute_A_recons(self, G_recons):
        ### Compute reconstruction error
        G_recons.add_nodes_from(self.G.vertices)
        A_recons = nx.to_numpy_matrix(G_recons, nodelist=self.G.vertices)
        ### Having "nodelist=G.nodes" is CRUCIAL!!!
        ### Need to use the same node ordering between A and G for A_recons and G_recons.
        return A_recons


### helper functions

def patch_masking(X, k, ratio=0, chain_edge_masking=0):
    ### input is (k^2 x N) matrix X
    ### make all entries corresponding to the edges of the conditioned chain motif zero
    ### This may be applied to patches matrix and also to the dictionary matrix

    ### Reshape X into (k x k x N) tensor
    X1 = X.copy()
    X1 = X1.reshape(k, k, -1)

    if chain_edge_masking < 1:
        ### for each slice along mode 2, make the entries along |x-y|=1 be zero
        for i in np.arange(X1.shape[-1]):
            for x in itertools.product(np.arange(k), repeat=2):
                if np.abs(x[0] - x[1]) == 1:
                    X1[x[0], x[1], i] *= chain_edge_masking
    else:
        for i in np.arange(X1.shape[-1]):
            U = np.random.rand(k,k)
            V = np.triu(U, 1)
            diag = np.diag((np.random.rand(k)>ratio).astype(int))
            V = V + V.T + diag
            mask = (V>ratio).astype(int)
            X1[:,:,i] = X1[:,:,i]*mask

    return X1.reshape(k ** 2, -1)


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def compute_denoising_stats(G,
                             edges_added,
                             G_recons,
                             G_overlap_count,
                             save_path):

    print('!!! num edges in G_recons', len(G_recons.get_edges()))
    print('!!! num edges in G_overlap_count', len(G_overlap_count.get_edges()))
    print('!!! num colored edges in G_overlap_count', len(G_overlap_count.colored_edges))

    denoising_dict = {}
    c = 0
    c_true = 0
    wt = 0
    wt_true = 0
    avg_n_visits2false_edge = 0
    avg_n_visits2true_edge = 0
    visit_counts_false = []
    visit_counts_true = []
    recons_weights_false = []
    recons_weights_true = []
    recons_color_false = [] # simultaenous reconstruction stored in colored weights
    recons_color_true = [] # simultaenous reconstruction stored in colored weights
    avg_dist_on_chain_false = []
    avg_dist_on_chain_true = []

    false_edge_reconstructed = []

    H = NNetwork()
    H.add_edges(edges_added)
    edges_added = H.get_edges()  ### make it ordered pairs

    for edge in G.get_edges():
        if edge in edges_added:
            if G_recons.has_edge(edge[0], edge[1]):
                c += 1
                wt += G_recons.get_edge_weight(edge[0], edge[1])
                false_edge_reconstructed.append(edge)
                avg_n_visits2false_edge += G_overlap_count.get_edge_weight(edge[0], edge[1])
                recons_weights_false.append(G_recons.get_edge_weight(edge[0], edge[1]))
                recons_color_false.append(G_recons.get_colored_edge_weight(edge[0], edge[1]))

                visit_counts_false.append(G_overlap_count.get_edge_weight(edge[0], edge[1]))

                colored_edge_weight = G_overlap_count.get_colored_edge_weight(edge[0], edge[1])[0]
                # avg_dist_on_chain_false.append(sum(colored_edge_weight)/len(colored_edge_weight))
                avg_dist_on_chain_false.append(colored_edge_weight)
                #if colored_edge_weight <= 2:
                #    print('!!!!! On-chain distance for false edge=', colored_edge_weight)
            else:
                recons_weights_false.append(0)

        else:
            if G_recons.has_edge(edge[0], edge[1]):
                c_true += 1
                wt_true += G_recons.get_edge_weight(edge[0], edge[1])
                false_edge_reconstructed.append(edge)
                avg_n_visits2true_edge += G_overlap_count.get_edge_weight(edge[0], edge[1])
                recons_weights_true.append(G_recons.get_edge_weight(edge[0], edge[1]))
                recons_color_true.append(G_recons.get_colored_edge_weight(edge[0], edge[1]))
                visit_counts_true.append(G_overlap_count.get_edge_weight(edge[0], edge[1]))

                colored_edge_weight = G_overlap_count.get_colored_edge_weight(str(edge[0]), str(edge[1]))[0]
                # avg_dist_on_chain_true.append(sum(colored_edge_weight)/len(colored_edge_weight))
                avg_dist_on_chain_true.append(colored_edge_weight)
                #if colored_edge_weight <= 2:
                #    print('!!!!! On-chain distance for true edge=', colored_edge_weight)
            else:
                recons_weights_true.append(0)

    ### Get rid of the top 2% largest elements
    a = len(visit_counts_true) // 50
    visit_counts_true = sorted(visit_counts_true, reverse=True)[a:]
    b = len(visit_counts_true) // 50
    visit_counts_false = sorted(visit_counts_false, reverse=True)[b:]

    print('!!! n_false_edges', len(recons_weights_false))
    print('!!! n_true_edges', len(recons_weights_true))
    print('!!! max visits to false edges', max(visit_counts_false))
    print('!!! max visits to true edges', max(visit_counts_true))

    print('# of false edges ever reconstructed= %i out of %i' % (c, len(edges_added)))
    print('ratio of false edges ever reconstructed=', c / len(edges_added))
    print('avg reconstructed weight of false edges', wt / len(edges_added))
    print('avg reconstructed weight of true edges', wt_true / len(G.edges))
    print('avg_n_visits2false_edge', avg_n_visits2false_edge / len(edges_added))
    print('avg_n_visits2true_edge', avg_n_visits2true_edge / len(G.edges))

    denoising_dict.update({'False edges added': edges_added})
    denoising_dict.update({'False edges ever reconstructed': false_edge_reconstructed})
    denoising_dict.update({'ratio of false edges ever reconstructed': c / len(edges_added)})
    denoising_dict.update({'avg reconstructed weight of false edges': wt / len(edges_added)})
    denoising_dict.update({'avg reconstructed true of false edges': wt_true / len(G.edges)})
    denoising_dict.update({'recons_weights_false': recons_weights_false})
    denoising_dict.update({'recons_weights_true': recons_weights_true})
    denoising_dict.update({'recons_colored_weights_false': recons_color_false})
    denoising_dict.update({'recons_colored_weights_true': recons_color_true})
    denoising_dict.update({'visit_counts_false': visit_counts_false})
    denoising_dict.update({'visit_counts_true': visit_counts_true})
    denoising_dict.update({'avg_dist_on_chain_false': avg_dist_on_chain_false})
    denoising_dict.update({'avg_dist_on_chain_true': avg_dist_on_chain_true})
    return denoising_dict
