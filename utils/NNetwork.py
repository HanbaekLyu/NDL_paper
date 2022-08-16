import numpy as np
import csv
import ast
import pickle

"""
Neighborhood-based Network package
Instead of storing list of edges, store list of neighbors of each node
This makes local MC updates and neighborhood query times O(1) time
Simple and weighted network objects with built-in neighborhood list for scalable random walk and MCMC sampling
Contains sampling algorithms for mesoscale network patch
Default class is network with weighted edges, which can also have list-valued weights as its 'color'
Author: Hanbaek Lyu and Josh Vendrow
"""

class NNetwork():
    '''
    Class for weighted and undirectional network
    Can store additional tensor information in colored_edge_weights -- list of nonnegative numbers
    Underlying NNetwork serves as the skeleton for RWs and motif sampling
    colored edges = additional information on the (already weighted) edges
    Think of edges = pixel location and colored edges = color of that pixel
    '''

    def __init__(self):
        # self.edges = []
        self.wtd_edges = {}
        self.neighb = {}
        self.vertices = []
        self.vertices_set = set()
        self.number_nodes = 0
        self.colored_edges = {}
        self.color_dim = 0

    def add_edges(self, edges, edge_weight=1, increment_weights=True):
        """Given an edgelist, add each edge in the edgelist to the network"""
        for edge in edges:
            self.add_edge(edge, weight=edge_weight, increment_weights=increment_weights)
            # self.edge_weights.update({str(edge): default_edge_weight})

    def add_edge(self, edge, weight=float(1), increment_weights=False, is_dict=False):
        """Given an edge, add this edge to the Network"""
        # if is_dict, then edge is the form of "['123', '456']".
        if not is_dict:
            u, v = edge
        else:
            u, v = eval(edge)

        u = str(u)
        v = str(v)
        # self.edges.append(edge)
        if not increment_weights or not self.has_edge(u, v):
            wt = weight
        else:
            wt = self.get_edge_weight(u, v) + weight

        self.wtd_edges.update({str([str(u), str(v)]): wt})
        self.wtd_edges.update({str([str(v), str(u)]): wt})

        if u not in self.neighb:
            self.neighb[u] = {v}
            self.vertices.append(u)
            self.vertices_set.add(u)
            self.number_nodes += 1
        else:
            self.neighb[u].add(v)

        if v not in self.neighb:
            self.neighb[v] = {u}
            self.vertices.append(v)
            self.vertices_set.add(v)
            self.number_nodes += 1
        else:
            self.neighb[v].add(u)

    def add_nodes(self, nodes):
        """Given a list of nodes, adds the nodes to the Network"""
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        """Given a single node, adds the node to the Network"""
        if node not in self.vertices:
            self.vertices.append(node)
            self.neighb[node] = set()
            self.number_nodes += 1
            self.vertices_set.add(node)

    def set_neighbors(self):
        for edge in self.edges:
            self.add_edge(edge)

    def uniq(self, lst):
        ### get rid of duplicates in an input list "lst"
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item
        ### Returns a generator object: take "list" to turn the output into a list

    def intersection(self, Network_other):
        """
        Given another network, returns all edges found in both networks.
        """
        common_nodeset = self.vertices_set.intersection(Network_other.vertices_set)
        common_edgelist = []

        for x in common_nodeset:
            for y in self.neighb[x].intersection(Network_other.neighbors(x)):
                common_edgelist.append([x, y])

        return common_edgelist

    def intersection_slow(self, Network_other):
        """
        Given another network, returns all edges found in both networks.
        """
        edges_other = list(self.uniq(Network_other.get_edges()))
        print('edges_other', edges_other)
        edges = list(self.uniq(self.get_edges()))
        print('edges', edges)
        intersection = [e for e in edges if e in edges_other]
        print('intersection', intersection)
        return intersection

    def neighbors(self, node):
        """
        Given a node, returns all the neighbors of the node.
        """
        if node not in self.nodes():
            print('ERROR: node %s not in the node set' % str(node))

        return self.neighb[node]

    def has_edge(self, u, v):
        """
        Given two nodes, returns true of these is an edge between then, false otherwise.
        """
        try:
            return v in self.neighb[u]
        except KeyError:
            return False

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def read_adj(self, adj):
        """
        adj = n x n (weighted) adjacency matrix
        """
        for i in np.arange(adj.shape[0]):
            nb_list = self.indices(adj[:, i], lambda x: x == 1)
            for j in nb_list:
                self.add_edge(edge=[i,j], weight=float(adj[i,j]), increment_weights=False, is_dict=False)

    def has_colored_edge(self, u, v):
        """
        Given two nodes, returns true of these is a colored edge between then, false otherwise.
        """
        colored_edge = self.get_colored_edge_weight(u, v)
        return colored_edge != None

    def nodes(self, is_set=False):
        return self.vertices if not is_set else self.vertices_set

    def num_nodes(self):
        return self.number_nodes

    def get_edge_weight(self, u, v):
        return self.wtd_edges.get(str([str(u), str(v)]))

    def get_edges(self):
        set_edgelist = []
        for x in self.vertices:
            if x in self.neighb:  ### x has a neighbor
                for y in self.neighb[x]:
                    set_edgelist.append([x, y])
        self.edges = set_edgelist
        return self.edges

    def get_wtd_edgelist(self):
        edgelist = self.get_edges()
        wtd_edgelist = []
        for edge in edgelist:
            u, v = edge
            wtd_edgelist.append([u, v, abs(self.get_edge_weight(u, v))])

        return wtd_edgelist

    def get_abs_wtd_edgelist(self):
        edgelist = self.get_edges()
        wtd_edgelist = []
        for edge in edgelist:
            u, v = edge
            wtd_edgelist.append([u, v, self.get_edge_weight(u, v)])

        return wtd_edgelist

    def save_wtd_edgelist(self, default_folder='Temp_save_graphs', default_name='temp_wtd_edgelist'):
        edgelist = self.get_edges()
        wtd_edgelist = []
        for edge in edgelist:
            u, v = edge
            wtd_edgelist.append([u, v, self.get_edge_weight(u, v)])

        ### Todo: Use pickle later
        with open(default_folder + '/' + default_name + '.txt', "w") as file:
            file.write(str(wtd_edgelist))
        return wtd_edgelist

    def save_wtd_edges(self, path):
        # Does not creat additional list file and saves directly self.wtd_edges as a np dictionary.
        with open(path, 'wb') as handle:
            pickle.dump(self.wtd_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_colored_edges(self, path):
        # Does not creat additional list file and saves directly self.wtd_edges as a np dictionary.
        with open(path, 'wb') as handle:
            pickle.dump(self.colored_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_wtd_edges(self, edges, increment_weights=False):
        """Given an edgelist (or dictionary), add each edge in the edgelist to the network"""
        """if is dict, edges is a dictionary {edge: weight}"""
        if isinstance(edges, list):
            for wtd_edge in edges:
                if len(wtd_edge) == 2:
                    self.add_edge(wtd_edge[0:2], weight=1, increment_weights=increment_weights)
                else:
                    self.add_edge(wtd_edge[0:2], weight=float(wtd_edge[-1]), increment_weights=increment_weights)
        else:
            for edge in edges.keys():
                if isinstance(edge, str):
                    edge = eval(edge)
                self.add_edge(edge, weight=edges.get(str(edge)), increment_weights=increment_weights)

    def load_add_wtd_edges(self, path, delimiter=',', increment_weights=False, use_genfromtxt=False,
                           is_pickle=False):

        #if is_dict and not is_pickle:
        #    wtd_edges = np.load(path, allow_pickle=True).items()
        if is_pickle:
            with open(path, 'rb') as handle:
                wtd_edges = pickle.load(handle)

        elif not use_genfromtxt:
            with open(path, "r") as file:
                wtd_edges = eval(file.readline())
        elif delimiter == '\t':
            with open(path) as f:
                reader = csv.reader(f, delimiter="\t")
                wtd_edges = list(reader)

        else:
            wtd_edges = np.genfromtxt(path, delimiter=delimiter, dtype=str)
            wtd_edges = wtd_edges.tolist()

        self.add_wtd_edges(edges=wtd_edges, increment_weights=increment_weights)

    def get_min_max_edge_weights(self):
        list_wts = []
        for edge in self.get_wtd_edgelist():
            list_wts.append(edge[-1])
        print('minimum edge weight:', min(list_wts))
        print('maximum edge weight:', max(list_wts))

    def clear_edges(self):
        node_set = self.vertices
        self.__init__()
        self.add_nodes(node_set)

    def threshold2simple(self, threshold=0.5):
        ### Threshold edges of the given weighted network and return a simple graph
        G_simple = NNetwork()
        for edge in self.get_edges():
            u, v = edge
            edge_weight = self.get_edge_weight(u, v)
            if edge_weight > threshold:
                G_simple.add_edge([u, v])
        return G_simple


    def add_colored_edges(self, colored_edges, is_dict=True):
        """Given a list/dictionary of colored edges, add each colored edge to the network"""
        if isinstance(colored_edges, list):
            for colored_edge in colored_edges:
                self.add_colored_edge(colored_edge)
        else:
            for edge in colored_edges.keys():
                u, v = eval(edge)
                self.add_colored_edge([u,v, colored_edges.get(edge)])


    def add_colored_edge(self, colored_edge):
        """Given a colored edge, [u, v, [a1, a2, ...]], add this to the Network"""
        u, v = colored_edge[0:2]

        self.colored_edges.update({str([str(u), str(v)]): colored_edge[2:][0]})
        self.colored_edges.update({str([str(v), str(u)]): colored_edge[2:][0]})
        if not self.has_edge(u,v):
            self.add_edge(edge = [u, v], weight=1)

    def load_add_colored_edges(self, path):
        # pickle only
        with open(path, 'rb') as handle:
            colored_edges = pickle.load(handle)
        self.add_colored_edges(colored_edges=colored_edges)

    def set_clrd_edges_signs(self):
        """
        colored edge weight of and edge with weight w = [+(w), -(w)] (split positive and negative parts as a vector)
        """
        edgelist = self.get_edges()

        for edge in edgelist:
            u, v = edge
            w = self.get_edge_weight(u, v)
            colored_edge = [u, v, w * (w >= 0), -w * (w < 0)]
            self.add_colored_edge(colored_edge)

        u, v = self.get_edges()[0]
        self.color_dim = len(self.get_colored_edge_weight(u, v))

    def get_colored_edge_weight(self, u, v):
        color = self.colored_edges.get(str([str(u), str(v)]))
        return color


    def get_adjacency_matrix(self, ordered_node_list=None):
        if ordered_node_list is None:
            ordered_node_list = self.vertices
        mx = np.zeros(shape=(len(ordered_node_list), len(ordered_node_list)))
        for i in np.arange(len(ordered_node_list)):
            for j in np.arange(len(ordered_node_list)):
                if self.get_edge_weight(ordered_node_list[i], ordered_node_list[j]) is not None:
                    mx[i, j] = self.get_edge_weight(ordered_node_list[i], ordered_node_list[j])
        return mx

    def subgraph(self, nodelist):
        ### Take induced subgraph on the specified nodeset
        V0 = set(nodelist).intersection(self.vertices)
        G_sub = NNetwork()
        for u in V0:
            nbh_sub = self.neighbors(u).intersection(set(nodelist))
            if len(nbh_sub) > 0:
                for v in nbh_sub:
                    G_sub.add_edge([u, v], weight=self.get_edge_weight(u, v))

        return G_sub

    def r_neighborhood(self, nodelist, radius=1):
        ### Take the induced subgraph of the r-neighborhood of a given subset of nodes
        V = set(nodelist).intersection(self.vertices)
        for r in np.arange(radius):
            for u in V:
                V = V.union(self.neighbors(u))

        return self.subgraph(V)

    def count_k_step_walks(self, u, radius=1):
        ### Counts the number of k-step walks starting from u
        ### Take the r-neighborhood, take the induced adjacency matrix Ar, and take r th power, and then sum
        G_r_nbh = self.r_neighborhood([u], radius=radius)
        vertices = G_r_nbh.vertices
        idx = vertices.index(u)

        Ar = np.zeros(shape=(len(G_r_nbh.vertices), len(G_r_nbh.vertices)))
        for i in np.arange(len(self.vertices)):
            for j in np.arange(len(self.vertices)):
                if G_r_nbh.has_edge(self.vertices[i], self.vertices[j]):
                    Ar[i, j] = G_r_nbh.get_edge_weight(self.vertices[i], self.vertices[j])

        B = np.linalg.matrix_power(Ar, radius - 1)
        print('!!!! B', B)

        return sum(B[idx, :])

    def k_node_ind_subgraph(self, k, center=None):
        # Computes a random k-node induced subgraph
        # Perform simple symmetric RW until collecting k distinct nodes
        # Once a set of k distinct nodes are collected, take the induced subgraph on it
        if k > len(self.vertices):
            print("cannot take i% distinct nodes from a graph with %i nodes" % (k, len(self.vertices)))

        V0 = []
        while (len(V0) < k): # RW may be initialized in a small component with <k nodes
        # -- restart if it doesn't end within k^2 steps
            i = 0
            V0 = []
            if i > k**2:
                return None

            x = np.random.choice(self.vertices)
            if center is not None:
                x = center
            V0.append(x)
            while (len(V0) < k):
                x = np.random.choice(list(self.neighbors(x)))
                V0.append(x)
                V0 = list(set(V0))
                i += 1
                if i > k**2:
                    return None
            # now take subgraph on V0:
            i += 1
        return self.subgraph(V0)

    def k_node_IDLA_subgraph(self, k, center=None):
        # Computes a random k-node induced subgraph
        # Uses Internal Diffusion-Limited Aggregation (IDLA)
        # Random walker starts from the root and walks until discovering new node
        # Repeat k-1 times to get k-node subgraph.
        if k > len(self.vertices):
            print("cannot take i% distinct nodes from a graph with %i nodes" % (k, len(self.vertices)))

        x0 = np.random.choice(self.vertices)
        if center is not None:
            x0 = center

        V0 = [x0]
        while (len(V0) < k):
            i = 0
            z = x0

            if i > k**2:
                return None

            while (z in V0):
                z = np.random.choice(list(self.neighbors(z)))
                i += 1
                if i > k**2:
                    return None

            V0.append(z)
            V0 = list(dict.fromkeys(V0)) # respect original ordering
            i += 1
        return self.subgraph(V0)

    def colorededges2wtdedges(self, index=0):
        wtd_edges = {}
        for edge in self.colored_edges.key():
            edge = eval(edge)
            colored_edge_weight = self.get_colored_edge_weight(edge[0], edge[1])
            wtd_edges.update({str(edge): colored_edge_weight[index]})
        return wtd_edges
