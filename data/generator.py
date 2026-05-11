"""
Description: Utilize NetworkX for graph generation. Graphs used for training/testing in s2v_dqn framework on combinatorial optimization problems.
"""

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import numpy as np
import networkx as nx
#from itertools import combinations
from dataclasses import dataclass
#setup_logging()

@dataclass
class S2VGraph:
    """
    This is designed to work with C++ backend that 
    Args:
        num_nodes
        num_edges
        edge_pairs: int32 numpy array of length 2E storing:
            [u0, v0, u1, v1, ..., u_{E-1}, v_{E-1}]
        adj_list: stores list of neighbors per node
        nx_graph: This stores the graph in a NetworkX graph for debugging
    """

    num_nodes: int
    num_edges: int
    adj_list: list[list[int]]
    edge_pairs: np.ndarray
    nx_graph : nx.Graph | None = None



def generate_network(n,m=None, kind="er",p=0.01, k=None,seed=None,connected=False,save_graph_nx=False) -> S2VGraph:
    """
    Generates an Erdos-Renyi network with `n` nodes and linking probability `p`.
    Alternatively, if `k` is given, creates a random network with average degree `k` and ignores `p`.
    
    Graph is formatted in S2V graph with edge_pairs to be compatible with C++ backend s2v embedding.
    
    Note: Nodes must be labeled 0..n-1 for backend

    Args:
        n (int) : The number of nodes to generate.
        p (float) : The linking probability in the interval [0,1].
        k (int) : (Optional) The average degree of the network. If `k` is given, `p` is ignored and calculated accordingly.
        seed (int) : (Optional)    For reproducibility 
    Returns:
        S2VGraph object that represents the graph in format compatible with backend.
    """ 
    if k is not None:
        if n <= 1:
            raise ValueError("n must be > 1 if k is used.")
        p = k / (n - 1)

    def sample_graph(attempt_seed):
        #networkx has its own random number generator for seed if one is not passed in.
        if kind=="er":
            return nx.gnp_random_graph(n, p, seed=attempt_seed)  # nodes are 0..n-1
        elif kind=='ba':
            if m is None:
                raise ValueError("Need to provide m for BA graph")
            return nx.barabasi_albert_graph(n,m=m)
        elif kind =='powerlaw_cluster':
            if m is None:
                raise ValueError("Need to provide m for Powerlaw cluster graph")
            return nx.powerlaw_cluster_graph(n=n,p=p,m=m, seed=attempt_seed)  # 
        #default to er 
        else:
            return nx.gnp_random_graph(n, p, seed=attempt_seed)  # 
        #add in: nx.connected_watts_strogatz_graph(n=20, k=8, p=0.2, seed=0)

    if connected:
        base_seed = seed
        for attempt in range(1500):
            attempt_seed = None if base_seed is None else (base_seed + attempt)
            g = sample_graph(attempt_seed)
            if nx.is_connected(g):
                break
        else:
            g = sample_graph(attempt_seed)
    else:
        g = sample_graph(seed)

    # edge_pairs for C++ backend
    edges = np.array(list(g.edges()), dtype=np.int32).reshape(-1, 2)
    edge_pairs = np.ascontiguousarray(edges.reshape(-1), dtype=np.int32)
    num_edges = g.number_of_edges()

    # adjacency list for env/training (fast, simple)
    adj_list = [list(g.neighbors(i)) for i in range(n)]

    return S2VGraph(
        num_nodes=n,
        num_edges=num_edges,
        edge_pairs=edge_pairs,
        adj_list=adj_list,
        nx_graph=g if save_graph_nx else None,
    )





def draw_network(g, node_color='red'):
    import matplotlib.pyplot as plt
    nx.draw(g, pos=nx.spring_layout(g, scale=5),
            node_size=100,
            node_color=node_color,
            edgecolors='black',  # node `edgecolors` pertain to Matplotlib, node the network (this is the edge (outline) of the node symbol!)
            alpha=0.75,
            with_labels=True,
            linewidths=2)
    plt.show()

if __name__ == "__main__":
    n=20
    p=0.20
    m=4
    c=True
    kind='powerlaw_cluster'
    seed=1
    g =generate_network(n=n,p=p,m=m,kind=kind,connected=c,seed=1,save_graph_nx=True)
    print(g)
    draw_network(g.nx_graph)
