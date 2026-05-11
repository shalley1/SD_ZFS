"""

Description: dataclasses to store information about the graph and greedy/optimal solutions to ZFS problem
"""


import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass
import pickle

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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
            [[1,2,4],[0]...]
        nx_graph: This stores the graph in a NetworkX graph for debugging
    """

    num_nodes: int
    num_edges: int
    adj_list: list[list[int]]
    edge_pairs: np.ndarray
    nx_graph : nx.Graph | None = None



#added kw_only bc nx_graph is defaulted in parent class
@dataclass(kw_only=True)
class GraphWSolutions(S2VGraph):
    #will include nx_graph object
    optimal: Optional[int]=None
    greedy: Optional[int]=None
    greedy_sol: Optional[list[int]]
    time_greedy: Optional[float]=None
    optimal_sol: Optional[list[int]]
    source: Optional[str]
    optimal_feasible: Optional[str]









