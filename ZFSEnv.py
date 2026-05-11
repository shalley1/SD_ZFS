

"""
File: ZFSEnv
Description: 
    - ZFS Environment where the problem of finding a minimum ZFS on a graph occurs
    - The environment includes relevant information in typical reinforcement learning structure like action/state/reward
    - Functions to initialize, step(select an action and update), prune(remove redundant blue nodes), propagate(spread blue nodes to white)
Usage:
    used in conjunction with q-network(s2v-dqn) NN. The model is trained to generate q-scores for actions(selecting a node in the graph) and then the model
    will gather info about next state within this environment and be trained on how effective it generated a q-score to solve problems within ZFS environment


"""




# ruff: noqa: F401
# ruff: noqa: E402
from collections import deque
#from typing import List, Tuple

import numpy as np
import networkx as nx
from data.generator import generate_network
from data.s2vgraph import S2VGraph

class ZFSEnv:
    def __init__(self, graph):
        self.g = graph
        self.n = graph.num_nodes
        self.edge_pairs = graph.edge_pairs.astype(np.int32)
        self.adj_list = graph.adj_list
        clustering_dict = nx.clustering(self.g.nx_graph)
        self.clustering = np.array(
            [clustering_dict[v] for v in range(self.n)],
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.selected = np.zeros(self.n, dtype=np.float32)
        self.total = 0
        self.blue = self.selected.copy()
        self.valid = self.legal_actions()

        state=self.selected.copy()
        #white_deg = self.compute_white_deg_norm(self.blue)
        #node_feats = np.stack([self.selected,self.blue],axis=1)
        #node_feats = np.stack([self.selected, self.blue, white_deg], axis=1)
        node_feats = np.stack([self.selected, self.blue, self.clustering], axis=1)
        return state,node_feats

    def legal_actions(self):
        return np.where(self.selected == 0)[0]
    
    def closure(self,S):
        blue = S.astype(np.int32).copy()
        white_deg = np.zeros(self.n, dtype=np.int32)
    
        for v in range(self.n):
            cnt = 0
            for u in self.adj_list[v]:
                if blue[u] == 0:
                    cnt += 1
            white_deg[v] = cnt

        q = deque([v for v in range(self.n) if blue[v] == 1])
    
        while q:
            v = q.popleft()

            if blue[v] == 0:
                continue
            if white_deg[v] != 1:
                continue

            u_force = None
            for u in self.adj_list[v]:
                if blue[u] == 0:
                    u_force = u
                    break
            if u_force is None:
                continue

            blue[u_force] = 1

            cnt = 0
            for x in self.adj_list[u_force]:
                if blue[x] == 0:
                    cnt += 1
            white_deg[u_force] = cnt

            for w in self.adj_list[u_force]:
                if blue[w] == 1:
                    white_deg[w] -= 1
                    q.append(w)

            q.append(u_force)
        return blue.astype(np.float32)

    def zero_force_propagation(self):

        self.blue =self.closure(self.selected)

    def blue_size(self):
        return int(np.sum(self.blue))
    #def compute_white_deg_norm(self, blue):
        white_deg = np.zeros(self.n, dtype=np.float32)
        for v in range(self.n):
            cnt = 0
            for u in self.adj_list[v]:
                if blue[u] == 0:
                    cnt += 1
            deg_v = max(1, len(self.adj_list[v]))
            white_deg[v] = cnt / deg_v
        return white_deg

    def is_done(self):
        return bool(np.all(self.blue == 1.0))

    def is_zfs(self, seed):
            blue = self.closure(seed)
            return bool(np.all(blue == 1.0))

    def prune(self):
        if not self.is_zfs(self.selected):
            raise ValueError("running prune on non-blued set")

        changed=True
        while changed:
            changed=False

            selected_nodes = np.where(self.selected==1)[0]
            for v in selected_nodes:
                trial = self.selected.copy()
                trial[v]=0.0

                if self.is_zfs(trial):
                    self.selected=trial
                    self.total-=1
                    changed=True
                    #break to restart the scan for redundancy
                    break
        self.total = int(np.sum(self.selected))
        self.zero_force_propagation()
        self.valid=self.legal_actions()
        return self.selected.copy(),self.total

    def step(self, action):
        if action<0 or action >=self.n:
            raise ValueError("action out of bounds")
        
        if self.selected[action] == 1:
            raise ValueError("Picked already selected node")
        self.selected[action] = 1.0
        self.total += 1
        if self.total > self.n:
            raise ValueError("Total nodes> num_nodes after step")

        self.zero_force_propagation()
        self.valid = self.legal_actions()
        done = self.is_done()
        reward = -1.0

        next_state=self.selected.copy()
        #next_node_feats = np.stack([self.selected,self.blue],axis=1)
        #white_deg = self.compute_white_deg_norm(self.blue)
        #next_node_feats = np.stack([self.selected, self.blue, white_deg], axis=1)
        next_node_feats = np.stack([self.selected, self.blue, self.clustering], axis=1)
        info = {
            "blue_size":self.blue_size()
                }

        return next_state,next_node_feats, reward, done, info

def main():
    pass
    # g = generate_network(p=0.2, n=10, connected=True)

    # env = ZFSEnv(g)
    # s, f = env.reset()
    # assert s.shape == (g.num_nodes,)
    # assert f.shape == (g.num_nodes, 2)

    # a = env.legal_actions()[0]
    # ns, nf, r, done, _ = env.step(a)
    # assert ns.shape == (g.num_nodes,)
    # assert nf.shape == (g.num_nodes, 2)

if __name__=="__main__":
    main()
