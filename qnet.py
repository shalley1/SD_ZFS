"""
File: Qnet
Description: Multi-Component neural network composed of S2V component and DQN 

Usage:
    Training:
    - Model will be trained on batches of graph data
    - S2V component generates graph embedding/individual node embeddings for a full pool of graphs
    - Model returns a tensor of q-scores which are estimated quantitative values of each node in the graph
    Optimizing:
    - A replay buffer stores all transitions
    - Transitions are randomly sampled and combined into mini-batch
    - Loss is calculated for each n-step transition using Bell equation
    - Loss can be double DQN or vanilla

Dependencies:
   The S2V framework is an external library that has light modifications to return full graph embedding
   Original S2V code: https://github.com/Hanjun-Dai/pytorch_structure2vec
   Modified code: external/s2v_lib/



"""





# ruff: noqa: F401
# ruff: noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#import time
import random
import importlib
import sys

#s2v backend
s2v_module = importlib.import_module("external.s2v_lib.s2v_lib")
sys.modules["s2v_lib"] = s2v_module
pytorch_util_module = importlib.import_module("external.s2v_lib.pytorch_util")
sys.modules["pytorch_util"] = pytorch_util_module
pytorch_embedding_module = importlib.import_module("external.s2v_lib.embedding")
sys.modules["embedding"] = pytorch_embedding_module

#s2v backend
from s2v_lib import S2VLIB
from pytorch_util import gnn_spmm
from embedding import EmbedMeanField

from transition import ReplayBuffer,Transition
from data.generator import generate_network
from data.s2vgraph import S2VGraph

class QNet(nn.Module):
    def __init__(self,latent_dim,out_dim,num_node_feats,hidden_dim,max_lv=3):
        super().__init__()
        
        """
        nn composed of two components: 
            1. Embedding component to capture node/graph structure 
            2. Q-scorer component to generate q-score for each node
        """
        self.embedder = EmbedMeanField(
            latent_dim = latent_dim,
            output_dim=out_dim,
            num_node_feats=num_node_feats,
            num_edge_feats=0,
            max_lv=max_lv
        )

        self.q_scorer = nn.Sequential(
            #out_dim*2 because it will be node emb+graph embedding concatenated
            nn.Linear(out_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
                )
    
    def forward(self,graph_list,node_feat):
        #H = node embeddings, g = graph embedding
        H, g = self.embedder(graph_list, node_feat, edge_feat=None)   
        
        #list of number of nodes in each graph within the batch
        sizes = [gr.num_nodes for gr in graph_list] 
        
        # 1 graph embedding associated with each node embedding
        g_node = torch.repeat_interleave(g, torch.as_tensor(sizes, device=g.device), dim=0)  
        combined_embedding = torch.cat([H,g_node],dim=1)
        
        #returning the q-scores along with the size of each graph
        q_flat = self.q_scorer(combined_embedding).squeeze(-1)
        return q_flat,sizes

#converting the node features to tensors
def build_node_features(node_features: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(node_features, dtype=torch.float32, device=device)
    
    #for when there is 1 feature
    if x.ndim == 1:
        x = x.unsqueeze(1)  

    return x


#need to offset the node labels for processing batches of graphs
def prefix_offsets(sizes: list[int], device) -> torch.Tensor:
    # offsets[i] = sum_{j<i} sizes[j]
    return torch.cumsum(torch.tensor([0] + sizes[:-1], device=device), dim=0)

#converting the tensor to mask illegal actions by making q-score negative inf
def masked_max_tensor(q_nodes: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(legal_mask):
        return torch.zeros((), device=q_nodes.device, dtype=q_nodes.dtype)
    neg_inf = torch.finfo(q_nodes.dtype).min
    q_masked = torch.where(legal_mask, q_nodes, neg_inf)
    return q_masked.max()

#argmax
def masked_argmax_tensor(q_nodes: torch.Tensor, legal_mask: torch.Tensor) -> int:
    if not torch.any(legal_mask):
        return -1
    neg_inf = torch.finfo(q_nodes.dtype).min
    q_masked = torch.where(legal_mask, q_nodes, neg_inf)
    return int(torch.argmax(q_masked).item())



#back propagate the losses and return loss
def optimize_q_network(
    policy_qnet: nn.Module,
    target_qnet: nn.Module,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    double: bool,
    gamma: float,
    device: torch.device,
) -> float:
    if len(replay) < batch_size:
        return 0.0

    batch = replay.sample(batch_size)

    policy_qnet.train()
    target_qnet.eval()

    graph_list = [tr.graph for tr in batch]

    x_state = torch.cat([build_node_features(tr.node_features, device) for tr in batch], dim=0)       
    x_next  = torch.cat([build_node_features(tr.next_node_features, device) for tr in batch], dim=0) 


    #prediction --> Q(s,a)
    q_state_flat, sizes = policy_qnet(graph_list, x_state)
    offsets = prefix_offsets(sizes, device=device)
    
    # Vanilla --> y = r + Gamma * Max(Q_target(s',a))
    # Double --> y = r + Gamma * Q_target(s',argmax(Q_policy(s',a)))
    
    if double:
        with torch.no_grad():
            q_next_policy_flat, _ = policy_qnet(graph_list, x_next)
            q_next_target_flat, _ = target_qnet(graph_list, x_next)
            q_next_policy_splits = torch.split(q_next_policy_flat, sizes, dim=0)
            q_next_target_splits = torch.split(q_next_target_flat, sizes, dim=0)
    else:
        with torch.no_grad():
            q_next_target_flat, _ = target_qnet(graph_list,x_next) 
            q_next_target_splits = torch.split(q_next_target_flat, sizes, dim=0)

    losses = []
    for i, tr in enumerate(batch):

        #need to get action number in global context of graph 
        a_global = int(offsets[i].item()) + int(tr.action)
        q_sa = q_state_flat[a_global]

        if tr.done:
            y = torch.tensor(tr.reward, device=device, dtype=q_sa.dtype)
        else:
            legal_mask = (torch.as_tensor(tr.next_state, device=device) == 0)  
                
            if double:
                best_action = masked_argmax_tensor(q_next_policy_splits[i],legal_mask)
                if best_action==-1:
                    #shouldn't get triggered but this is a failsafe..
                    next_val = torch.zeros((),device=device,dtype=q_sa.dtype)
                else:
                    next_val = q_next_target_splits[i][best_action]
            else:
                next_val = masked_max_tensor(q_next_target_splits[i],legal_mask)

            #max_next = masked_max_tensor(q_next_splits[i], legal_mask)
            #bellman equation
            #y = torch.tensor(tr.reward, device=device, dtype=q_sa.dtype) + gamma * next_val
            #update for n-step
            y = torch.tensor(tr.reward, device=device, dtype=q_sa.dtype) + (gamma ** tr.n_steps) * next_val
        losses.append(F.smooth_l1_loss(q_sa, y))

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_qnet.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def main():
    
    pass
if __name__=="__main__":
    main()

