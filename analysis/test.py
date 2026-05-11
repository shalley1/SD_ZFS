"""
Description: Test model performance on IMDB dataset
"""



import os
from tqdm import tqdm
import pickle
import torch

from data.s2vgraph import S2VGraph
from ZFSEnv import ZFSEnv
from data.s2vgraph import GraphWSolutions

from qnet import QNet
from train_two_feats import select_action
import csv

def load_pickle(filepath):
    with open(filepath,'rb') as file:
        return pickle.load(file)

def load_params(model_dict):
    params = model_dict['args']
    latent_dim = params['latent_dim']
    out_dim = params['out_dim']
    num_node_feats = params['num_node_feats']
    hidden_dim  = params['hidden_dim']
    max_lv= params['max_lv']
    return latent_dim,out_dim,num_node_feats,hidden_dim,max_lv

def main(): 
    device = torch.device('cpu')

    models = [os.path.splitext(f)[0] for f in os.listdir('models') if f.endswith('.pkl')]
    er_solutions = load_pickle('data/imdb_binary.pkl')
    
    rows=[]
    for mod_pkl in tqdm(models):
        model_info = load_pickle('models/'+mod_pkl+'.pkl')
        latent_dim,out_dim,num_node_feats,hidden_dim,max_lv = load_params(model_info)
        model = QNet(latent_dim=latent_dim,out_dim=out_dim,num_node_feats=num_node_feats,hidden_dim=hidden_dim,max_lv=max_lv)
        model.load_state_dict(torch.load('models/'+mod_pkl+'.pt'))
        model.eval()
        g_total=0
        opt_list=[]
        dev_total=0
        for obj in tqdm(er_solutions):
            formatted = S2VGraph(num_nodes = obj.num_nodes, 
                                 num_edges=obj.num_edges,
                                 adj_list=obj.adj_list,
                                 edge_pairs=obj.edge_pairs,
                                 nx_graph=obj.nx_graph)
            env = ZFSEnv(formatted)
            state,features = env.reset()
            done=False

            while not done:
                if features.shape[1] != num_node_feats:
                    features = features[:, :num_node_feats]
                action = select_action(qnet=model,graph=env.g,state=state,node_feats=features,eps=0,device=device)
                next_state, next_features, reward, done, _= env.step(action)
                state=next_state
                features=next_features
            pruned_selected, pruned_size = env.prune()
            diff_greedy = obj.greedy - pruned_size
            dev = diff_greedy/obj.greedy
            if obj.optimal==0:
                diff_optimal = "N/A"
            else:
                diff_optimal = obj.optimal - pruned_size
            row = {
                    "run_id":model_info['run_id'],
                    'graph_id':obj.source,
                    'num_nodes':obj.num_nodes,
                    'num_edges':obj.num_edges,
                    'greedy_size':obj.greedy,
                    'optimal_size':obj.optimal,
                    'model_size':pruned_size,
                    'diff_greedy':diff_greedy,
                    'diff_optimal':diff_optimal
                    }
            rows.append(row)
            g_total+=diff_greedy
            dev_total+=dev
            if isinstance(diff_optimal, int):
                opt_list.append(diff_optimal)
        print()
        print(f"{mod_pkl} Greedy Mean Diff. : {(g_total/len(er_solutions)):.3f}" )
        print(f"{mod_pkl} Greedy Mean Dev. : {(100*dev_total/len(er_solutions)):.3f}%" )
    
    with open('IMDB_MODEL_PERFORMANCE.csv','w',newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__=="__main__":
    main()
