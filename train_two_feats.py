
"""
File: Train Model on two features
Description: This script trains a Q-network on the ZFS problem. Two features used are if a node is blue and if a node is already selected in the initial set. 
Usage:
    - Train the ML framework on a set of graph data
    - Networks are either introduced, or generated randomly for each episode.


Dependencies:
   The S2V framework is an external library that has light modifications to return full graph embedding
   Original S2V code: https://github.com/Hanjun-Dai/pytorch_structure2vec
   Modified code: external/s2v_lib/



"""





# ruff: noqa: F401
# ruff: noqa: F841
# ruff: noqa: E402
import argparse
import os
import time
import json
import pickle
import random
from dataclasses import dataclass,asdict
from typing import Any,Tuple
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from qnet import QNet, optimize_q_network, build_node_features
from transition import Transition, ReplayBuffer
from data.generator import generate_network
from ZFSEnv import ZFSEnv
#from MVCEnv import MVCEnv

def ensure_directory(path:str):
    os.makedirs(path,exist_ok=True)

def target_update(target,online):
    target.load_state_dict(online.state_dict())

def load_pickle(filepath):
    with open(filepath,'rb') as file:
        return pickle.load(file)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_name: str, graph) -> Any:
    env_name = env_name.lower().strip()
    #if env_name == "mvc":
    #    return MVCEnv(graph)
    if env_name == "zfs":
        return ZFSEnv(graph)
    raise ValueError(f"Unknown env '{env_name}'. Use --env mvc or --env zfs.")



# episolon different types of decay. 
def epsilon_decay(kind:str,step:int,epsilon_beg:float,epsilon_end:float,decay_steps:int,k: float = 10.0,beta: float = 0.9999)->float:
    if decay_steps <= 0:
        return epsilon_end

    t = min(step / decay_steps, 1.0)

    if kind == "linear":
        return epsilon_beg + t * (epsilon_end - epsilon_beg)

    elif kind == "exp":
        decay_rate = -5 
        normalized = (1 - np.exp(decay_rate * t)) / (1 - np.exp(decay_rate))
        return epsilon_beg + normalized * (epsilon_end - epsilon_beg)

    elif kind == "inverse":
        f = 1.0 / (1.0 + k * t)
        f0 = 1.0
        fT = 1.0 / (1.0 + k)
        normalized = (f - fT) / (f0 - fT)
        return epsilon_end + (epsilon_beg - epsilon_end) * normalized

    else:
        raise ValueError(f"Unknown epsilon decay kind: {kind}")



# during traing an n-step transition is built to contain last n-steps that occurred in the environment to later be used to optimize NN
def build_n_step_transition(n_step_buffer, gamma, max_n_steps):
    first = n_step_buffer[0]
    graph=first[0]
    state=first[1]
    node_features=first[2]
    action=first[3]

    cumulative=0.0
    discount=1.0
    n_steps = 0

    next_state = None 
    next_node_features = None
    done = False

    for j,step in enumerate(n_step_buffer):
        if j>= max_n_steps:
            break

        reward=step[4]
        cumulative += discount*reward
        discount*=gamma

        next_state=step[5]
        next_node_features=step[6]
        done=step[7]

        n_steps+=1

        if done:
            break
        
    t = Transition(graph=graph,
                   state=state,
                   node_features=node_features,
                   action=action,
                   reward=cumulative,
                   n_steps=n_steps,
                   next_state=next_state,
                   next_node_features=next_node_features,
                   done=done)

    return t

@torch.no_grad()
def select_action(qnet,graph,state,node_feats,eps,device:torch.device)->int:
    if state.ndim != 1:
        print(f"[WARNING] state should be 1D but got shape {state.shape}")

    legal = np.where(state==0)[0]
    if len(legal)==0:
        #raise Exception("Selecting an action from illegal list")
        return -1

    if random.random()<eps:
        return int(np.random.choice(legal))

    x=build_node_features(node_feats,device)
    q_scores, _ = qnet([graph],x)
    q=q_scores
    
    legal_mask = torch.as_tensor(state==0,device=device)
    neg_inf = torch.finfo(q.dtype).min
    #q_score of illegal action becomes -infinity so it won't get selected in argmax
    #needed because there can be negative q-scores for legal actions, i.e [ 0.20, -3.4, 0.10, 0.50] -->
    # ->[ 0.20, -3.4e38, 0.10, 0.50]
    q_masked = torch.where(legal_mask,q,neg_inf)

    return int(torch.argmax(q_masked).item())
 
def sample_small_biased_int(low: int, high: int) -> int:
    """Bias toward smaller integers in [low, high]."""
    if low is None or high is None:
        raise ValueError("low/high cannot be None")
    if low > high:
        raise ValueError(f"Invalid range: {low} > {high}")
    u = random.random()
    # squaring keeps more mass near 0 -> smaller values
    return low + int((high - low) * (u ** 2))


def sample_small_biased_float(low: float, high: float) -> float:
    """Bias toward smaller floats in [low, high]."""
    if low is None or high is None:
        raise ValueError("low/high cannot be None")
    if low > high:
        raise ValueError(f"Invalid range: {low} > {high}")
    u = random.random()
    return low + (high - low) * (u ** 2)


def sample_graph_params(args) -> tuple[int, int | None, float | None]:
    """
    Returns (n, m, p), biased toward smaller values.
    """

    n = sample_small_biased_int(args.n_min, args.n_max)

    m = None

    if args.network_type in {"ba", "powerlaw_cluster"}:
        m_low = args.m_min if args.m_min is not None else args.m
        m_high = args.m_max if args.m_max is not None else args.m
        m = sample_small_biased_int(m_low, m_high)

    p = None
    if args.network_type in {"er", "powerlaw_cluster"}:
        p_low = args.p_min
        p_high = args.p_max
        p = sample_small_biased_float(p_low, p_high)
    else: 
        p=0.2

    return n, m, p



#Create a graph for training based on user input
def make_training_graph(args,graph_list=None):

    if args.network_type == 'real':
       assert graph_list is not None
       sz=len(graph_list) 
       rn = random.randrange(sz)
       return graph_list[rn]


    n, m, p = sample_graph_params(args)

    if args.network_type == "er":
        assert p is not None
        return generate_network(p=p, n=n, connected=True, kind=args.network_type,save_graph_nx=True)

    elif args.network_type == "ba":
        return generate_network(m=m, n=n, connected=True, kind=args.network_type,save_graph_nx=True)

    elif args.network_type == "powerlaw_cluster":
        assert p is not None
        return generate_network(m=m, n=n, p=p, connected=True, kind=args.network_type)

    else:
        raise ValueError(f"Unknown network type: {args.network_type}")


def create_run_id(args)->str:
    parts = [
        args.env,
        f"seed{args.seed}",
        f"n{args.n}",
        f"p{args.p}",
        f"networktype{args.network_type}",
        f"doubleDQN{args.double_dqn}",
        f"nsteps{args.n_steps}",
        f"lr{args.lr}",
        f"gamma{args.gamma}",
        f"bs{args.batch_size}",
        f"eps{args.eps_start}-{args.eps_end}-dec{args.eps_decay_steps}",
        f"tgt{args.target_update}",
        f"steps{args.train_steps}",
        f"hid{args.hidden_dim}",
        f"lat{args.latent_dim}",
        f"out{args.out_dim}",
        f"lv{args.max_lv}",
    ]
    return "__".join(parts)
    
@dataclass
class Result:
    run_id:str
    args:dict
    start:float
    end:float
    losses:list
    episode_returns:list
    episode_lengths:list
    steps:int

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True, choices=["mvc", "zfs"])
    ap.add_argument("--network-type",type=str,choices = ['er','ba','powerlaw_cluster','mixed','real'],default='er')
    ap.add_argument("--network-fname",type=str,default=None)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-min",type=int,default=20)
    ap.add_argument("--n-max",type=int,default=50)
    ap.add_argument("--p", type=float, default=0.20)
    ap.add_argument("--p-min", type=float, default=0.20)
    ap.add_argument("--p-max", type=float, default=0.20)
    ap.add_argument("--m", type=int,default=None)
    ap.add_argument("--m-min", type=int,default=None) 
    ap.add_argument("--m-max", type=int,default=None)

    # model params
    ap.add_argument("--latent-dim", dest="latent_dim", type=int, default=64)
    ap.add_argument("--out-dim", dest="out_dim", type=int, default=64)
    ap.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
    ap.add_argument("--max-lv", dest="max_lv", type=int, default=3)

    # features 
    ap.add_argument("--num-node-feats", dest="num_node_feats", type=int, default=1)

    
    # RL params
    ap.add_argument("--load-params",action="store_true")
    ap.add_argument("--ptrained-modelpath",type=str,default="")
    ap.add_argument("--train-steps", type=int, default=20_000)
    ap.add_argument("--episode-horizon", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-steps",type=int,default=1)
    ap.add_argument("--replay-capacity", type=int, default=100_000)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--double-dqn",action="store_true")
    ap.add_argument("-msg_average", type=int, default=0)

    # epsilon schedule
    ap.add_argument("--eps-schedule", type=str, default='inverse')
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-steps", type=int, default=10000)

    # output
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)

    return ap.parse_args()
def train(args):
    device=torch.device(args.device)
    set_seed(args.seed)

    policy = QNet(
        latent_dim=args.latent_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        num_node_feats=args.num_node_feats,
        max_lv=args.max_lv,
            ).to(device)
    target = QNet(
        latent_dim=args.latent_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        num_node_feats=args.num_node_feats,
        max_lv=args.max_lv,
            ).to(device)

    if args.load_params:
        policy.load_state_dict(torch.load(args.ptrained_modelpath, map_location=device)) 
        target.load_state_dict(torch.load(args.ptrained_modelpath, map_location=device))
    else: 
        target_update(target,policy)


    optimizer = optim.Adam(policy.parameters(),lr=args.lr)
    replay = ReplayBuffer(capacity=args.replay_capacity)

    losses=[]
    episode_returns=[]
    #how long is takes for episode to reach terminal
    episode_lengths=[]

    run_id = args.run_id or create_run_id(args)
    t0=time.time()

    #ep1
    training_graphs = None
    if args.network_type=='real':
        if args.network_fname ==None:
            raise ValueError("no valid filepath for training graphs")
        training_graphs = load_pickle(args.network_fname)
        graph=make_training_graph(args,training_graphs)
    else:
        graph = make_training_graph(args)

    env = make_env(args.env,graph)
    state,features=env.reset()
    features = features[:, :2] #drop clustering coef.
    n_step_buffer= deque()
    episode_return=0.0
    episode_length=0
    num_episodes=0
    for step in tqdm(range(args.train_steps), desc="Training"):
        epsilon = epsilon_decay(
        kind=args.eps_schedule,
        step=step,
        epsilon_beg=args.eps_start,
        epsilon_end=args.eps_end,
        decay_steps=args.eps_decay_steps
        )
        action = select_action(qnet=policy,graph=env.g,state=state,node_feats=features,eps=epsilon,device=device)

        if action==-1:
            print("action=-1 triggered.")
            next_state=state.copy()
            next_features=features.copy()
            reward=0.0
            done=True
        else:
            next_state, next_features, reward, done, _= env.step(action)
            next_features = next_features[:, :2]   # drop clust coef
        #for n-step we are storing the multiple steps before calculating loss between target/actual in backpropagation
        raw_step = (
                env.g,
                state,
                features,
                int(action if action !=-1 else 0),
                float(reward),
                next_state,
                next_features,
                bool(done)
            )
        n_step_buffer.append(raw_step)
        
        if len(n_step_buffer) >= args.n_steps:
            tr_n = build_n_step_transition(n_step_buffer,args.gamma,args.n_steps)
            replay.push(tr_n)
            n_step_buffer.popleft()

        loss = optimize_q_network(
            policy_qnet=policy,
            target_qnet=target,
            optimizer=optimizer,
            replay=replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            double=args.double_dqn,
            device=device,
        )
        if loss!=0:
            losses.append(loss)

        if (step+1)%args.target_update ==0: 
            target_update(target=target,online=policy)
        episode_return+=reward
        episode_length+=1
        state=next_state
        features=next_features

        if done or episode_length>=args.episode_horizon:
            while n_step_buffer:
                tr_n = build_n_step_transition(n_step_buffer,args.gamma,args.n_steps)
                replay.push(tr_n)
                n_step_buffer.popleft()


            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            num_episodes+=1

            if num_episodes %5 ==0:
                if args.network_type=='real':
                    graph = make_training_graph(args,training_graphs)
                else:
                    graph = make_training_graph(args)

                env = make_env(args.env,graph)
            

            state,features=env.reset() 
            features = features[:, :2] # drop clust coef.

            n_step_buffer=deque()
            episode_return=0.0
            episode_length=0
        
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            last_ret = (
                sum(episode_returns[-10:]) / len(episode_returns[-10:])
                if episode_returns else None
            )
            last_loss = (
                sum(losses[-10:]) / len(losses[-10:])
                if losses else None
            )
            print(
                f"step {step+1}/{args.train_steps} eps={epsilon:.3f} "
                f"episodes={len(episode_returns)} last_ret={last_ret} last_loss={last_loss}"
            )            


    #end for loop and trainiing
    t1=time.time()
    
    result = Result(
        run_id=run_id,
        args=vars(args),
        start=t0,
        end=t1,
        losses=losses,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        steps=args.train_steps
    )

    print("Model trained: ",run_id)
    return result, policy

def main():
    args=parse_args()
    ensure_directory(args.outdir)
    #device = torch.device("cpu")
    results,qnet = train(args)
    
    pkl_path = os.path.join(args.outdir, f'{results.run_id}.pkl')
    with open(pkl_path,'wb') as file:
        pickle.dump(asdict(results),file)

    if args.save_weights:
        pt_path = os.path.join(args.outdir, f'{results.run_id}.pt')
        torch.save(qnet.state_dict(),pt_path)
    
    summary_path = os.path.join(args.outdir, f"{results.run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": results.run_id,
                "seconds": results.end - results.start,
                "num_losses": len(results.losses),
                "num_episodes": len(results.episode_returns),
                "last_return": results.episode_returns[-1] if results.episode_returns else None,
                "n_steps":args.n_steps,
                "double_DQN":args.double_dqn
            },
            f,
            indent=2,
        )

    print("Saved:", pkl_path)

if __name__ =='__main__':
    main()
