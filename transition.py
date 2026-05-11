
"""
File: Transition
Description: 
    - Transitions store each action taken in the problem along with information about the reward and current/next state and features
    - Replay Buffer stores all the transitions to later be sampled to back-propagate losses


Usage:
    Model Training
    ALl transitions are stored in the replay buffer and then sampled to calculate and back-propagte loss to optimize the S2V-DQN NN

"""


import numpy as np
#import networkx as nx
from dataclasses import dataclass
from collections import deque
import random
from typing import List



#note: the reward stored is actually an n-step reward, not the immediate reward
@dataclass
class Transition:
    graph: object
    state: np.ndarray
    node_features: np.ndarray
    action: int
    reward: float
    n_steps: int
    next_state: np.ndarray
    next_node_features: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, tr: Transition):
        self.buf.append(tr)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)

