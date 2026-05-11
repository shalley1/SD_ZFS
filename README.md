#  SD-ZFS (Structure2Vec-DQN-ZFS)

> Reinforcement learning framework for the Zero Forcing Set problem using Structure2Vec and Deep Q-Learning.

---

## Overview

The purpose of this project is to adapt the s2vdqn deep reinforement learning framework to the problem of finding a minimum zero-forcing set. 
The framework is adapated to an environment where nodes are iteratively added to an initial set until the initial set forces the entire network under the zero-forcing constraint.
Multiple models are included:

- Model A: Trained on Erdos-Renyi Networks
- Model B: Trained on BArabasi-Albert networks
- Model C: trained on real-world facebook networks


The framework is tested against optimal solutions found through the wavefront algorithm and greedy solutions found through the greedy heuristic that prioritizes single node maximum closure.

---

## Installation

```bash
git clone https://github.com/shalley1/SD_ZFS.git
cd SD_ZFS
cd requirements
conda env create -f environment.yml
conda activate zfs

cd ../external/s2v_lib
make -j4
```

---

## Usage Examples

### Training

```bash
./train_example.sh
```

### Evaluation

```bash
python -m analysis.test
```

---

## Project Structure

```text
project/
├── train_three_feats.py
├── train_two_feats.py
├── qnet.py
├── ZFSEnv.py
├── data/
├── analysis/
├── models/
└── README.md
```

---


### References

External library for S2V framework:
    https://github.com/Hanjun-Dai/pytorch_structure2vec

Datasets
    https://github.com/OUAhmad/Zero-Forcing-Set-GML

