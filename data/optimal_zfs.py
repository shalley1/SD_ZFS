"""
optimal_zfs.py

Description: 
    -compute greedy ZFS size based on single node largest closure
    -compute optimal ZFS size using wavefront algorithm       

"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


# ZFS propagation (closure)

def zfs_closure(G: nx.Graph, S: Iterable[int]) -> Set[int]:
    blue: Set[int] = set(S)

    # white_deg[v] = number of neighbors of v that are currently white
    white_deg = {}
    for v in G.nodes():
        cnt = 0
        for u in G.neighbors(v):
            if u not in blue:
                cnt += 1
        white_deg[v] = cnt

    q = deque([v for v in G.nodes() if v in blue])

    while q:
        v = q.popleft()

        # only blue nodes with exactly one white neighbor can force
        if v not in blue:
            continue
        if white_deg[v] != 1:
            continue

        # find the unique white neighbor
        u_force = None
        for u in G.neighbors(v):
            if u not in blue:
                u_force = u
                break
        if u_force is None:
            continue

        # force it blue
        blue.add(u_force)

        # compute white_deg for newly blue node
        cnt = 0
        for x in G.neighbors(u_force):
            if x not in blue:
                cnt += 1
        white_deg[u_force] = cnt

        # newly blue node is no longer white for its blue neighbors
        for w in G.neighbors(u_force):
            if w in blue:
                white_deg[w] -= 1
                q.append(w)

        # newly blue node may now be able to force
        q.append(u_force)

    return blue


def is_zfs(G: nx.Graph, S: Sequence[int]) -> bool:
    """Verify S is a zero forcing set."""
    return len(zfs_closure(G, S)) == G.number_of_nodes()


# Greedy heuristic
def zfs_greedy_gain(G: nx.Graph) -> Tuple[int, List[int]]:
    nodes = list(G.nodes())
    S: Set[int] = set()
    blue: Set[int] = set()

    n = G.number_of_nodes()
    while len(blue) < n:
        best_v = None
        best_gain = -1
        best_blue = None

        for v in nodes:
            if v in S:
                continue
            cand_blue = zfs_closure(G, S | {v})
            gain = len(cand_blue) - len(blue)
            if gain > best_gain:
                best_gain = gain
                best_v = v
                best_blue = cand_blue

        if best_v is None:
            remaining = [v for v in nodes if v not in S]
            best_v = remaining[0]
            best_blue = zfs_closure(G, S | {best_v})

        S.add(best_v)
        blue = best_blue  

    sol = sorted(S)
    return len(sol), sol



def zfs_optimal_wavefront(
    G: nx.Graph,
    *,
    verify_solution: bool = True,
) -> Optional[Tuple[str, int, List[int]]]:
    nodes = list(G.nodes())
    all_blue = frozenset(nodes)

    start_closure = frozenset()
    start_seeds = frozenset()

    # best known cost for each closure
    best_cost = {start_closure: 0}
    best_seeds = {start_closure: start_seeds}

    q = deque([start_closure])

    best_full_cost = None
    best_full_seeds = None

    while q:
        S = q.popleft()
        r = best_cost[S]
        seed_set = set(best_seeds[S])
        S_set = set(S)

        # prune if we already have a better complete solution
        if best_full_cost is not None and r >= best_full_cost:
            continue

        if S == all_blue:
            if best_full_cost is None or r < best_full_cost:
                best_full_cost = r
                best_full_seeds = frozenset(seed_set)
            continue

        for v in nodes:
            base_seeds = set(seed_set)

            # add v if it is not already in the closure
            if v not in S_set:
                base_seeds.add(v)

            outside_neighbors = [u for u in G.neighbors(v) if u not in S_set]

            candidate_seed_sets = []

            if len(outside_neighbors) <= 1:
                # v already has at most one neighbor outside S,
                # so no extra neighbor seeds are needed
                candidate_seed_sets.append(base_seeds)
            else:
                # leave exactly one outside neighbor unseeded so v can force it
                for leave_unseeded in outside_neighbors:
                    cand_seeds = set(base_seeds)
                    for u in outside_neighbors:
                        if u != leave_unseeded:
                            cand_seeds.add(u)
                    candidate_seed_sets.append(cand_seeds)

            for cand_seeds in candidate_seed_sets:
                cand_cost = len(cand_seeds)

                if best_full_cost is not None and cand_cost >= best_full_cost:
                    continue

                cand_closure = frozenset(zfs_closure(G, cand_seeds))

                old_cost = best_cost.get(cand_closure)
                if old_cost is not None and old_cost <= cand_cost:
                    continue

                best_cost[cand_closure] = cand_cost
                best_seeds[cand_closure] = frozenset(cand_seeds)
                q.append(cand_closure)

                if cand_closure == all_blue:
                    if best_full_cost is None or cand_cost < best_full_cost:
                        best_full_cost = cand_cost
                        best_full_seeds = frozenset(cand_seeds)

    if best_full_seeds is None:
        return None

    S = sorted(best_full_seeds)

    if verify_solution and not is_zfs(G, S):
        raise ValueError(f"Wavefront returned an invalid ZFS. solution={S}")

    return "OPTIMAL", len(S), S

def _ensure_integer_labels_0_to_n_minus_1(G: nx.Graph) -> nx.Graph:
    nodes = list(G.nodes())
    if nodes and set(nodes) == set(range(len(nodes))):
        return G
    return nx.convert_node_labels_to_integers(G, first_label=0)


def main():
    pass
    # try:
    #     from data.generator import generate_network
    #     g = generate_network(n=50, p=0.15, connected=True, save_graph_nx=True)
    #     G = g.nx_graph
    # except Exception:
    #     print("error")
    #     return

    # G = _ensure_integer_labels_0_to_n_minus_1(G)

    # # GREEDY
    # k_g, S_g = zfs_greedy_gain(G)
    # ok_g = is_zfs(G, S_g)
    # print("Greedy ZFS size:", k_g)
    # print("Greedy ZFS:", S_g)
    # print("Greedy verifies:", ok_g)
    # if not ok_g:
    #     print("WARNING: greedy set did not verify as a ZFS under zfs_closure().")


if __name__ == "__main__":
    main()
