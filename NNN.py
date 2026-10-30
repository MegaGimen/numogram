"""Numogram Neural Network Runner and Graph Analysis (NNN.py).

Provides:
  - Directed graph conversion from Numogram topology (Currents, Gates, Syzygies).
  - Arbitrary-length directed path search between zones.
  - Neural network recurrence, BPTT unrolled gradient tracking, and group state summaries.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn


def numogram_to_directed_graph(
    numogram: Any = 10,
    as_networkx: bool = False,
) -> dict[int, list[int]] | Any:
    """Converts a Numogram (or node count n) into a directed graph.

    Rules:
      1. Currents: If A::B -> C, directed edges A -> C and B -> C are added.
      2. Gates: Directly connected zones src -> dst add a directed edge src -> dst.
      3. Syzygies: Mutually connected zones A::B add bidirectional edges A -> B and B -> A.

    Args:
        numogram: A Numogram model instance or node count integer (default 10).
        as_networkx: If True, returns a networkx.DiGraph instance. Otherwise returns
            an adjacency dictionary dict[int, list[int]] with sorted neighbor lists.

    Returns:
        The directed graph representation.
    """
    if isinstance(numogram, int):
        from numogram import Numogram
        numogram = Numogram(n=numogram)

    n = numogram.n
    adj: dict[int, set[int]] = {i: set() for i in range(n)}

    # Rule 1: Currents (A::B -> C => A -> C and B -> C)
    for (u, v), tr in numogram.currents:
        adj[u].add(tr)
        adj[v].add(tr)

    # Rule 2: Gates (src -> dst)
    for src, dst in numogram.gates:
        adj[src].add(dst)

    # Rule 3: Syzygies (A::B => A -> B and B -> A)
    for u, v in numogram.syzygies:
        adj[u].add(v)
        adj[v].add(u)

    if as_networkx:
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            for u in range(n):
                for v in sorted(adj[u]):
                    G.add_edge(u, v)
            return G
        except ImportError:
            pass

    return {i: sorted(adj[i]) for i in range(n)}


# Alias for convenience
to_directed_graph = numogram_to_directed_graph


def _get_neighbors(graph: Any, node: int) -> list[int]:
    """Helper to retrieve neighbor nodes from dict or NetworkX DiGraph."""
    if hasattr(graph, "successors"):
        nbrs = graph.successors(node)
    elif hasattr(graph, "neighbors"):
        nbrs = graph.neighbors(node)
    elif isinstance(graph, dict):
        nbrs = graph.get(node, [])
    else:
        try:
            nbrs = graph[node]
        except Exception:
            nbrs = []
    if isinstance(nbrs, (set, tuple)):
        return sorted(nbrs)
    elif isinstance(nbrs, list):
        return nbrs
    return list(nbrs)


def find_paths(
    graph: Any,
    start: int,
    end: int,
    n: int,
    *,
    length: int | None = None,
) -> list[list[int]]:
    """Finds all paths of length n from start to end in the directed graph.

    Args:
        graph: Directed graph (adjacency dict or NetworkX DiGraph).
        start: Starting zone index.
        end: Ending zone index.
        n: Path length (number of directed edges / steps).
        length: Optional alias for n.

    Returns:
        List of paths, each being a list of node indices of length n + 1:
        [start, v_1, v_2, ..., end].
    """
    target_len = length if length is not None else n
    if target_len < 0:
        return []
    if target_len == 0:
        return [[start]] if start == end else []

    results: list[list[int]] = []
    stack: list[tuple[int, list[int], int]] = [(start, [start], target_len)]

    while stack:
        curr, path, remaining = stack.pop()
        if remaining == 1:
            for nxt in _get_neighbors(graph, curr):
                if nxt == end:
                    results.append(path + [nxt])
        else:
            neighbors = _get_neighbors(graph, curr)
            for nxt in reversed(neighbors):
                stack.append((nxt, path + [nxt], remaining - 1))

    return results


# Aliases for convenience
find_all_paths = find_paths
find_paths_of_length = find_paths


if __name__ == "__main__":
    from numogram import Numogram

    parser = argparse.ArgumentParser(description="Numogram Neural Network Runner")
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of zones/nodes (even integer >= 2, default: 10)",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    model = Numogram(n=args.n, d=1, h=4)

    # Initial input: batch of 2 samples, n zones, d=1
    features = torch.randn(2, model.n, 1, requires_grad=True)
    # Reserved for forward
    x = torch.randn(2, model.n, 1, requires_grad=True)
    # Target sequence: n vectors of length d for the final state
    labels = torch.randn_like(features)

    # Run for 100 recurrence steps with step gradient tracking enabled
    steps = 100
    x_final, trajectory = model(features, steps=steps, track_step_grads=True)

    criterion = nn.MSELoss()
    loss = criterion(x_final, labels)

    print(f"Loss: {loss.item():.6f}")

    # Backward pass: unrolled computation graph backpropagates through all steps
    loss.backward()

    # 1. Display unrolled gradient norms step-by-step
    model.print_step_grad_summary()

    # 2. Display backward group-wise average gradient norm flow
    model.print_group_backward_summary()

    # 3. Run forward pass with x and display top 3 active zones
    _, fwd_trajectory = model(x, steps=steps, track_step_grads=False)
    model.print_forward_summary(fwd_trajectory, top_k=3)

    # 4. Display forward group-wise average state norm evolution
    model.print_group_forward_summary(fwd_trajectory)

    # 5. Graph conversion & sample path finding
    graph = model.to_directed_graph()
    sample_paths = find_paths(graph, start=2, end=6, n=2)
    print(f"\n=== Graph Conversion & Path Finding (Sample: Zone 2 -> Zone 6, length 2) ===")
    print(f"Total graph nodes: {len(graph)}")
    print(f"Paths found: {sample_paths}")
