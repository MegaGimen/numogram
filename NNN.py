"""Numogram Neural Network Runner (NNN.py).

Dedicated runner for the Numogram recurrent neural network (NNN):
  - Recurrence execution and trajectory simulation.
  - Backpropagation Through Time (BPTT) unrolled step gradient tracking.
  - Jacobian spectral radius computation rho(J).
  - Forward and backward group-wise dynamics across the Warp, Torque, and Plex syzygy systems.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from numogram import Numogram, find_paths, numogram_to_directed_graph, to_directed_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numogram Neural Network Runner (NNN)")
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
    sample_paths = model.find_paths(start=2, end=6, n=2)
    print(f"\n=== Graph Conversion & Path Finding (Sample: Zone 2 -> Zone 6, length 2) ===")
    print(f"Total graph nodes: {len(graph)}")
    print(f"Paths found: {sample_paths}")
