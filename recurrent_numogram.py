"""Recurrent Numogram Neural Network (RNN-style Numogram).

Architecture:
  - 10 nodes (Zones 0..9), each carrying a state tensor x_i in R^d (default d=1).
  - Syzygy (对子互映射): 5 pairs (9::0, 8::1, 7::2, 6::3, 5::4).
    Each pair (u, v) has two d x d linear mappings: u -> v and v -> u.
  - Currents (激流): 5 syzygies mapped to their tractors:
    (5, 4) -> 1, (8, 1) -> 7, (7, 2) -> 5, (6, 3) -> 3, (9, 0) -> 9.
    Concatenates [x_u; x_v] (2d) -> Linear -> d, incoming to tractor node t.
  - Gates (门控通道): 9 canonical triangular gates:
    SwiGLU / MoE-style gating:
      A = sigmoid(W_A x_src + b_A)  (score in R^h)
      B = W_B x_src + b_B           (candidate in R^h)
      M = A * B                     (element-wise product)
      out = W_out M + b_out         (projected to R^d, incoming to dst)
  - Synchronous aggregation: Each node's next state is the sum of all incoming
    inputs this round.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NumogramGate(nn.Module):
    """Qwen / SwiGLU-style gated projection from d -> h -> d."""

    def __init__(self, d: int = 1, h: int = 4) -> None:
        super().__init__()
        self.fc_gate = nn.Linear(d, h)  # A (sigmoid score)
        self.fc_val = nn.Linear(d, h)   # B (linear candidate)
        self.fc_out = nn.Linear(h, d)   # back to node dimension d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        a = torch.sigmoid(self.fc_gate(x))
        b = self.fc_val(x)
        m = a * b
        return self.fc_out(m)


class NumogramCurrent(nn.Module):
    """Current connection: concatenates [x_u; x_v] and projects 2d -> d."""

    def __init__(self, d: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * d, d)

    def forward(self, x_u: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        # Concatenate on the feature dimension
        cat = torch.cat([x_u, x_v], dim=-1)
        return self.fc(cat)


class RecurrentNumogram(nn.Module):
    """Recurrent Graph Network over the canonical Numogram topology."""

    # 5 Syzygy pairs (sum = 9)
    SYZYGIES = [
        (9, 0),
        (8, 1),
        (7, 2),
        (6, 3),
        (5, 4),
    ]

    # 5 Currents: (pair_u, pair_v) -> tractor
    CURRENTS = [
        ((5, 4), 1),
        ((8, 1), 7),
        ((7, 2), 5),
        ((6, 3), 3),
        ((9, 0), 9),
    ]

    # 9 Canonical triangular gates: src -> dst (via gate id)
    # Gt-01: 1->1, Gt-03: 2->3, Gt-06: 3->6, Gt-10: 4->1, Gt-15: 5->6,
    # Gt-21: 6->3, Gt-28: 7->1, Gt-36: 8->9, Gt-45: 9->9
    GATES = [
        (1, 1),  # Gt-01
        (2, 3),  # Gt-03
        (3, 6),  # Gt-06
        (4, 1),  # Gt-10
        (5, 6),  # Gt-15
        (6, 3),  # Gt-21
        (7, 1),  # Gt-28
        (8, 9),  # Gt-36
        (9, 9),  # Gt-45
    ]

    def __init__(self, d: int = 1, h: int = 4) -> None:
        super().__init__()
        self.d = d
        self.h = h

        # 1. Syzygy mutual linear mappings: u -> v and v -> u (10 linear layers)
        self.syz_layers = nn.ModuleDict()
        for u, v in self.SYZYGIES:
            self.syz_layers[f"{u}_{v}"] = nn.Linear(d, d, bias=False)
            self.syz_layers[f"{v}_{u}"] = nn.Linear(d, d, bias=False)

        # 2. Current layers: (u, v) -> tractor (5 linear layers)
        self.curr_layers = nn.ModuleList([
            NumogramCurrent(d=d) for _ in self.CURRENTS
        ])

        # 3. Gate layers: src -> dst (9 GLU-style gated modules)
        self.gate_layers = nn.ModuleList([
            NumogramGate(d=d, h=h) for _ in self.GATES
        ])

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One synchronous update step.

        Args:
            x: Node states of shape [..., 10, d] (or [10, d]).

        Returns:
            x_next: Updated node states of shape [..., 10, d].
        """
        # Prepare accumulator for the 10 incoming sums
        # x_split: list of 10 tensors, each shape [..., d]
        x_nodes = [x[..., i, :] for i in range(10)]
        in_acc = [torch.zeros_like(x_nodes[i]) for i in range(10)]

        # --- A. Accumulate Syzygy mutual mappings ---
        for u, v in self.SYZYGIES:
            # u -> v
            in_acc[v] = in_acc[v] + self.syz_layers[f"{u}_{v}"](x_nodes[u])
            # v -> u
            in_acc[u] = in_acc[u] + self.syz_layers[f"{v}_{u}"](x_nodes[v])

        # --- B. Accumulate Currents ---
        for layer, ((u, v), tr) in zip(self.curr_layers, self.CURRENTS):
            in_acc[tr] = in_acc[tr] + layer(x_nodes[u], x_nodes[v])

        # --- C. Accumulate Gates ---
        for layer, (src, dst) in zip(self.gate_layers, self.GATES):
            in_acc[dst] = in_acc[dst] + layer(x_nodes[src])

        # Stack back into [..., 10, d]
        return torch.stack(in_acc, dim=-2)

    def forward(self, x0: torch.Tensor, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the Numogram recurrence for a fixed number of steps.

        Args:
            x0: Initial states [..., 10, d].
            steps: Number of recurrence steps T.

        Returns:
            x_final: State at step T, shape [..., 10, d].
            trajectory: All states [T+1, ..., 10, d].
        """
        history = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            history.append(x)
        trajectory = torch.stack(history, dim=0)
        return x, trajectory


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== Testing Recurrent Numogram (d=1, h=4) ===")
    model = RecurrentNumogram(d=1, h=4)

    # Initial input: batch of 2 samples, 10 zones, d=1
    x0 = torch.randn(2, 10, 1, requires_grad=True)

    # Run for 10 recurrence steps
    steps = 10
    x_final, trajectory = model(x0, steps=steps)

    print(f"Initial x0 shape:     {x0.shape}")
    print(f"Final x_final shape:  {x_final.shape}")
    print(f"Trajectory shape:     {trajectory.shape} (T+1, B, 10, d)")

    # Target sequence: 10 vectors of length d for the final state
    target = torch.randn_like(x_final)
    criterion = nn.MSELoss()
    loss = criterion(x_final, target)

    print(f"Loss: {loss.item():.6f}")

    # Backward pass to check gradient flow
    loss.backward()

    # Check gradients on all parameters
    all_grad_ok = True
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            all_grad_ok = False
            print(f"[FAIL] Param {name} has NO grad!")
        else:
            grad_norms[name] = param.grad.norm().item()

    print(f"Input x0.grad norm:    {x0.grad.norm().item():.4f}")
    print(f"Total model params:   {len(list(model.parameters()))}")
    print(f"All parameters have gradient: {all_grad_ok}")
    print("Sample gradient norms:")
    for k in list(grad_norms.keys())[:6]:
        print(f"  {k:30s} -> norm={grad_norms[k]:.4e}")
