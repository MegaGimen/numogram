""" Numogram Neural Network (RNN-style Numogram).

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
  - Synchronous aggregation: in_acc is the sum of all incoming inputs this round.
  - Zone Update Layer:
    Takes concatenation of current state x_i and accumulated input in_acc_i ([x; in_acc] in R^{2d})
    and projects through a neural network to produce next state x_i' in R^d.
  - Step-wise Unrolled Gradient Tracking:
    Retains gradients along the unrolled BPTT computation graph so users can inspect
    the gradient norm at each step t in [0, T], both overall and per-zone.
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


class Numogram(nn.Module):
    """ Graph Network over the canonical Numogram topology."""

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

    # 3 Syzygy System Groups (Warp, Torque, Plex)
    SYSTEM_GROUPS = {
        "Warp (3::6)": [3, 6],
        "Torque (4::5, 2::7, 1::8)": [1, 2, 4, 5, 7, 8],
        "Plex (9::0)": [0, 9],
    }

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

        # 4. Zone Update Layer: [x_i; in_acc_i] (dim 2d) -> x_next (dim d)
        self.update_layer = nn.Linear(2 * d, d)

        # Container for unrolled state tensors during forward pass
        self._unrolled_states: list[torch.Tensor] = []

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One synchronous update step.

        Args:
            x: Node states of shape [..., 10, d] (or [10, d]).

        Returns:
            x_next: Updated node states of shape [..., 10, d].
        """
        x_nodes = [x[..., i, :] for i in range(10)]
        in_acc = [torch.zeros_like(x_nodes[i]) for i in range(10)]

        # --- A. Accumulate Syzygy mutual mappings ---
        for u, v in self.SYZYGIES:
            in_acc[v] = in_acc[v] + self.syz_layers[f"{u}_{v}"](x_nodes[u])
            in_acc[u] = in_acc[u] + self.syz_layers[f"{v}_{u}"](x_nodes[v])

        # --- B. Accumulate Currents ---
        for layer, ((u, v), tr) in zip(self.curr_layers, self.CURRENTS):
            in_acc[tr] = in_acc[tr] + layer(x_nodes[u], x_nodes[v])

        # --- C. Accumulate Gates ---
        for layer, (src, dst) in zip(self.gate_layers, self.GATES):
            in_acc[dst] = in_acc[dst] + layer(x_nodes[src])

        in_acc_tensor = torch.stack(in_acc, dim=-2)

        # --- D. Update Layer: f([x; in_acc]) -> x_next ---
        cat = torch.cat([x, in_acc_tensor], dim=-1)
        return self.update_layer(cat)

    def forward(
        self,
        x0: torch.Tensor,
        steps: int = 10,
        track_step_grads: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the Numogram recurrence for a fixed number of steps.

        Args:
            x0: Initial states [..., 10, d].
            steps: Number of recurrence steps T.
            track_step_grads: If True, calls retain_grad() on each unrolled step
                so that gradients at every step can be inspected after backward().

        Returns:
            x_final: State at step T, shape [..., 10, d].
            trajectory: All states [T+1, ..., 10, d].
        """
        self._unrolled_states = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            self._unrolled_states.append(x)

        if track_step_grads:
            for s in self._unrolled_states:
                if s.requires_grad:
                    s.retain_grad()

        trajectory = torch.stack(self._unrolled_states, dim=0)
        return x, trajectory

    def compute_spectral_radius(self, x_step: torch.Tensor) -> float:
        """Computes the spectral radius rho(J) of the single-step Jacobian at state x_step."""
        if x_step.ndim > 2:
            x_item = x_step[0]
        else:
            x_item = x_step
        x_flat = x_item.reshape(-1)

        def step_flat(v: torch.Tensor) -> torch.Tensor:
            return self.step(v.view(1, 10, -1)).view(-1)

        J = torch.autograd.functional.jacobian(step_flat, x_flat)
        return torch.linalg.eigvals(J).abs().max().item()

    def get_step_grad_norms(self, compute_rho: bool = True) -> list[dict[str, object]]:
        """Extracts the gradient norm and Jacobian spectral radius for each unrolled step.

        Args:
            compute_rho: Whether to compute rho(J) for each step state.

        Returns:
            List of dicts per step t, each containing:
              - 'step': time step index t in [0, T]
              - 'rho': Jacobian spectral radius rho(J) at step t (if compute_rho=True)
              - 'total_norm': total gradient Frobenius norm at step t
              - 'zone_norms': list of 10 float gradient norms for Zones 0..9
        """
        records = []
        for t, s in enumerate(self._unrolled_states):
            if s.grad is None:
                continue
            total_norm = s.grad.norm().item()
            zone_norms = [s.grad[..., i, :].norm().item() for i in range(10)]
            rec: dict[str, object] = {
                "step": t,
                "total_norm": total_norm,
                "zone_norms": zone_norms,
            }
            if compute_rho:
                rec["rho"] = self.compute_spectral_radius(s.detach())
            records.append(rec)
        return records

    def print_step_grad_summary(self, compute_rho: bool = True) -> None:
        """Prints a readable table of step-unrolled gradient norms, rho(J), and top zones."""
        records = self.get_step_grad_norms(compute_rho=compute_rho)
        if not records:
            print("No gradients available. Did you run loss.backward()?")
            return

        print("\n=== Unrolled Step-by-Step Gradient Flow (BPTT) ===")
        if compute_rho:
            print(f"{'Step':<6} | {'rho(J)':<10} | {'Total Grad Norm':<16} | Top Contributing Zones")
            print("-" * 75)
            for r in records:
                t = r["step"]
                rho = r["rho"]
                tot = r["total_norm"]
                zone_ranking = sorted(
                    enumerate(r["zone_norms"]), key=lambda item: item[1], reverse=True
                )[:3]
                top_str = ", ".join([f"Z{i}: {n:.4f}" for i, n in zone_ranking])
                print(f"t={t:<4} | {rho:<10.6f} | {tot:<16.6f} | {top_str}")
        else:
            print(f"{'Step':<6} | {'Total Grad Norm':<16} | Top Contributing Zones")
            print("-" * 65)
            for r in records:
                t = r["step"]
                tot = r["total_norm"]
                zone_ranking = sorted(
                    enumerate(r["zone_norms"]), key=lambda item: item[1], reverse=True
                )[:3]
                top_str = ", ".join([f"Z{i}: {n:.4f}" for i, n in zone_ranking])
                print(f"t={t:<4} | {tot:<16.6f} | {top_str}")

    def print_forward_summary(
        self,
        trajectory: torch.Tensor,
        top_k: int = 3,
        sample_idx: int = 0,
    ) -> None:
        """Prints the step-by-step state norm and top_k zones by magnitude.

        Args:
            trajectory: Tensor of shape [steps + 1, batch, 10, d].
            top_k: Number of highest-magnitude zones to display (default 3).
            sample_idx: Index of batch sample to display (default 0).
        """
        print(f"\n=== Forward Step-by-Step Evolution (Sample {sample_idx}, Top {top_k} Zones) ===")
        print(f"{'Step':<6} | {'Total State Norm':<18} | Top Contributing Zones")
        print("-" * 65)
        for t in range(trajectory.shape[0]):
            state = trajectory[t, sample_idx]  # [10, d]
            tot_norm = state.norm().item()
            # Rank zones by magnitude/L2-norm descending
            ranked = sorted(
                range(10),
                key=lambda i: state[i].norm().item(),
                reverse=True,
            )[:top_k]
            if state.shape[-1] == 1:
                top_str = ", ".join([f"Z{i}: {state[i, 0].item():.4f}" for i in ranked])
            else:
                top_str = ", ".join([f"Z{i}: {state[i].norm().item():.4f}" for i in ranked])
            print(f"t={t:<4} | {tot_norm:<18.6f} | {top_str}")

    def print_group_backward_summary(self) -> None:
        """Prints the step-by-step average gradient norm for the three syzygy system groups."""
        records = self.get_step_grad_norms(compute_rho=False)
        if not records:
            print("No gradients available. Did you run loss.backward()?")
            return

        print("\n=== Backward Group-Wise Average Gradient Norm Flow ===")
        print(f"{'Step':<6} | {'Warp (3::6)':<14} | {'Torque (4::5, 2::7, 1::8)':<26} | {'Plex (9::0)':<14}")
        print("-" * 70)
        for r in records:
            t = r["step"]
            zn = r["zone_norms"]
            w_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Warp (3::6)"]) / len(self.SYSTEM_GROUPS["Warp (3::6)"])
            t_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Torque (4::5, 2::7, 1::8)"]) / len(self.SYSTEM_GROUPS["Torque (4::5, 2::7, 1::8)"])
            p_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Plex (9::0)"]) / len(self.SYSTEM_GROUPS["Plex (9::0)"])
            print(f"t={t:<4} | {w_avg:<14.6f} | {t_avg:<26.6f} | {p_avg:<14.6f}")

    def print_group_forward_summary(self, trajectory: torch.Tensor, sample_idx: int = 0) -> None:
        """Prints the step-by-step average state norm for the three syzygy system groups."""
        print(f"\n=== Forward Group-Wise Average State Norm Evolution (Sample {sample_idx}) ===")
        print(f"{'Step':<6} | {'Warp (3::6)':<14} | {'Torque (4::5, 2::7, 1::8)':<26} | {'Plex (9::0)':<14}")
        print("-" * 70)
        for t in range(trajectory.shape[0]):
            state = trajectory[t, sample_idx]  # [10, d]
            zn = [state[i].norm().item() for i in range(10)]
            w_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Warp (3::6)"]) / len(self.SYSTEM_GROUPS["Warp (3::6)"])
            t_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Torque (4::5, 2::7, 1::8)"]) / len(self.SYSTEM_GROUPS["Torque (4::5, 2::7, 1::8)"])
            p_avg = sum(zn[i] for i in self.SYSTEM_GROUPS["Plex (9::0)"]) / len(self.SYSTEM_GROUPS["Plex (9::0)"])
            print(f"t={t:<4} | {w_avg:<14.6f} | {t_avg:<26.6f} | {p_avg:<14.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    model = Numogram(d=1, h=4)

    # Initial input: batch of 2 samples, 10 zones, d=1
    features = torch.randn(2, 10, 1, requires_grad=True)
    # Reserved for forward
    x = torch.randn(2, 10, 1, requires_grad=True)
    # Target sequence: 10 vectors of length d for the final state
    labels = torch.randn_like(features)
    
    # Run for 10 recurrence steps with step gradient tracking enabled
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
