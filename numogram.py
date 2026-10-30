"""Numogram Neural Network Core Module (numogram.py).

Defines the Numogram recurrent neural network architecture:
  - 10 nodes (Zones 0..9) with state tensor x_i in R^d (default d=1).
  - Syzygy mutual linear mappings across complementary pairs.
  - Current feedforward projections to tractor nodes.
  - SwiGLU / MoE-style canonical triangular gates.
  - Recurrent zone updates with unrolled BPTT gradient tracking.
  - Integration with graph analysis (NNN.py) and demon ritual music (numogrammusic.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Import graph utilities from NNN.py
from NNN import (
    find_all_paths,
    find_paths,
    find_paths_of_length,
    numogram_to_directed_graph,
    to_directed_graph,
)

# Import music utilities from numogrammusic.py
from numogrammusic import (
    FREQ_MAX,
    FREQ_MIN,
    NUM_PITCH_INTERVALS,
    PITCH_LEVELS,
    NumogramError,
    NumogramMusicError,
    OccultError,
    SyzygeticError,
    calculate_demon_tempo,
    generate_ritual_audio,
    get_pitch_intervals,
    load_pandemonium,
    parse_ritual_frequencies,
    zone_to_frequency,
)


class NumogramGate(nn.Module):
    """Qwen / SwiGLU-style gated projection from d -> h -> d."""

    def __init__(self, d: int = 1, h: int = 4) -> None:
        super().__init__()
        self.fc_gate = nn.Linear(d, h)  # A (sigmoid score)
        self.fc_val = nn.Linear(d, h)   # B (linear candidate)
        self.fc_out = nn.Linear(h, d)   # back to node dimension d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        cat = torch.cat([x_u, x_v], dim=-1)
        return self.fc(cat)


class Numogram(nn.Module):
    """Graph Network over the dynamic Numogram topology."""

    def __init__(self, n: int = 10, d: int = 1, h: int = 4) -> None:
        """Initializes the Numogram neural network.

        Args:
            n: Number of zones/nodes (must be an even integer >= 2, default 10).
            d: Feature dimension per zone (default 1).
            h: Hidden dimension for gated projections (default 4).
        """
        super().__init__()
        if n < 2 or n % 2 != 0:
            raise ValueError(f"n must be an even integer >= 2, got {n}")
        self.n = n
        self.d = d
        self.h = h

        m = n - 1
        # Syzygies: n // 2 pairs (u, v) summing to m
        self.syzygies = [(m - u, u) for u in range(n // 2)]
        # Currents: syzygy (u, v) -> tractor |u - v|
        self.currents = [((u, v), abs(u - v)) for (u, v) in self.syzygies]
        # Gates: canonical triangular gate mappings k -> T_k mod m (in 1..m)
        self.gates = []
        for k in range(1, n):
            t_k = (k * (k + 1)) // 2
            rem = t_k % m
            dst = m if rem == 0 else rem
            self.gates.append((k, dst))
        # Warp/Torque/Plex grouping is only retained for classical n=10
        if n == 10:
            self.system_groups = {
                "Warp": [3, 6],
                "Torque": [1, 2, 4, 5, 7, 8],
                "Plex": [0, 9],
            }
        else:
            self.system_groups = {}

        # 1. Syzygy mutual linear mappings: u -> v and v -> u
        self.syz_layers = nn.ModuleDict()
        for u, v in self.syzygies:
            self.syz_layers[f"{u}_{v}"] = nn.Linear(d, d, bias=False)
            self.syz_layers[f"{v}_{u}"] = nn.Linear(d, d, bias=False)

        # 2. Current layers: (u, v) -> another zone
        self.curr_layers = nn.ModuleList([
            NumogramCurrent(d=d) for _ in self.currents
        ])

        # 3. Gate layers: src -> dst
        self.gate_layers = nn.ModuleList([
            NumogramGate(d=d, h=h) for _ in self.gates
        ])

        # 4. Zone Update Layer: [x_i; in_acc_i] (dim 2d) -> x_next (dim d)
        self.update_layer = nn.Linear(2 * d, d)

        # Container for unrolled state tensors during forward pass
        self._unrolled_states: list[torch.Tensor] = []

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One synchronous update step.

        Args:
            x: Node states of shape [..., n, d] (or [n, d]).

        Returns:
            x_next: Updated node states of shape [..., n, d].
        """
        x_nodes = [x[..., i, :] for i in range(self.n)]
        in_acc = [torch.zeros_like(x_nodes[i]) for i in range(self.n)]

        # --- A. Accumulate Syzygy mutual mappings ---
        for u, v in self.syzygies:
            in_acc[v] = in_acc[v] + self.syz_layers[f"{u}_{v}"](x_nodes[u])
            in_acc[u] = in_acc[u] + self.syz_layers[f"{v}_{u}"](x_nodes[v])

        # --- B. Accumulate Currents ---
        for layer, ((u, v), tr) in zip(self.curr_layers, self.currents):
            in_acc[tr] = in_acc[tr] + layer(x_nodes[u], x_nodes[v])

        # --- C. Accumulate Gates ---
        for layer, (src, dst) in zip(self.gate_layers, self.gates):
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
            x0: Initial states [..., n, d].
            steps: Number of recurrence steps T.
            track_step_grads: If True, calls retain_grad() on each unrolled step
                so that gradients at every step can be inspected after backward().

        Returns:
            x_final: State at step T, shape [..., n, d].
            trajectory: All states [T+1, ..., n, d].
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
            return self.step(v.view(1, self.n, -1)).view(-1)

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
              - 'zone_norms': list of n float gradient norms for Zones 0..n-1
        """
        records = []
        for t, s in enumerate(self._unrolled_states):
            if s.grad is None:
                continue
            total_norm = s.grad.norm().item()
            zone_norms = [s.grad[..., i, :].norm().item() for i in range(self.n)]
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
            trajectory: Tensor of shape [steps + 1, batch, n, d].
            top_k: Number of highest-magnitude zones to display (default 3).
            sample_idx: Index of batch sample to display (default 0).
        """
        print(f"\n=== Forward Step-by-Step Evolution (Sample {sample_idx}, Top {top_k} Zones) ===")
        print(f"{'Step':<6} | {'Total State Norm':<18} | Top Contributing Zones")
        print("-" * 65)
        for t in range(trajectory.shape[0]):
            state = trajectory[t, sample_idx]  # [n, d]
            tot_norm = state.norm().item()
            ranked = sorted(
                range(self.n),
                key=lambda i: state[i].norm().item(),
                reverse=True,
            )[:top_k]
            if state.shape[-1] == 1:
                top_str = ", ".join([f"Z{i}: {state[i, 0].item():.4f}" for i in ranked])
            else:
                top_str = ", ".join([f"Z{i}: {state[i].norm().item():.4f}" for i in ranked])
            print(f"t={t:<4} | {tot_norm:<18.6f} | {top_str}")

    def print_group_backward_summary(self) -> None:
        """Prints the step-by-step average gradient norm for the syzygy system groups (n=10 only)."""
        if not self.system_groups:
            print(f"\n[Note] Warp/Torque/Plex grouping is only defined for classical n=10 (current n={self.n}).")
            return
        records = self.get_step_grad_norms(compute_rho=False)
        if not records:
            print("No gradients available. Did you run loss.backward()?")
            return

        groups = list(self.system_groups.items())
        col_widths = [max(len(name) + 2, 14) for name, _ in groups]
        header = f"{'Step':<6} | " + " | ".join([f"{name:<{w}}" for (name, _), w in zip(groups, col_widths)])
        print("\n=== Backward Group-Wise Average Gradient Norm Flow ===")
        print(header)
        print("-" * len(header))
        for r in records:
            t = r["step"]
            zn = r["zone_norms"]
            vals = [
                f"{sum(zn[i] for i in nodes) / len(nodes):<{w}.6f}" if nodes else f"{0.0:<{w}.6f}"
                for (_, nodes), w in zip(groups, col_widths)
            ]
            print(f"t={t:<4} | " + " | ".join(vals))

    def print_group_forward_summary(self, trajectory: torch.Tensor, sample_idx: int = 0) -> None:
        """Prints the step-by-step average state norm for the syzygy system groups (n=10 only)."""
        if not self.system_groups:
            print(f"\n[Note] Warp/Torque/Plex grouping is only defined for classical n=10 (current n={self.n}).")
            return
        groups = list(self.system_groups.items())
        col_widths = [max(len(name) + 2, 14) for name, _ in groups]
        header = f"{'Step':<6} | " + " | ".join([f"{name:<{w}}" for (name, _), w in zip(groups, col_widths)])
        print(f"\n=== Forward Group-Wise Average State Norm Evolution (Sample {sample_idx}) ===")
        print(header)
        print("-" * len(header))
        for t in range(trajectory.shape[0]):
            state = trajectory[t, sample_idx]  # [n, d]
            zn = [state[i].norm().item() for i in range(self.n)]
            vals = [
                f"{sum(zn[i] for i in nodes) / len(nodes):<{w}.6f}" if nodes else f"{0.0:<{w}.6f}"
                for (_, nodes), w in zip(groups, col_widths)
            ]
            print(f"t={t:<4} | " + " | ".join(vals))

    def to_directed_graph(self, as_networkx: bool = False) -> dict[int, list[int]] | Any:
        """Converts this Numogram instance into a directed graph."""
        return numogram_to_directed_graph(self, as_networkx=as_networkx)

    def find_paths(
        self,
        start: int,
        end: int,
        n: int,
        *,
        length: int | None = None,
    ) -> list[list[int]]:
        """Finds all paths of length n from start to end in this Numogram's graph."""
        graph = self.to_directed_graph(as_networkx=False)
        return find_paths(graph, start, end, n, length=length)

    def parse_ritual_frequencies(
        self,
        rituals: Any,
        *,
        route_id: str | None = None,
        f_min: float = 33.0,
        f_max: float = 3960.0,
        ana_is_high: bool = True,
        as_dict: bool = True,
        errors: str = "raise",
    ) -> Any:
        """Parses ritual routes into zone frequencies using this Numogram's order n."""
        return parse_ritual_frequencies(
            rituals,
            numogram_order=self.n,
            route_id=route_id,
            f_min=f_min,
            f_max=f_max,
            ana_is_high=ana_is_high,
            as_dict=as_dict,
            errors=errors,
        )

    def generate_ritual_audio(
        self,
        demons: list[str | int | dict[str, Any]] | None = None,
        output_path: str | Path = "ritual_audio.mp3",
        duration: float = 60.0,
        *,
        timbre: str = "buzzer",
        click_duration: float = 0.03,
        stagger_phase: bool = False,
        note_duration: float | None = None,
        use_netspan_tempo: bool = True,
        sample_rate: int = 44100,
        f_min: float = 33.0,
        f_max: float = 3960.0,
        ana_is_high: bool = True,
        stereo: bool = True,
        skip_syzygy_and_occult: bool = True,
    ) -> Path:
        """Generates an MP3 audio file with one looped ritual track per selected demon."""
        return generate_ritual_audio(
            demons=demons,
            output_path=output_path,
            duration=duration,
            timbre=timbre,
            click_duration=click_duration,
            stagger_phase=stagger_phase,
            numogram_order=self.n,
            note_duration=note_duration,
            use_netspan_tempo=use_netspan_tempo,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            ana_is_high=ana_is_high,
            stereo=stereo,
            skip_syzygy_and_occult=skip_syzygy_and_occult,
        )
