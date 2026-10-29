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

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class NumogramError(Exception):
    """Base exception for Numogram domain errors."""
    pass


class NumogramMusicError(NumogramError):
    """Base exception for Numogram ritual frequency and music errors."""
    pass


class OccultError(NumogramMusicError):
    """Raised when encountering an unmapped/occult Xenodemon path ('?')."""

    def __init__(self, message: str = "Unmapped Xenodemon Abyss") -> None:
        super().__init__(message)


class SyzygeticError(NumogramMusicError):
    """Raised when encountering a syzygetic tangential feedback loop ('X')."""

    def __init__(self, message: str = "Tangential Feedback Loops") -> None:
        super().__init__(message)


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
            # Rank zones by magnitude/L2-norm descending
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
        note_duration: float = 0.5,
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
            numogram_order=self.n,
            note_duration=note_duration,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            ana_is_high=ana_is_high,
            stereo=stereo,
            skip_syzygy_and_occult=skip_syzygy_and_occult,
        )


def numogram_to_directed_graph(
    numogram: Numogram | int = 10,
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
        numogram = Numogram(n=numogram)
    elif not isinstance(numogram, Numogram):
        raise TypeError(f"Expected Numogram instance or int, got {type(numogram)}")

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


PITCH_LEVELS: list[str] = [
    "Ana-7",
    "Ana-6",
    "Ana-5",
    "Ana-4",
    "Ana-3",
    "Ana-2",
    "Ana-1",
    "Null",
    "Cth-1",
    "Cth-2",
    "Cth-3",
    "Cth-4",
    "Cth-5",
    "Cth-6",
    "Cth-7",
]

FREQ_MIN: float = 33.0
FREQ_MAX: float = 3960.0
NUM_PITCH_INTERVALS: int = 15

_PANDEMONIUM_CACHE: dict[str, Any] | None = None


def load_pandemonium(filepath: str | Path | None = None) -> dict[str, Any]:
    """Loads Pandemonium.json matrix data, caching in memory."""
    global _PANDEMONIUM_CACHE
    if filepath is None and _PANDEMONIUM_CACHE is not None:
        return _PANDEMONIUM_CACHE

    candidates: list[Path] = []
    if filepath is not None:
        candidates.append(Path(filepath))
    candidates.extend([
        Path(__file__).parent / "Pandemonium.json",
        Path("/home/numogram/Pandemonium.json"),
        Path.cwd() / "Pandemonium.json",
    ])

    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if filepath is None:
                    _PANDEMONIUM_CACHE = data
                return data

    raise FileNotFoundError("Could not locate Pandemonium.json in workspace.")


def get_pitch_intervals(
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
    ana_is_high: bool = True,
) -> dict[str, tuple[float, float]]:
    """Divides frequency range [f_min, f_max] into 15 pitch intervals.

    Args:
        f_min: Minimum frequency in Hz (default 33.0).
        f_max: Maximum frequency in Hz (default 3960.0).
        ana_is_high: If True (default), Ana-7 (highest pitch) maps to the highest
            frequency interval, and Cth-7 (lowest pitch) maps to the lowest.
            If False, follows literal textual index order (Ana-7 at lowest).

    Returns:
        Mapping from pitch string to (a, b) interval tuple.
    """
    step = (f_max - f_min) / float(NUM_PITCH_INTERVALS)
    intervals: dict[str, tuple[float, float]] = {}
    for i, p in enumerate(PITCH_LEVELS):
        idx = (NUM_PITCH_INTERVALS - 1 - i) if ana_is_high else i
        a = f_min + idx * step
        b = a + step
        intervals[p] = (a, b)
    return intervals


def zone_to_frequency(
    zone: int,
    pitch: str,
    n: int = 10,
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
    ana_is_high: bool = True,
) -> float:
    """Calculates the frequency of a zone within a pitch's interval [a, b].

    Formula:
        x = zone / n
        freq = a + (b - a) * x (i.e. position x within interval [a, b])

    Args:
        zone: Zone node ID (0 <= zone < n).
        pitch: Pitch descriptor (e.g. 'Ana-1', 'Null', 'Cth-7').
        n: Numogram order / node count (default 10).
        f_min: Minimum frequency (default 33.0 Hz).
        f_max: Maximum frequency (default 3960.0 Hz).
        ana_is_high: Pitch orientation (default True).

    Returns:
        Frequency in Hz as a float.
    """
    clean_pitch = pitch.split()[0] if pitch else "Null"
    intervals = get_pitch_intervals(f_min=f_min, f_max=f_max, ana_is_high=ana_is_high)
    if clean_pitch not in intervals:
        raise ValueError(f"Unknown pitch '{pitch}', expected one of {PITCH_LEVELS}")
    a, b = intervals[clean_pitch]
    x = zone / float(n)
    return a + (b - a) * x


def _find_demon(ident: str | int, demons: list[dict[str, Any]]) -> dict[str, Any]:
    """Finds a demon in Pandemonium by index, ID, name, or alias."""
    if isinstance(ident, int):
        if 0 <= ident < len(demons):
            return demons[ident]
        raise ValueError(f"Demon index {ident} out of range (0..{len(demons)-1})")
    ident_str = str(ident).strip().lower()
    for d in demons:
        m_id = d.get("id", "").lower()
        idx_str = str(d.get("index", ""))
        mesh_idx = f"mesh-{idx_str}"
        if ident_str in [m_id, idx_str, mesh_idx]:
            return d
        if d.get("name", "").lower() == ident_str:
            return d
        if any(a.lower() == ident_str for a in d.get("aliases", [])):
            return d
    raise ValueError(f"Demon '{ident}' not found in Pandemonium Matrix")


def _parse_single_route_frequencies(
    path: Any,
    pitch: str,
    n: int = 10,
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
    ana_is_high: bool = True,
) -> list[float]:
    """Parses a single route path into a list of zone frequencies."""
    if path == "?" or (isinstance(path, str) and "?" in path):
        raise OccultError("Unmapped Xenodemon Abyss")
    if path == "X" or (isinstance(path, str) and "X" in path.upper()):
        raise SyzygeticError("Tangential Feedback Loops")

    if isinstance(path, list):
        if any(item == "?" for item in path):
            raise OccultError("Unmapped Xenodemon Abyss")
        if any(str(item).upper() == "X" for item in path):
            raise SyzygeticError("Tangential Feedback Loops")
        nodes = [int(item) for item in path]
    elif isinstance(path, str):
        nodes = [int(c) for c in path if c.isdigit()]
    elif isinstance(path, int):
        nodes = [path]
    else:
        raise ValueError(f"Unsupported path format: {path}")

    return [
        zone_to_frequency(
            z,
            pitch=pitch,
            n=n,
            f_min=f_min,
            f_max=f_max,
            ana_is_high=ana_is_high,
        )
        for z in nodes
    ]


def parse_ritual_frequencies(
    rituals: Any,
    numogram_order: int = 10,
    *,
    route_id: str | None = None,
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
    ana_is_high: bool = True,
    as_dict: bool = True,
    errors: str = "raise",
) -> Any:
    """Parses Numogram ritual routes into zone frequencies.

    Uses pitch intervals divided into 15 ranges across [f_min, f_max] (33Hz to 3960Hz)
    from highest Ana-7 to lowest Cth-7.

    If a demon has multiple routes (e.g. Rt-1, Rt-2, Rt-3), all routes are parsed
    and returned separately in a dictionary mapping route ID to its note frequency list.

    Raises:
        OccultError: If the route is an unmapped Xenodemon hidden path ('?').
        SyzygeticError: If the route is a syzygetic tangential feedback loop ('X').

    Args:
        rituals: A demon identifier (e.g. 'Mesh-00', 'Lurgo', 0), a demon dict,
            a route dict, or a list of rituals.
        numogram_order: Order / zone count of Numogram (default 10).
        route_id: Optional route ID to filter (e.g. 'Rt-1').
        f_min: Minimum frequency (default 33.0 Hz).
        f_max: Maximum frequency (default 3960.0 Hz).
        ana_is_high: If True (default), Ana-7 is highest frequency (3960Hz)
            and Cth-7 is lowest (33Hz).
        as_dict: If True (default), returns dict of route IDs to frequency lists.
            If False, returns a list of frequency lists (or single list if 1 route).
        errors: Error handling mode ('raise', 'skip', 'store'). Default 'raise'.

    Returns:
        Frequencies of each path as dict or list of float frequencies.
    """
    if isinstance(rituals, list) and not all(isinstance(x, int) for x in rituals):
        return [
            parse_ritual_frequencies(
                r,
                numogram_order=numogram_order,
                route_id=route_id,
                f_min=f_min,
                f_max=f_max,
                ana_is_high=ana_is_high,
                as_dict=as_dict,
                errors=errors,
            )
            for r in rituals
        ]

    pitch = "Null"
    routes_to_parse: list[tuple[str, Any]] = []

    if isinstance(rituals, (str, int)):
        pandemonium = load_pandemonium()
        demon = _find_demon(rituals, pandemonium.get("demons", []))
        pitch_val = demon.get("pitch", {})
        pitch = pitch_val.get("value", "Null") if isinstance(pitch_val, dict) else str(pitch_val)
        all_routes = demon.get("routes", [])
        if route_id is not None:
            matched = [
                r for r in all_routes
                if r.get("id") == route_id or str(r.get("route_number")) == str(route_id)
            ]
            if not matched:
                raise ValueError(f"Route '{route_id}' not found for demon '{demon.get('name')}'")
            routes_to_parse = [(r.get("id", f"Rt-{i}"), r.get("path")) for i, r in enumerate(matched)]
        else:
            routes_to_parse = [(r.get("id", f"Rt-{i}"), r.get("path")) for i, r in enumerate(all_routes)]

    elif isinstance(rituals, dict):
        if "routes" in rituals:
            pitch_val = rituals.get("pitch", "Null")
            pitch = pitch_val.get("value", "Null") if isinstance(pitch_val, dict) else str(pitch_val)
            all_routes = rituals.get("routes", [])
            if route_id is not None:
                matched = [
                    r for r in all_routes
                    if r.get("id") == route_id or str(r.get("route_number")) == str(route_id)
                ]
                if not matched:
                    raise ValueError(f"Route '{route_id}' not found in ritual")
                routes_to_parse = [(r.get("id", f"Rt-{i}"), r.get("path")) for i, r in enumerate(matched)]
            else:
                routes_to_parse = [(r.get("id", f"Rt-{i}"), r.get("path")) for i, r in enumerate(all_routes)]
        elif "path" in rituals:
            pitch_val = rituals.get("pitch", "Null")
            pitch = pitch_val.get("value", "Null") if isinstance(pitch_val, dict) else str(pitch_val)
            rid = rituals.get("id", "Rt-0")
            routes_to_parse = [(rid, rituals.get("path"))]
        else:
            raise ValueError(f"Dict ritual must contain 'routes' or 'path', got keys: {list(rituals.keys())}")
    else:
        raise TypeError(f"Unsupported ritual input type: {type(rituals)}")

    route_results: dict[str, Any] = {}
    for rid, path in routes_to_parse:
        try:
            freqs = _parse_single_route_frequencies(
                path,
                pitch=pitch,
                n=numogram_order,
                f_min=f_min,
                f_max=f_max,
                ana_is_high=ana_is_high,
            )
            route_results[rid] = freqs
        except (OccultError, SyzygeticError) as exc:
            if errors == "raise":
                raise
            elif errors == "store":
                route_results[rid] = str(exc)
            elif errors == "skip":
                continue
            else:
                raise ValueError(f"Unknown errors mode '{errors}', expected 'raise', 'skip', or 'store'")

    if route_id is not None:
        if len(route_results) == 1:
            return list(route_results.values())[0]
        return route_results

    if not as_dict:
        if len(route_results) == 1:
            return list(route_results.values())[0]
        return list(route_results.values())

    return route_results


# Aliases for convenience
get_ritual_frequencies = parse_ritual_frequencies


def generate_ritual_audio(
    demons: list[str | int | dict[str, Any]] | None = None,
    output_path: str | Path = "ritual_audio.mp3",
    duration: float = 60.0,
    *,
    numogram_order: int = 10,
    note_duration: float = 0.5,
    sample_rate: int = 44100,
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
    ana_is_high: bool = True,
    stereo: bool = True,
    skip_syzygy_and_occult: bool = True,
) -> Path:
    """Generates an MP3 audio track combining looped rituals of selected demons.

    For each demon, all of its ritual routes are concatenated sequentially into a
    single melody loop. A separate track is generated for each demon by repeating
    its ritual loop across the specified duration. The n tracks are then mixed,
    normalized, and encoded as an MP3 file.

    If demons is None or empty:
        Defaults to all demons in the Pandemonium Matrix, automatically skipping
        syzygetic demons ([X]) and chaotic xenodemons with unmapped/occult paths ([?]).

    Args:
        demons: List of demon names, IDs (e.g. 'Mesh-00', 'Lurgo'), indices, or
            demon dicts. If None, defaults to all non-syzygetic, non-occult demons.
        output_path: Path where the resulting MP3 file will be written.
        duration: Audio playback duration in seconds (default 60.0).
        numogram_order: Order / zone count of Numogram (default 10).
        note_duration: Duration of each zone note in seconds (default 0.5).
        sample_rate: Audio sampling rate in Hz (default 44100).
        f_min: Minimum frequency in Hz (default 33.0).
        f_max: Maximum frequency in Hz (default 3960.0).
        ana_is_high: Pitch mapping direction (default True).
        stereo: If True (default), spreads demon tracks across stereo field.
        skip_syzygy_and_occult: If True (default), skips [X] and [?] routes.

    Returns:
        Path to the generated MP3 file.
    """
    pandemonium = load_pandemonium()
    all_demons = pandemonium.get("demons", [])

    selected_demons: list[dict[str, Any]] = []

    if demons is None or len(demons) == 0:
        for d in all_demons:
            is_syzygy = bool(d.get("syzygy")) or "Syzygetic" in d.get("demon_type", {}).get("order", "")
            has_occult = any(
                r.get("type") == "occult_path" or r.get("path_raw") == "?" or r.get("path") == "?"
                for r in d.get("routes", [])
            )
            has_x = any(
                r.get("type") == "syzygetic_rite" or r.get("path_raw") == "X" or r.get("path") == "X"
                for r in d.get("routes", [])
            )
            if skip_syzygy_and_occult and (is_syzygy or has_occult or has_x):
                continue
            selected_demons.append(d)
    else:
        for item in demons:
            if isinstance(item, dict):
                selected_demons.append(item)
            else:
                d = _find_demon(item, all_demons)
                selected_demons.append(d)

    if not selected_demons:
        raise ValueError("No valid demons selected for audio generation.")

    n_samples = max(1, int(round(duration * sample_rate)))
    note_len = max(1, int(round(note_duration * sample_rate)))

    # Attack, decay, and anti-click boundary fade envelope
    tt = np.arange(note_len, dtype=np.float32) / float(sample_rate)
    attack_len = min(int(0.005 * sample_rate), note_len)
    att = np.ones(note_len, dtype=np.float32)
    if attack_len > 0:
        att[:attack_len] = np.linspace(0.0, 1.0, attack_len, endpoint=False)
    decay = np.exp(-tt * 2.5, dtype=np.float32)
    env = att * decay
    fade_len = min(int(0.005 * sample_rate), note_len)
    if fade_len > 0:
        env[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, endpoint=False)

    master_l = np.zeros(n_samples, dtype=np.float32)
    master_r = np.zeros(n_samples, dtype=np.float32)

    n_demons = len(selected_demons)
    active_tracks = 0

    for idx, d in enumerate(selected_demons):
        pitch_val = d.get("pitch", {})
        pitch_str = pitch_val.get("value", "Null") if isinstance(pitch_val, dict) else str(pitch_val)

        # Concatenate all rituals for this demon into a single sequence
        demon_freqs: list[float] = []
        for rt in d.get("routes", []):
            path = rt.get("path")
            try:
                rt_freqs = _parse_single_route_frequencies(
                    path,
                    pitch=pitch_str,
                    n=numogram_order,
                    f_min=f_min,
                    f_max=f_max,
                    ana_is_high=ana_is_high,
                )
                demon_freqs.extend(rt_freqs)
            except (OccultError, SyzygeticError):
                if skip_syzygy_and_occult:
                    continue
                else:
                    raise

        if not demon_freqs:
            continue

        active_tracks += 1
        track = np.zeros(n_samples, dtype=np.float32)
        n_notes = int(np.ceil(n_samples / note_len))

        for k in range(n_notes):
            f = demon_freqs[k % len(demon_freqs)]
            two_pi_f = 2.0 * np.pi * f
            note = (
                np.sin(two_pi_f * tt)
                + 0.35 * np.sin(2.0 * two_pi_f * tt)
                + 0.12 * np.sin(3.0 * two_pi_f * tt)
            ) * env

            start_idx = k * note_len
            end_idx = min(start_idx + note_len, n_samples)
            seg_len = end_idx - start_idx
            track[start_idx:end_idx] += note[:seg_len]

        if stereo:
            pan = (idx / float(n_demons - 1)) if n_demons > 1 else 0.5
            w_l = float(np.cos(pan * (np.pi / 2.0)))
            w_r = float(np.sin(pan * (np.pi / 2.0)))
            master_l += track * w_l
            master_r += track * w_r
        else:
            master_l += track
            master_r += track

    if active_tracks == 0:
        raise ValueError("No active musical tracks were generated (selected demons lacked valid numeric routes).")

    # Normalize to -0.9 dBFS
    max_val = max(float(np.max(np.abs(master_l))), float(np.max(np.abs(master_r))), 1e-6)
    scale = 0.9 * 32767.0 / max_val
    left_pcm = (master_l * scale).astype(np.int16)
    right_pcm = (master_r * scale).astype(np.int16)

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if Path(ffmpeg_bin).is_file():
        if stereo:
            pcm_data = np.empty((n_samples, 2), dtype=np.int16)
            pcm_data[:, 0] = left_pcm
            pcm_data[:, 1] = right_pcm
            channels = "2"
        else:
            pcm_data = left_pcm
            channels = "1"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", channels,
            "-i", "pipe:0",
            "-b:a", "192k",
            str(out_p),
        ]
        proc = subprocess.run(
            cmd,
            input=pcm_data.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg MP3 encoding failed: {err_msg}")
    else:
        import wave
        wav_path = out_p.with_suffix(".wav")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2 if stereo else 1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if stereo:
                pcm_data = np.empty((n_samples, 2), dtype=np.int16)
                pcm_data[:, 0] = left_pcm
                pcm_data[:, 1] = right_pcm
                wf.writeframes(pcm_data.tobytes())
            else:
                wf.writeframes(left_pcm.tobytes())
        return wav_path

    return out_p


# Aliases for convenience
generate_numogram_audio = generate_ritual_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numogram Neural Network")
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of zones/nodes (even integer >= 2, default: 10)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Generate ritual MP3 audio to specified file path (default: None)",
    )
    parser.add_argument(
        "--demons",
        type=str,
        default=None,
        help="Comma-separated demon names/IDs to include in audio (default: all non-syzygy non-occult)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of generated audio in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--note-duration",
        type=float,
        default=0.5,
        help="Duration of each note in seconds (default: 0.5)",
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

    # 5. Graph conversion & sample path finding
    graph = model.to_directed_graph()
    sample_paths = find_paths(graph, start=2, end=6, n=2)
    print(f"\n=== Graph Conversion & Path Finding (Sample: Zone 2 -> Zone 6, length 2) ===")
    print(f"Total graph nodes: {len(graph)}")
    print(f"Paths found: {sample_paths}")

    # 6. Ritual Frequency Parsing Demo
    print(f"\n=== Ritual Frequency Parsing (Numogram Order n={model.n}) ===")
    lurgo_freqs = model.parse_ritual_frequencies("Lurgo")
    print(f"Mesh-00 Lurgo (Ana-1): {lurgo_freqs}")
    duoddod_freqs = parse_ritual_frequencies("Duoddod", numogram_order=model.n)
    print(f"Mesh-01 Duoddod (Ana-2): {duoddod_freqs}")

    # Verify exception raising on [X] and hidden paths
    try:
        model.parse_ritual_frequencies("Katak")
    except SyzygeticError as exc:
        print(f"Katak ([X] Syzygetic rite) raised expected exception: {exc}")

    try:
        model.parse_ritual_frequencies("Ixix")
    except OccultError as exc:
        print(f"Ixix ([?] Occult path) raised expected exception: {exc}")

    # 7. Audio Generation (if --audio specified)
    if args.audio:
        demon_list = [d.strip() for d in args.demons.split(",")] if args.demons else None
        audio_file = model.generate_ritual_audio(
            demons=demon_list,
            output_path=args.audio,
            duration=args.duration,
            note_duration=args.note_duration,
        )
        print(f"\n=== Ritual Audio Generated ===")
        print(f"Merged MP3 saved to: {audio_file}")
