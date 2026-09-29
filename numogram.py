"""Parameterized CCRU Numogram (decimal labyrinth) as CPU torch tensors.

Zones are {0, 1, ..., n}. The original CCRU diagram is n=9 (base 10).
All arithmetic is the base-(n+1) analogue of the decimal construction:
  - syzygies: n-sum twinning
  - digital reduction: residue mod n, with n standing for 0 (except 0 itself)
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


def digital_root(x: int, n: int) -> int:
    """Single-digit reduction in base n+1. 0 stays 0; multiples of n become n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if x == 0:
        return 0
    r = x % n
    return n if r == 0 else r


def digital_root_tensor(x: torch.Tensor, n: int) -> torch.Tensor:
    x = x.to(dtype=torch.long)
    r = torch.remainder(x, n)
    return torch.where(x == 0, x, torch.where(r == 0, torch.full_like(r, n), r))


def triangular(k: int) -> int:
    return k * (k + 1) // 2


@dataclass(frozen=True)
class Syzygy:
    hi: int
    lo: int
    tractor: int

    @property
    def pair(self) -> tuple[int, int]:
        return (self.hi, self.lo)

    @property
    def self_fold(self) -> bool:
        return self.tractor in (self.hi, self.lo)

def _syzygy_kind(s: Syzygy, n: int) -> str:
    if s.lo == 0 and s.hi == n:
        return "plex"
    if s.hi == s.lo:
        return "center"
    if s.self_fold:
        return "warp"
    return "circuit"


@dataclass
class Orbit:
    cycle: tuple[int, ...]
    transients: tuple[int, ...]
    role: str  # plex / warp / time-circuit / other / zero


class Numogram(nn.Module):
    """Directed multi-graph of zones 0..n plus typed adjacency buffers."""

    def __init__(self, n: int = 9) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1 (CCRU original is n=9)")
        self.n = int(n)
        m = self.n + 1

        syzygies = self._build_syzygies()
        kinds = [_syzygy_kind(s, self.n) for s in syzygies]
        self.syzygies: list[Syzygy] = syzygies
        self.syzygy_kinds: list[str] = kinds

        zones = torch.arange(m, dtype=torch.long)
        hi = torch.tensor([s.hi for s in syzygies], dtype=torch.long)
        lo = torch.tensor([s.lo for s in syzygies], dtype=torch.long)
        tractor = torch.tensor([s.tractor for s in syzygies], dtype=torch.long)

        gate_src = torch.arange(1, m, dtype=torch.long)
        gate_val = gate_src * (gate_src + 1) // 2
        channel_dst = digital_root_tensor(gate_val, self.n)

        doubling = digital_root_tensor(zones * 2, self.n)

        A_current = torch.zeros(m, m, dtype=torch.float32)
        A_current[hi, tractor] = 1.0
        A_current[lo, tractor] = 1.0

        A_channel = torch.zeros(m, m, dtype=torch.float32)
        A_channel[gate_src, channel_dst] = 1.0

        A_doubling = torch.zeros(m, m, dtype=torch.float32)
        A_doubling[zones, doubling] = 1.0

        A_syzygy = torch.zeros(m, m, dtype=torch.float32)
        A_syzygy[hi, lo] = 1.0
        A_syzygy[lo, hi] = 1.0

        A_door = torch.zeros(m, m, dtype=torch.float32)
        A_door[gate_src, 0] = 1.0

        self.register_buffer("zones", zones)
        self.register_buffer("syzygy_hi", hi)
        self.register_buffer("syzygy_lo", lo)
        self.register_buffer("tractor", tractor)
        self.register_buffer("gate_src", gate_src)
        self.register_buffer("gate_val", gate_val)
        self.register_buffer("channel_dst", channel_dst)
        self.register_buffer("doubling", doubling)
        self.register_buffer("A_current", A_current)
        self.register_buffer("A_channel", A_channel)
        self.register_buffer("A_doubling", A_doubling)
        self.register_buffer("A_syzygy", A_syzygy)
        self.register_buffer("A_door", A_door)

        self.orbits: list[Orbit] = self._build_orbits()
        self.role_of: dict[int, str] = {}
        for orb in self.orbits:
            for z in orb.cycle + orb.transients:
                self.role_of[z] = orb.role

    @property
    def num_zones(self) -> int:
        return self.n + 1

    def _build_syzygies(self) -> list[Syzygy]:
        pairs: list[Syzygy] = []
        seen: set[tuple[int, int]] = set()
        for a in range(self.n, -1, -1):
            b = self.n - a
            key = (max(a, b), min(a, b))
            if key in seen:
                continue
            seen.add(key)
            tractor = abs(a - b)
            pairs.append(Syzygy(hi=key[0], lo=key[1], tractor=tractor))
        return pairs

    def _build_orbits(self) -> list[Orbit]:
        dbl = [int(self.doubling[i]) for i in range(self.num_zones)]

        def canon_cycle(start: int) -> tuple[int, ...]:
            seen: dict[int, int] = {}
            path: list[int] = []
            x = start
            while x not in seen:
                seen[x] = len(path)
                path.append(x)
                x = dbl[x]
            cyc = path[seen[x] :]
            k = min(range(len(cyc)), key=lambda i: cyc[i])
            return tuple(cyc[k:] + cyc[:k])

        basins: dict[tuple[int, ...], list[int]] = {}
        for z in range(self.num_zones):
            cyc = canon_cycle(z)
            basins.setdefault(cyc, []).append(z)

        orbits = []
        for cyc, members in basins.items():
            cset = set(cyc)
            transients = tuple(sorted(z for z in members if z not in cset))
            orbits.append(Orbit(cycle=cyc, transients=transients, role="other"))
        orbits.sort(key=lambda o: (o.cycle[0], o.cycle))
        return self._label_orbits(orbits)

    def _label_orbits(self, orbits: list[Orbit]) -> list[Orbit]:
        labeled: list[Orbit] = []
        for orb in orbits:
            cyc = orb.cycle
            if cyc == (0,):
                role = "zero"
            elif self.n in cyc and len(cyc) == 1:
                role = "plex"
            elif len(cyc) == 2:
                role = "warp"
            elif 1 in cyc or 1 in orb.transients:
                role = "time-circuit"
            else:
                role = "other"
            labeled.append(Orbit(cycle=cyc, transients=orb.transients, role=role))
        # if 1 feeds a warp/plex, keep that role and note the feed
        return labeled

    def channel_target(self, z: int) -> int | None:
        if z == 0:
            return None
        return int(self.channel_dst[z - 1])

    def gate_id(self, z: int) -> str | None:
        if z == 0:
            return None
        return f"Gt-{int(self.gate_val[z - 1]):02d}"

    def net_spans(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(1, self.num_zones) for j in range(i)]

    def doors(self) -> list[tuple[int, int]]:
        return [(i, 0) for i in range(1, self.num_zones)]

    def wormholes(self) -> list[tuple[int, int, str, str]]:
        """Channel edges that jump between doubling-orbit roles."""
        out = []
        for z in range(1, self.num_zones):
            t = self.channel_target(z)
            if t is None:
                continue
            rz, rt = self.role_of.get(z, "?"), self.role_of.get(t, "?")
            if rz != rt:
                out.append((z, t, rz, rt))
        return out

    def current_functional_graph(self) -> dict[int, int]:
        """Each zone's current destination (syzygy partner-pair tractor)."""
        dest = {}
        for s in self.syzygies:
            dest[s.hi] = s.tractor
            dest[s.lo] = s.tractor
        return dest

    def follow(self, start: int, steps: int, kind: str = "doubling") -> list[int]:
        x = start
        path = [x]
        for _ in range(steps):
            if kind == "doubling":
                x = int(self.doubling[x])
            elif kind == "channel":
                if x == 0:
                    break
                x = int(self.channel_dst[x - 1])
            elif kind == "current":
                x = self.current_functional_graph()[x]
            else:
                raise ValueError(kind)
            path.append(x)
            if x == path[0] and len(path) > 1:
                break
        return path

    def typed_edge_index(self) -> dict[str, torch.Tensor]:
        """COO edge_index [2, E] per relation, for message-passing."""

        def from_adj(A: torch.Tensor) -> torch.Tensor:
            src, dst = torch.nonzero(A, as_tuple=True)
            return torch.stack([src, dst], dim=0)

        return {
            "current": from_adj(self.A_current),
            "channel": from_adj(self.A_channel),
            "doubling": from_adj(self.A_doubling),
            "syzygy": from_adj(self.A_syzygy),
            "door": from_adj(self.A_door),
        }

    def closed_loop_step(
        self,
        h: torch.Tensor,
        w_current: float = 1.0,
        w_channel: float = 1.0,
        w_doubling: float = 1.0,
        decay: float = 0.1,
    ) -> torch.Tensor:
        """One linear mixing step on node states h: [..., n+1, d] or [n+1]."""
        A = (
            w_current * self.A_current
            + w_channel * self.A_channel
            + w_doubling * self.A_doubling
        )
        # in-degree mix + residual
        # A[src, dst] = 1  →  gather into destinations
        indeg = A.sum(dim=0).clamp_min(1.0)
        mixed = A.T @ h
        if mixed.ndim == 1:
            mixed = mixed / indeg
        else:
            mixed = mixed / indeg.unsqueeze(-1)
        return (1.0 - decay) * mixed + decay * h

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.closed_loop_step(h, **kwargs)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Numogram n={self.n}  zones=0..{self.n}  (base {self.n + 1})",
            "",
            "Syzygies (n-sum twinning) and currents |A-B| -> tractor:",
        ]
        for s, kind in zip(self.syzygies, self.syzygy_kinds):
            fold = "  self-fold" if s.self_fold else ""
            lines.append(
                f"  {s.hi}::{s.lo}  |{s.hi}-{s.lo}|={s.tractor}  -> {s.tractor}  [{kind}]{fold}"
            )
        lines.append("")
        lines.append("Doubling orbits  x |-> digital_root(2x):")
        for orb in self.orbits:
            cyc = " -> ".join(map(str, orb.cycle + (orb.cycle[0],)))
            extra = f"  feeds={list(orb.transients)}" if orb.transients else ""
            lines.append(f"  [{orb.role}]  {cyc}{extra}")
        lines.append("")
        lines.append("Gates / channels (triangular then digital reduction):")
        for z in range(1, self.num_zones):
            gv = int(self.gate_val[z - 1])
            t = int(self.channel_dst[z - 1])
            lines.append(f"  Zone {z}  {self.gate_id(z)}={gv}  -> {t}")
        wh = self.wormholes()
        lines.append("")
        if wh:
            lines.append("Wormholes (channel jumps across time-systems):")
            for z, t, rz, rt in wh:
                lines.append(f"  {z} -> {t}   ({rz} => {rt})")
        else:
            lines.append("Wormholes: none")
        lines.append("")
        lines.append(f"Doors (#::0): {self.doors()}")
        return lines

    def __repr__(self) -> str:
        return f"Numogram(n={self.n}, zones=0..{self.n}, syzygies={len(self.syzygies)})"


def kaprekar_step(value: int, digits: int, base: int) -> int:
    """Sort digits descending/ascending in `base` and subtract, pad to `digits`."""
    ds = []
    x = value
    for _ in range(digits):
        ds.append(x % base)
        x //= base
    if x:
        # overflow digits: still include them so the map stays honest
        while x:
            ds.append(x % base)
            x //= base
        ds = (ds + [0] * digits)[:digits]
    desc = sorted(ds, reverse=True)
    asc = sorted(ds)
    hi = lo = 0
    for d in desc:
        hi = hi * base + d
    for d in asc:
        lo = lo * base + d
    return hi - lo


def kaprekar_cycles(digits: int, base: int) -> list[tuple[int, ...]]:
    space = base**digits
    seen_comp: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in range(space):
        if start in seen_comp:
            continue
        path: list[int] = []
        idx: dict[int, int] = {}
        x = start
        while x not in idx:
            idx[x] = len(path)
            path.append(x)
            x = kaprekar_step(x, digits, base)
        cyc = tuple(path[idx[x] :])
        # rotate to min for dedup
        k = cyc.index(min(cyc))
        cyc = cyc[k:] + cyc[:k]
        if cyc not in cycles:
            cycles.append(cyc)
        for y in path:
            seen_comp.add(y)
    cycles.sort(key=lambda c: (len(c), c))
    return cycles


def build(n: int = 9) -> Numogram:
    return Numogram(n)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    g = Numogram(n)
    print("\n".join(g.summary_lines()))
