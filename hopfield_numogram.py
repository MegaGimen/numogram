"""Energy-based (Hopfield-style) Numogram: no I/O, only descent on E(s)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from numogram import Numogram, triangular


class HopfieldNumogram(nn.Module):
    """Scalar activations s[0..n] relax by decreasing a typed energy.

    Terms (intentionally separate):
      syzygy  – pair complementarity (s_A + s_B)^2
      current – tractor tracks pair tension |s_A - s_B|
      binary  – double well toward ±1
      gate    – heavy zones prefer 0 until kicked (quadratic well ~ Gt)
      channel – equalizing wormhole, conductance = σ(|s_z| - thresh(Gt))
    """

    def __init__(
        self,
        n: int = 9,
        lam_syz: float = 1.0,
        lam_curr: float = 0.35,
        lam_bin: float = 0.8,
        lam_gate: float = 0.0,
        lam_chan: float = 0.45,
        gate_gain: float = 12.0,
        thresh_floor: float = 0.12,
        thresh_span: float = 0.78,
    ) -> None:
        super().__init__()
        self.n = int(n)
        topo = Numogram(n)
        self.topo = topo

        syz = torch.tensor(
            [[s.hi, s.lo, s.tractor] for s in topo.syzygies], dtype=torch.long
        )
        src = torch.arange(1, n + 1, dtype=torch.long)
        gt = src * (src + 1) // 2
        dst = topo.channel_dst.clone().long()
        gt_max = float(gt.max().clamp_min(1))
        thresh = thresh_floor + thresh_span * (gt.float() / gt_max)
        gt_w = gt.float() / gt_max

        self.register_buffer("syz_hi", syz[:, 0])
        self.register_buffer("syz_lo", syz[:, 1])
        self.register_buffer("syz_tr", syz[:, 2])
        self.register_buffer("chan_src", src)
        self.register_buffer("chan_dst", dst)
        self.register_buffer("gt_val", gt.float())
        self.register_buffer("gt_w", gt_w)
        self.register_buffer("thresh", thresh)

        self.lam_syz = lam_syz
        self.lam_curr = lam_curr
        self.lam_bin = lam_bin
        self.lam_gate = lam_gate
        self.lam_chan = lam_chan
        self.gate_gain = gate_gain

    def gate_open(self, s: torch.Tensor) -> torch.Tensor:
        mag = s[self.chan_src].abs()
        return torch.sigmoid(self.gate_gain * (mag - self.thresh))

    def energy_terms(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        a = s[self.syz_hi]
        b = s[self.syz_lo]
        t = s[self.syz_tr]
        # complementary pairs
        e_syz = 0.5 * ((a + b) ** 2).sum()
        # current is a directed pump: pair tension lowers energy only if tractor is positive
        e_curr = -(t * (a - b).square()).sum()
        # spin-like wells at ±1 (so 0 is a saddle, not a sink)
        e_bin = 0.25 * ((s.square() - 1.0) ** 2).sum()
        # gate is NOT a restoring spring to 0; it only sets channel conductance
        e_gate = s.new_zeros(())
        g = self.gate_open(s)
        # open wormhole: ferromagnetic Hopfield coupling src—dst
        e_chan = -(g * s[self.chan_src] * s[self.chan_dst]).sum()
        return {
            "syz": self.lam_syz * e_syz,
            "curr": self.lam_curr * e_curr,
            "bin": self.lam_bin * e_bin,
            "gate": self.lam_gate * e_gate,
            "chan": self.lam_chan * e_chan,
        }

    def energy(self, s: torch.Tensor) -> torch.Tensor:
        return sum(self.energy_terms(s).values())

    def step(self, s: torch.Tensor, lr: float = 0.08) -> tuple[torch.Tensor, float]:
        s = s.detach().clone().requires_grad_(True)
        e = self.energy(s)
        e.backward()
        with torch.no_grad():
            s_new = (s - lr * s.grad).clamp(-1.0, 1.0)
        return s_new, float(e.detach())

    @torch.no_grad()
    def snapshot(self, s: torch.Tensor) -> dict:
        s = s.detach()
        terms = {k: float(v) for k, v in self.energy_terms(s).items()}
        g = self.gate_open(s)
        pairs = []
        for i in range(len(self.syz_hi)):
            a = int(self.syz_hi[i])
            b = int(self.syz_lo[i])
            t = int(self.syz_tr[i])
            pairs.append(
                {
                    "pair": f"{a}::{b}",
                    "tractor": t,
                    "sA": float(s[a]),
                    "sB": float(s[b]),
                    "sum": float(s[a] + s[b]),
                    "tension": float((s[a] - s[b]).abs()),
                    "sT": float(s[t]),
                }
            )
        chans = []
        for i in range(len(self.chan_src)):
            z = int(self.chan_src[i])
            d = int(self.chan_dst[i])
            chans.append(
                {
                    "edge": f"{z}->{d}",
                    "gt": float(self.gt_val[i]),
                    "thresh": float(self.thresh[i]),
                    "open": float(g[i]),
                    "sz": float(s[z]),
                    "sd": float(s[d]),
                }
            )
        return {
            "s": [float(x) for x in s],
            "E": float(self.energy(s)),
            "terms": terms,
            "pairs": pairs,
            "channels": chans,
        }


def relax(
    model: HopfieldNumogram,
    s0: torch.Tensor,
    steps: int = 400,
    lr: float = 0.08,
    log_every: int = 50,
) -> dict:
    s = s0.detach().clone().float()
    trace = []
    energies = []
    for t in range(steps + 1):
        snap = model.snapshot(s)
        energies.append(snap["E"])
        if t % log_every == 0 or t == steps:
            trace.append((t, snap))
        if t == steps:
            break
        s, _ = model.step(s, lr=lr)
    return {"s": s, "energies": energies, "trace": trace, "final": model.snapshot(s)}


def fmt_vec(xs: list[float], nd: int = 2) -> str:
    return "[" + " ".join(f"{x:+.{nd}f}" for x in xs) + "]"
