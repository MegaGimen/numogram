#!/usr/bin/env python3
"""Run the energy Numogram from several initial states and print what actually happens."""

from __future__ import annotations

import torch

from hopfield_numogram import HopfieldNumogram, fmt_vec, relax


def print_snap(title: str, snap: dict, n: int) -> None:
    print(f"\n-- {title}  E={snap['E']:.4f}  terms={ {k: round(v, 3) for k,v in snap['terms'].items()} }")
    print(f"   s = {fmt_vec(snap['s'])}")
    print("   syzygy  sA,sB  sum  tension  sT")
    for p in snap["pairs"]:
        print(
            f"     {p['pair']:>4} -> {p['tractor']}   "
            f"{p['sA']:+.2f},{p['sB']:+.2f}  sum={p['sum']:+.2f}  "
            f"|A-B|={p['tension']:.2f}  sT={p['sT']:+.2f}"
        )
    opened = [c for c in snap["channels"] if c["open"] > 0.5]
    print("   open channels (σ>0.5):", end=" ")
    if opened:
        print(", ".join(f"{c['edge']}({c['open']:.2f})" for c in opened))
    else:
        print("none")


def run_case(model: HopfieldNumogram, name: str, s0: torch.Tensor, steps: int = 500, lr: float = 0.08) -> dict:
    out = relax(model, s0, steps=steps, lr=lr, log_every=max(steps // 5, 1))
    e0, e1 = out["energies"][0], out["energies"][-1]
    mono = all(out["energies"][i] + 1e-6 >= out["energies"][i + 1] for i in range(len(out["energies"]) - 1))
    print(f"\n===== {name} =====")
    print(f"E: {e0:.4f} -> {e1:.4f}   strictly nonincreasing={mono}   Δ={e1-e0:+.4f}")
    print("s0", fmt_vec(s0.tolist()))
    for t, snap in out["trace"]:
        n_open = sum(1 for c in snap["channels"] if c["open"] > 0.5)
        print(f"   t={t:3d}  E={snap['E']:+.3f}  |s|_max={max(abs(x) for x in snap['s']):.2f}  open={n_open}  s={fmt_vec(snap['s'], 1)}")
    print_snap("final", out["final"], model.n)
    return out


def cluster_attractors(states: list[list[float]], tol: float = 0.15) -> list[list[int]]:
    groups: list[list[int]] = []
    cents: list[torch.Tensor] = []
    for i, xs in enumerate(states):
        v = torch.tensor(xs)
        hit = None
        for j, c in enumerate(cents):
            if (v - c).abs().max() < tol:
                hit = j
                break
        if hit is None:
            cents.append(v)
            groups.append([i])
        else:
            groups[hit].append(i)
            cents[hit] = 0.5 * (cents[hit] + v)
    return groups


def main() -> None:
    torch.manual_seed(0)
    print("CPU Hopfield-Numogram  n=9")
    m = HopfieldNumogram(9)

    run_case(m, "all-zero + tiny noise", 0.05 * torch.randn(10))
    s1 = torch.zeros(10)
    s1[1] = 1.0
    run_case(m, "impulse Zone 1", s1)
    s8 = torch.zeros(10)
    s8[8] = 1.0
    run_case(m, "impulse Zone 8", s8)
    s9 = torch.zeros(10)
    s9[9] = 1.0
    run_case(m, "impulse Zone 9", s9)
    comp = torch.zeros(10)
    for a, b in [(9, 0), (8, 1), (7, 2), (6, 3), (5, 4)]:
        comp[a], comp[b] = 0.8, -0.8
    run_case(m, "already complementary ±0.8", comp)

    print("\n===== 40 random inits → attractor clusters =====")
    finals = []
    records = []
    for k in range(40):
        s0 = torch.empty(10).uniform_(-1.0, 1.0)
        out = relax(m, s0, steps=400, lr=0.08, log_every=400)
        fin = out["final"]
        finals.append(fin["s"])
        opened = tuple(c["edge"] for c in fin["channels"] if c["open"] > 0.5)
        sums = tuple(round(p["sum"], 2) for p in fin["pairs"])
        records.append((fin["E"], opened, sums, fin["s"][9], fin["s"][3], fin["s"][6]))

    groups = cluster_attractors(finals, tol=0.2)
    print(f"distinct attractors (L∞<0.2): {len(groups)}")
    for gi, idxs in enumerate(groups):
        proto = finals[idxs[0]]
        e_mean = sum(records[i][0] for i in idxs) / len(idxs)
        s9_mean = sum(records[i][3] for i in idxs) / len(idxs)
        open_set = records[idxs[0]][1]
        print(
            f"  cluster {gi}: n={len(idxs):2d}  E~{e_mean:.3f}  s9~{s9_mean:+.2f}  "
            f"open={open_set or '-'}  s={fmt_vec(proto, 1)}"
        )

    print("\n===== n=8 vs n=9  (same random seed inits, 12 trials) =====")
    m8 = HopfieldNumogram(8)
    for n, model, dim in [(9, m, 10), (8, m8, 9)]:
        torch.manual_seed(1)
        e_list, s_last = [], []
        n_open = []
        for _ in range(12):
            s0 = torch.empty(dim).uniform_(-1.0, 1.0)
            fin = relax(model, s0, steps=350, lr=0.08, log_every=350)["final"]
            e_list.append(fin["E"])
            s_last.append(fin["s"][-1])
            n_open.append(sum(1 for c in fin["channels"] if c["open"] > 0.5))
        print(
            f"  n={n}: final E mean={sum(e_list)/len(e_list):.3f}  "
            f"last-zone mean={sum(s_last)/len(s_last):+.2f}  "
            f"open-channels mean={sum(n_open)/len(n_open):.2f}"
        )


if __name__ == "__main__":
    main()
