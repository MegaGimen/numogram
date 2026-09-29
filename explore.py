#!/usr/bin/env python3
"""Verify n=9 against the CCRU spec, then hunt regularities for general n."""

from __future__ import annotations

from collections import defaultdict

from numogram import Numogram, digital_root, kaprekar_cycles


def assert_ccru_n9() -> None:
    g = Numogram(9)
    assert g.n == 9 and g.num_zones == 10
    pairs = [(s.hi, s.lo, s.tractor) for s in g.syzygies]
    assert pairs == [(9, 0, 9), (8, 1, 7), (7, 2, 5), (6, 3, 3), (5, 4, 1)], pairs
    assert g.syzygy_kinds == ["plex", "circuit", "circuit", "warp", "circuit"]

    channels = {z: g.channel_target(z) for z in range(1, 10)}
    assert channels == {1: 1, 2: 3, 3: 6, 4: 1, 5: 6, 6: 3, 7: 1, 8: 9, 9: 9}, channels
    gates = {z: int(g.gate_val[z - 1]) for z in range(1, 10)}
    assert gates == {1: 1, 2: 3, 3: 6, 4: 10, 5: 15, 6: 21, 7: 28, 8: 36, 9: 45}

    dbl = [int(g.doubling[i]) for i in range(10)]
    assert dbl == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9], dbl

    roles = {orb.role: orb for orb in g.orbits}
    assert roles["time-circuit"].cycle == (1, 2, 4, 8, 7, 5)
    assert roles["warp"].cycle == (3, 6)
    assert roles["plex"].cycle == (9,)
    assert roles["zero"].cycle == (0,)
    print("[ok] n=9 matches CCRU construction (syzygies, currents, gates, doubling)")


def current_attractors(g: Numogram) -> list[tuple[int, ...]]:
    dest = g.current_functional_graph()
    cycles = []
    seen = set()
    for start in range(g.num_zones):
        if start in seen:
            continue
        path, idx = [], {}
        x = start
        while x not in idx:
            idx[x] = len(path)
            path.append(x)
            x = dest[x]
        cyc = tuple(path[idx[x] :])
        k = cyc.index(min(cyc))
        cyc = cyc[k:] + cyc[:k]
        if cyc not in cycles:
            cycles.append(cyc)
        seen.update(path)
    return sorted(cycles, key=lambda c: (len(c), c))


def order_of_two(n: int) -> int | None:
    """Multiplicative order of 2 modulo n, if gcd(2,n)=1."""
    if n % 2 == 0:
        return None
    x, seen = 1, {}
    for i in range(1, n + 2):
        x = (2 * x) % n
        if x == 1:
            return i
        if x in seen:
            return None
        seen[x] = i
    return None


def scan_time_systems(nmax: int = 36) -> None:
    print("\n=== doubling orbits for n=2..{} ===".format(nmax))
    print(
        f"{'n':>3} {'parity':>6} {'3|n':>4}  "
        f"{'time-circuit':<28} {'warp':<12} {'plex/zero':<14} other"
    )
    for n in range(2, nmax + 1):
        g = Numogram(n)
        by = defaultdict(list)
        for orb in g.orbits:
            by[orb.role].append(orb)
        tc = by.get("time-circuit", [])
        wp = by.get("warp", [])
        ot = by.get("other", [])
        tc_s = ",".join(str(list(o.cycle)) for o in tc)
        if not tc:
            # 1 was swallowed by plex/warp
            feed = []
            for o in g.orbits:
                if 1 in o.cycle or 1 in o.transients:
                    feed.append(f"1 feeds {o.role}{list(o.cycle)}")
            tc_s = "; ".join(feed) or "-"
        wp_s = ",".join(str(list(o.cycle)) for o in wp) or "-"
        ot_s = ",".join(str(list(o.cycle)) for o in ot) or "-"
        pz = []
        for role in ("plex", "zero"):
            for o in by.get(role, []):
                pz.append(str(list(o.cycle)))
        print(
            f"{n:>3} {'even' if n%2==0 else 'odd':>6} {'yes' if n%3==0 else 'no':>4}  "
            f"{tc_s:<28} {wp_s:<12} {','.join(pz):<14} {ot_s}"
        )


def scan_algebra(nmax: int = 30) -> None:
    print("\n=== algebraic regularities ===")
    print("warp self-fold <=> 3|n ; even n: doubling of 1 falls into plex; channel(n):")
    for n in range(2, nmax + 1):
        g = Numogram(n)
        warp_pairs = [s for s, k in zip(g.syzygies, g.syzygy_kinds) if k == "warp"]
        center = [s for s, k in zip(g.syzygies, g.syzygy_kinds) if k == "center"]
        ch_n = g.channel_target(n)
        dbl_path = g.follow(1, steps=n + 2, kind="doubling")
        hits_plex = g.n in dbl_path
        ord2 = order_of_two(n)
        tc = next((o for o in g.orbits if o.role == "time-circuit"), None)
        tclen = len(tc.cycle) if tc else 0
        flags = []
        if (n % 3 == 0) != bool(warp_pairs):
            flags.append("WARP-MISMATCH")
        if n % 2 == 0 and not hits_plex:
            flags.append("EVEN-NO-PLEX-FALL")
        if n % 2 == 1 and ch_n != n:
            flags.append(f"ODD-CH-N={ch_n}")
        if n % 2 == 0 and ch_n != n // 2:
            flags.append(f"EVEN-CH-N={ch_n} (want {n//2})")
        if n % 2 == 1 and ord2 and tclen and ord2 != tclen:
            flags.append(f"ORD2={ord2}!=TCLEN={tclen}")
        extra = ""
        if center:
            extra += f"  center {center[0].hi}::{center[0].lo}->0"
        if warp_pairs:
            extra += "  warp " + ",".join(f"{s.hi}::{s.lo}->{s.tractor}" for s in warp_pairs)
        mark = "  !" + ",".join(flags) if flags else "  ok"
        print(
            f"  n={n:<2} ch(n)={ch_n:<2} ord2={str(ord2):<4} tclen={tclen:<2} "
            f"1-path hits plex={str(hits_plex):<5}{extra}{mark}"
        )


def scan_kaprekar() -> None:
    print("\n=== 2-digit Kaprekar in base n+1 vs syzygies ===")
    for n in [7, 8, 9, 10, 12, 15]:
        g = Numogram(n)
        base = n + 1
        syz_nums = set()
        for s in g.syzygies:
            syz_nums.add(s.hi * base + s.lo)
            syz_nums.add(s.lo * base + s.hi)
        cycles = kaprekar_cycles(digits=2, base=base)
        nontrivial = [c for c in cycles if c != (0,)]
        print(f"\n  n={n} base={base}  syzygies as 2-digit: {sorted(syz_nums)}")
        for cyc in nontrivial:
            overlap = [x for x in cyc if x in syz_nums]
            tag = "ALL-SYZYGY" if overlap == list(cyc) else f"overlap={overlap}"
            print(f"    cycle{list(cyc)}  {tag}")

    print("\n=== classic decimal Kaprekar (base 10) ===")
    for digits in (2, 3, 4):
        cycles = kaprekar_cycles(digits=digits, base=10)
        show = [c for c in cycles if not (len(c) == 1 and c[0] == 0)]
        print(f"  {digits}-digit cycles: {show[:12]}" + (" ..." if len(show) > 12 else ""))

    # 6174 digit bipartitions -> n=9 time-circuit syzygies
    print("\n  6174 three bipartitions -> digital-root pairs:")
    digits = [6, 1, 7, 4]
    splits = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
    for a, b, c, d in splits:
        p, q = digits[a] * 10 + digits[b], digits[c] * 10 + digits[d]
        rp, rq = digital_root(p, 9), digital_root(q, 9)
        print(f"    {p:02d}/{q:02d}  ->  {rp}::{rq}  (sum={rp+rq})")


def scan_currents_and_wormholes() -> None:
    print("\n=== current attractors (follow |A-B| repeatedly) ===")
    for n in [7, 8, 9, 10, 12, 15, 16, 27]:
        g = Numogram(n)
        print(f"  n={n}  {current_attractors(g)}")

    print("\n=== wormholes (channel jumps between time-systems) ===")
    for n in [7, 8, 9, 12, 15, 16, 27]:
        g = Numogram(n)
        wh = g.wormholes()
        print(f"  n={n}  {wh if wh else 'none'}")


def closed_loop_toy() -> None:
    print("\n=== closed-loop mix on n=9 (impulse at zone 1, 12 steps) ===")
    import torch

    g = Numogram(9)
    h = torch.zeros(10)
    h[1] = 1.0
    loc = []
    for t in range(12):
        loc.append(int(h.argmax()))
        h = g.closed_loop_step(h, w_current=1.0, w_channel=0.3, w_doubling=1.0, decay=0.05)
    print(f"  argmax path: {loc}")
    print(f"  mass by role:")
    mass = defaultdict(float)
    for z in range(10):
        mass[g.role_of[z]] += float(h[z])
    for k, v in mass.items():
        print(f"    {k:14s} {v:.4f}")


def main() -> None:
    assert_ccru_n9()
    g = Numogram(9)
    print()
    print("\n".join(g.summary_lines()))
    scan_time_systems(36)
    scan_algebra(32)
    scan_kaprekar()
    scan_currents_and_wormholes()
    closed_loop_toy()


if __name__ == "__main__":
    main()
