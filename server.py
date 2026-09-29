#!/usr/bin/env python3
"""HTTP backend and static file server for Numogram Diffusion Visualizer.

Provides REST APIs that call PyTorch models in numogram.py and hopfield_numogram.py,
while serving the modern web frontend on http://0.0.0.0:8088.
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numogram import Numogram
from hopfield_numogram import HopfieldNumogram


class NumogramState:
    """Manages active simulation models and states."""

    def __init__(self, n: int = 9):
        self.n = n
        self.gnn = Numogram(n)
        self.hopfield = HopfieldNumogram(n)
        self.reset_state("zone1")

    def reset_state(self, preset: str = "zone1", custom_vals: list[float] | None = None):
        if custom_vals and len(custom_vals) == self.n + 1:
            self.s = torch.tensor(custom_vals, dtype=torch.float32)
        elif preset == "zone1":
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
            self.s[1] = 1.0
        elif preset == "zone8":
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
            self.s[8] = 1.0
        elif preset == "zone9":
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
            self.s[9] = 1.0
        elif preset == "warp":
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
            self.s[3] = 0.9
            self.s[6] = -0.9
        elif preset == "comp":
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
            for a, b in [(9, 0), (8, 1), (7, 2), (6, 3), (5, 4)]:
                self.s[a] = 0.8
                self.s[b] = -0.8
        elif preset == "random":
            self.s = torch.empty(self.n + 1).uniform_(-1.0, 1.0)
        else:
            self.s = torch.zeros(self.n + 1, dtype=torch.float32)
        self.step_count = 0
        self.history = [self.get_snapshot()]

    def step_gnn(
        self,
        w_current: float = 1.0,
        w_channel: float = 0.45,
        w_doubling: 1.0 = 1.0,
        decay: float = 0.08,
    ) -> dict:
        self.s = self.gnn.closed_loop_step(
            self.s,
            w_current=w_current,
            w_channel=w_channel,
            w_doubling=w_doubling,
            decay=decay,
        )
        self.step_count += 1
        snap = self.get_snapshot()
        self.history.append(snap)
        if len(self.history) > 100:
            self.history.pop(0)
        return snap

    def step_hopfield(self, lr: float = 0.08) -> dict:
        self.s, _ = self.hopfield.step(self.s, lr=lr)
        self.step_count += 1
        snap = self.get_snapshot()
        self.history.append(snap)
        if len(self.history) > 100:
            self.history.pop(0)
        return snap

    def get_snapshot(self) -> dict:
        s_list = [float(x) for x in self.s]
        snap_hf = self.hopfield.snapshot(self.s)

        # Region energy totals
        warp_heat = float(abs(self.s[3]) + abs(self.s[6]))
        torque_heat = float(sum(abs(self.s[z]) for z in [1, 2, 4, 5, 7, 8]))
        plex_heat = float(abs(self.s[9]) + abs(self.s[0]))

        return {
            "step": self.step_count,
            "s": s_list,
            "energy": snap_hf["E"],
            "terms": snap_hf["terms"],
            "pairs": snap_hf["pairs"],
            "channels": snap_hf["channels"],
            "regions": {
                "warp": warp_heat,
                "torque": torque_heat,
                "plex": plex_heat,
            },
        }

    def get_metadata(self) -> dict:
        return {
            "n": self.n,
            "num_zones": self.n + 1,
            "syzygies": [
                {"hi": s.hi, "lo": s.lo, "tractor": s.tractor, "kind": k}
                for s, k in zip(self.gnn.syzygies, self.gnn.syzygy_kinds)
            ],
            "channels": [
                {
                    "src": z,
                    "dst": int(self.gnn.channel_dst[z - 1]),
                    "gate_id": self.gnn.gate_id(z),
                    "gate_val": int(self.gnn.gate_val[z - 1]),
                }
                for z in range(1, self.n + 1)
            ],
            "doubling": [int(x) for x in self.gnn.doubling],
            "orbits": [
                {
                    "cycle": list(o.cycle),
                    "transients": list(o.transients),
                    "role": o.role,
                }
                for o in self.gnn.orbits
            ],
            "wormholes": [
                {"src": w[0], "dst": w[1], "from_role": w[2], "to_role": w[3]}
                for w in self.gnn.wormholes()
            ],
            "doors": self.gnn.doors(),
        }


sim = NumogramState(n=9)
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class NumogramRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/info":
            self._send_json({"metadata": sim.get_metadata(), "state": sim.get_snapshot()})
        elif parsed.path == "/api/state":
            self._send_json(sim.get_snapshot())
        elif parsed.path == "/api/reset":
            qs = parse_qs(parsed.query)
            preset = qs.get("preset", ["zone1"])[0]
            sim.reset_state(preset)
            self._send_json(sim.get_snapshot())
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed.path == "/api/step_gnn":
            snap = sim.step_gnn(
                w_current=float(payload.get("w_current", 1.0)),
                w_channel=float(payload.get("w_channel", 0.45)),
                w_doubling=float(payload.get("w_doubling", 1.0)),
                decay=float(payload.get("decay", 0.08)),
            )
            self._send_json(snap)
        elif parsed.path == "/api/step_hopfield":
            snap = sim.step_hopfield(lr=float(payload.get("lr", 0.08)))
            self._send_json(snap)
        elif parsed.path == "/api/inject":
            zone = int(payload.get("zone", 1))
            val = float(payload.get("value", 1.0))
            if 0 <= zone <= sim.n:
                sim.s[zone] = val
            self._send_json(sim.get_snapshot())
        elif parsed.path == "/api/set_state":
            vals = payload.get("s", [])
            if len(vals) == sim.n + 1:
                sim.reset_state("custom", custom_vals=vals)
            self._send_json(sim.get_snapshot())
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_json(self, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)


def run(port: int = 8088):
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, NumogramRequestHandler)
    print(f"Numogram Visualizer server listening on http://0.0.0.0:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run(port)
