#!/usr/bin/env python3
"""HTTP backend and audio streaming server for Numogram Ritual Music.

Provides:
  - REST API (/api/rituals): Complete metadata for all Pandemonium demons,
    ritual routes, pitch scale mappings, and computed zone frequencies.
  - Live Audio Stream (/api/stream): Real-time continuous audio streaming endpoint
    synthesizing multi-track Lemurian buzzer clicks for selected ritual routes.
  - Audio Export (/api/export_audio): On-demand WAV synthesis for selected routes.
  - Static file serving for CCRU-styled Pandemonium Web Frontend on 0.0.0.0.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# Ensure numogram modules can be imported
WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from numogrammusic import (
    FREQ_MAX,
    FREQ_MIN,
    NUM_PITCH_INTERVALS,
    PITCH_LEVELS,
    calculate_demon_tempo,
    get_pitch_intervals,
    load_pandemonium,
    zone_to_frequency,
)

WEB_DIR = WORKSPACE_DIR / "web"


def build_rituals_manifest(numogram_order: int = 10) -> dict:
    """Pre-computes and structures the complete demon and ritual route manifest."""
    pandemonium = load_pandemonium()
    demons = pandemonium.get("demons", [])
    intervals = get_pitch_intervals(f_min=FREQ_MIN, f_max=FREQ_MAX, ana_is_high=True)

    demons_with_routes = []
    non_route_demons = []
    total_valid_routes = 0

    for d in demons:
        pitch_obj = d.get("pitch", {})
        pitch_str = pitch_obj.get("value", "Null") if isinstance(pitch_obj, dict) else str(pitch_obj)
        clean_pitch = pitch_str.split()[0] if pitch_str else "Null"
        interval = intervals.get(clean_pitch, (1000.0, 1200.0))
        tempo = calculate_demon_tempo(d, numogram_order=numogram_order)

        ns = d.get("net_span", {})
        pair = ns.get("pair", [0, 0]) if isinstance(ns, dict) else [0, 0]
        net_span_str = f"{pair[0]}::{pair[1]}" if len(pair) == 2 else ""

        door_info = d.get("door", {})
        door_name = door_info.get("name", "") if isinstance(door_info, dict) else str(door_info)

        valid_routes = []
        for r in d.get("routes", []):
            path = r.get("path")
            if isinstance(path, list) and len(path) > 0 and all(isinstance(x, int) for x in path):
                freq_list = []
                for z in path:
                    f = zone_to_frequency(z, pitch=clean_pitch, n=numogram_order)
                    freq_list.append({
                        "zone": int(z),
                        "freq": round(float(f), 2),
                    })
                rid = r.get("id", f"Rt-{len(valid_routes)+1}")
                raw_meaning = r.get("meaning", "")
                clean_meaning = raw_meaning.split("(")[0].strip() if raw_meaning else "Navigational Time-Circuit Route"
                valid_routes.append({
                    "id": rid,
                    "demon_name": d.get("name"),
                    "route_key": f"{d.get('name')}:{rid}",
                    "route_number": r.get("route_number"),
                    "path": path,
                    "path_str": " → ".join(f"Zone {z}" for z in path),
                    "description": r.get("description", ""),
                    "meaning": clean_meaning,
                    "frequencies": freq_list,
                    "raw_frequencies": [round(float(item["freq"]), 2) for item in freq_list],
                })

        demon_entry = {
            "name": d.get("name"),
            "id": d.get("id", ""),
            "index": d.get("index", 0),
            "title": d.get("title", ""),
            "demon_type": d.get("demon_type", {}).get("type", ""),
            "order": d.get("demon_type", {}).get("order", ""),
            "net_span": net_span_str,
            "pair": pair,
            "pitch": clean_pitch,
            "pitch_interval": [round(interval[0], 2), round(interval[1], 2)],
            "tempo": round(tempo, 4),
            "tempo_ms": int(round(tempo * 1000)),
            "door": door_name,
            "routes": valid_routes,
            "routes_count": len(valid_routes),
        }

        if valid_routes:
            total_valid_routes += len(valid_routes)
            demons_with_routes.append(demon_entry)
        else:
            raw_rts = []
            for r in d.get("routes", []):
                raw_rts.append({
                    "id": r.get("id", "Rt-0"),
                    "path": r.get("path"),
                    "description": r.get("description", ""),
                })
            demon_entry["raw_routes"] = raw_rts
            non_route_demons.append(demon_entry)

    return {
        "demons": demons_with_routes,
        "non_route_demons": non_route_demons,
        "pitch_intervals": {k: [round(v[0], 2), round(v[1], 2)] for k, v in intervals.items()},
        "meta": {
            "numogram_order": numogram_order,
            "total_demons": len(demons),
            "demons_with_routes_count": len(demons_with_routes),
            "total_valid_routes": total_valid_routes,
            "freq_min": FREQ_MIN,
            "freq_max": FREQ_MAX,
        },
    }


# Cache manifest in memory
RITUALS_MANIFEST = build_rituals_manifest(numogram_order=10)


def build_wav_header(sample_rate: int = 44100, channels: int = 2, bits_per_sample: int = 16, data_size: int = 0x7FFF0000) -> bytes:
    """Builds a WAV container header supporting indefinite streaming."""
    file_size = data_size + 36
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


class RitualMusicRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler serving static UI and real-time audio endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/rituals":
            self._handle_api_rituals()
        elif path == "/api/stream":
            self._handle_audio_stream(parsed.query)
        elif path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._serve_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/numogram.jpg":
            img_path = WORKSPACE_DIR / "numogram.jpg"
            if img_path.is_file():
                self._serve_file(img_path, "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
        else:
            # Fall back to standard static file serving from WEB_DIR
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/export_audio":
            self._handle_export_audio()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"File {file_path.name} not found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _handle_api_rituals(self):
        """Returns JSON manifest of demons, routes, and calculated frequencies."""
        body = json.dumps(RITUALS_MANIFEST).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_audio_stream(self, query_string: str):
        """Continuously streams real-time synthesized audio chunks over HTTP."""
        query = parse_qs(query_string)
        selected_keys_param = query.get("rts", [""])[0]

        all_route_lookup = {}
        for d in RITUALS_MANIFEST["demons"]:
            for r in d["routes"]:
                all_route_lookup[r["route_key"]] = {
                    "demon": d["name"],
                    "tempo": d["tempo"],
                    "pitch": d["pitch"],
                    "pair": d["pair"],
                    "frequencies": r["raw_frequencies"],
                }

        if selected_keys_param == "all" or not selected_keys_param:
            active_routes = list(all_route_lookup.values())
        else:
            active_keys = [k.strip() for k in selected_keys_param.split(",") if k.strip()]
            active_routes = [all_route_lookup[k] for k in active_keys if k in all_route_lookup]

        if not active_routes:
            active_routes = list(all_route_lookup.values())[:5]

        sample_rate = 44100
        click_duration = 0.03
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/x-wav")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Write WAV header
        header = build_wav_header(sample_rate=sample_rate, channels=2, bits_per_sample=16)
        try:
            self.wfile.write(header)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        # Voice state tracking
        voices = []
        for v_idx, r_info in enumerate(active_routes):
            voices.append({
                "tempo": r_info["tempo"],
                "freqs": r_info["frequencies"],
                "note_idx": 0,
                "next_time": 0.0,
                "pan": -0.8 + 1.6 * (v_idx % 9) / 8.0,
            })

        chunk_sec = 0.1
        chunk_samples = int(sample_rate * chunk_sec)
        t_global = 0.0

        click_samples = max(16, int(round(click_duration * sample_rate)))
        t_click = np.arange(click_samples, dtype=np.float32) / float(sample_rate)
        fade_samples = min(int(0.002 * sample_rate), click_samples // 4)
        click_win = np.ones(click_samples, dtype=np.float32)
        if fade_samples > 0:
            click_win[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
            click_win[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, endpoint=False)

        try:
            while True:
                buf_l = np.zeros(chunk_samples, dtype=np.float32)
                buf_r = np.zeros(chunk_samples, dtype=np.float32)
                chunk_end_time = t_global + chunk_sec

                for v in voices:
                    tempo = v["tempo"]
                    freqs = v["freqs"]
                    if not freqs or tempo <= 0:
                        continue

                    pan = v["pan"]
                    gain_l = math.cos((pan + 1.0) * math.pi / 4.0)
                    gain_r = math.sin((pan + 1.0) * math.pi / 4.0)

                    while v["next_time"] < chunk_end_time:
                        if v["next_time"] >= t_global:
                            start_offset = int(round((v["next_time"] - t_global) * sample_rate))
                            if start_offset < chunk_samples:
                                f = freqs[v["note_idx"] % len(freqs)]
                                two_pi_f = 2.0 * np.pi * f
                                tone = (
                                    np.sin(two_pi_f * t_click)
                                    + 0.25 * np.sin(3.0 * two_pi_f * t_click)
                                ) * click_win * 0.2

                                end_offset = min(start_offset + click_samples, chunk_samples)
                                seg_len = end_offset - start_offset
                                buf_l[start_offset:end_offset] += tone[:seg_len] * gain_l
                                buf_r[start_offset:end_offset] += tone[:seg_len] * gain_r

                        v["note_idx"] += 1
                        v["next_time"] += tempo

                master_max = max(np.max(np.abs(buf_l)), np.max(np.abs(buf_r)), 1e-4)
                if master_max > 0.95:
                    buf_l = (buf_l / master_max) * 0.95
                    buf_r = (buf_r / master_max) * 0.95

                pcm_l = (buf_l * 32767.0).astype(np.int16)
                pcm_r = (buf_r * 32767.0).astype(np.int16)
                interleaved = np.empty((chunk_samples * 2,), dtype=np.int16)
                interleaved[0::2] = pcm_l
                interleaved[1::2] = pcm_r

                self.wfile.write(interleaved.tobytes())
                self.wfile.flush()

                t_global = chunk_end_time
                time.sleep(chunk_sec * 0.85)

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_export_audio(self):
        """Generates and returns a downloadable WAV file for selected routes."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        selected_keys = payload.get("rts", [])
        duration = float(payload.get("duration", 30.0))

        all_route_lookup = {}
        for d in RITUALS_MANIFEST["demons"]:
            for r in d["routes"]:
                all_route_lookup[r["route_key"]] = {
                    "demon": d["name"],
                    "tempo": d["tempo"],
                    "frequencies": r["raw_frequencies"],
                }

        if not selected_keys or selected_keys == "all":
            active_routes = list(all_route_lookup.values())
        else:
            active_routes = [all_route_lookup[k] for k in selected_keys if k in all_route_lookup]

        sample_rate = 44100
        n_samples = int(duration * sample_rate)
        total_samples = n_samples + int(sample_rate * 0.5)
        master_l = np.zeros(total_samples, dtype=np.float32)
        master_r = np.zeros(total_samples, dtype=np.float32)

        click_duration = 0.03
        click_samples = max(16, int(round(click_duration * sample_rate)))
        t_click = np.arange(click_samples, dtype=np.float32) / float(sample_rate)
        fade_samples = min(int(0.002 * sample_rate), click_samples // 4)
        click_win = np.ones(click_samples, dtype=np.float32)
        if fade_samples > 0:
            click_win[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
            click_win[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, endpoint=False)

        for v_idx, r_info in enumerate(active_routes):
            tempo = r_info["tempo"]
            freqs = r_info["frequencies"]
            if not freqs or tempo <= 0:
                continue

            pan = -0.8 + 1.6 * (v_idx % 9) / 8.0
            gain_l = math.cos((pan + 1.0) * math.pi / 4.0)
            gain_r = math.sin((pan + 1.0) * math.pi / 4.0)

            cur_t = 0.0
            k = 0
            while cur_t < duration:
                start_s = int(round(cur_t * sample_rate))
                if start_s < total_samples:
                    f = freqs[k % len(freqs)]
                    two_pi_f = 2.0 * np.pi * f
                    tone = (
                        np.sin(two_pi_f * t_click)
                        + 0.25 * np.sin(3.0 * two_pi_f * t_click)
                    ) * click_win * 0.2

                    end_s = min(start_s + click_samples, total_samples)
                    seg_len = end_s - start_s
                    master_l[start_s:end_s] += tone[:seg_len] * gain_l
                    master_r[start_s:end_s] += tone[:seg_len] * gain_r
                cur_t += tempo
                k += 1

        master_l = master_l[:n_samples]
        master_r = master_r[:n_samples]
        peak = max(np.max(np.abs(master_l)), np.max(np.abs(master_r)), 1e-4)
        if peak > 0.95:
            master_l = (master_l / peak) * 0.95
            master_r = (master_r / peak) * 0.95

        pcm_l = (master_l * 32767.0).astype(np.int16)
        pcm_r = (master_r * 32767.0).astype(np.int16)
        interleaved = np.empty((n_samples * 2,), dtype=np.int16)
        interleaved[0::2] = pcm_l
        interleaved[1::2] = pcm_r
        data_bytes = interleaved.tobytes()

        header = build_wav_header(
            sample_rate=sample_rate,
            channels=2,
            bits_per_sample=16,
            data_size=len(data_bytes),
        )

        response_bytes = header + data_bytes
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Disposition", 'attachment; filename="numogram_rituals.wav"')
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)


def run_server(host: str = "0.0.0.0", port: int = 8088):
    """Starts the threaded HTTP server bound to 0.0.0.0."""
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, RitualMusicRequestHandler)
    print(f"============================================================")
    print(f"CCRU Numogram Ritual Music Server listening on:")
    print(f"  http://{host}:{port}/")
    print(f"  http://127.0.0.1:{port}/")
    print(f"Demons with valid routes: {RITUALS_MANIFEST['meta']['demons_with_routes_count']}")
    print(f"Total operational routes: {RITUALS_MANIFEST['meta']['total_valid_routes']}")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Ritual Music server.")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run_server(host="0.0.0.0", port=port_arg)
