"""Numogram Ritual Frequency and Music Module (numogrammusic.py).

Provides:
  - Pandemonium matrix loading and demon metadata inspection.
  - 15 pitch level definitions (Ana-7 to Cth-7 across [33 Hz, 3960 Hz]).
  - Frequency calculations for Numogram zones within pitch intervals.
  - Net-Span tempo determination: demon note duration based on ratio min(p, q) / max(p, q).
  - Ritual route parsing with OccultError and SyzygeticError exceptions.
  - Multi-track audio synthesis using music.py acoustic piano tone model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any
import wave

import numpy as np


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
    """
    clean_pitch = pitch.split()[0] if pitch else "Null"
    intervals = get_pitch_intervals(f_min=f_min, f_max=f_max, ana_is_high=ana_is_high)
    if clean_pitch not in intervals:
        raise ValueError(f"Unknown pitch '{pitch}', expected one of {PITCH_LEVELS}")
    a, b = intervals[clean_pitch]
    x = zone / float(n)
    return a + (b - a) * x


def calculate_demon_tempo(
    demon: dict[str, Any] | str | int,
    numogram_order: int = 10,
    default: float = 0.5,
) -> float:
    """Calculates the demon note duration (tempo) using the Numogram modular base-n rule.

    Given Net-Span a::b with a < b:
      1. In base n (numogram_order), forms the integer value ab = a * n + b.
      2. Computes ab mod (n - 1). If the remainder is 0, takes (n - 1).
      3. Tempo = (ab mod (n - 1)) / n.

    For classical n=10:
      Modulo is 9. Results in {1, 2, ..., 9} / 10 = {0.1s, 0.2s, ..., 0.9s}.
      Each of the 9 tempo slots contains exactly 5 demons, uniformly filling every second.
    """
    if not isinstance(demon, dict):
        pandemonium = load_pandemonium()
        demon = _find_demon(demon, pandemonium.get("demons", []))

    ns = demon.get("net_span", {})
    pair = ns.get("pair") if isinstance(ns, dict) else None
    if not pair or len(pair) != 2:
        return default

    p, q = int(pair[0]), int(pair[1])
    a, b = min(p, q), max(p, q)

    n = int(numogram_order)
    m = n - 1

    ab_val = a * n + b
    rem = ab_val % m
    val = m if rem == 0 else rem

    return float(val) / float(n)


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

    Args:
        rituals: Demon name/index/dict or route dictionary.
        numogram_order: Number of zones/nodes (default 10).
        route_id: Optional route ID to filter (e.g. 'Rt-1').
        f_min: Minimum frequency (default 33.0).
        f_max: Maximum frequency (default 3960.0).
        ana_is_high: Orientation of pitches (default True).
        as_dict: If True, returns dict mapping route ID to frequencies.
        errors: Error handling ('raise', 'skip', 'store').

    Returns:
        Parsed frequencies structure.
    """
    pandemonium = load_pandemonium()
    demons = pandemonium.get("demons", [])

    if isinstance(rituals, (int, str)):
        demon = _find_demon(rituals, demons)
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
            elif errors == "skip":
                continue
            elif errors == "store":
                route_results[rid] = str(exc)
            else:
                raise ValueError(f"Invalid errors mode '{errors}', choose 'raise', 'skip', or 'store'")

    if not as_dict and len(route_results) == 1:
        return next(iter(route_results.values()))
    return route_results


def generate_ritual_audio(
    demons: list[str | int | dict[str, Any]] | None = None,
    output_path: str | Path = "ritual_audio.mp3",
    duration: float = 60.0,
    *,
    timbre: str = "buzzer",
    click_duration: float = 0.03,
    stagger_phase: bool = False,
    numogram_order: int = 10,
    note_duration: float | None = None,
    use_netspan_tempo: bool = True,
    sample_rate: int = 44100,
    f_min: float = 33.0,
    f_max: float = 3960.0,
    ana_is_high: bool = True,
    stereo: bool = True,
    skip_syzygy_and_occult: bool = True,
) -> Path:
    """Generates an MP3 audio track combining looped rituals of selected demons.

    Timbre modes:
      - 'buzzer' / 'click' (default): Minimalist clicking sounds / buzzer beeps ('滴一下'),
        a short electronic blip (default 30ms) at the ritual note frequency, leaving crisp
        transparent space between beats.
      - 'piano': Acoustic piano tone simulation from music.py.

    Individual demon tempo & Phase Staggering:
      - when use_netspan_tempo=True (default), each demon's tempo is determined by its
        Net-Span ratio min(p, q) / max(p, q) (e.g. 3::1 -> 0.3333s).
      - when stagger_phase=True (default), offsets initial start phases so that rational
        tempo fractions (1/2, 1/3, 1/4... 1/10) do not simultaneously collide at every 1.0s,
        completely eliminating artificial periodic gaps/pauses.

    Args:
        demons: List of demon names, IDs, indices, or dicts (default: all non-syzygy non-occult).
        output_path: Output audio file path (MP3/WAV).
        duration: Audio duration in seconds (default 60.0).
        timbre: Sound generator type ('buzzer'/'click' or 'piano', default 'buzzer').
        click_duration: Duration of buzzer click in seconds (default 0.03 = 30ms).
        stagger_phase: If True (default), staggers start phase across voices to prevent common-multiple drops.
        numogram_order: Order / zone count of Numogram (default 10).
        note_duration: Fixed note duration in seconds if use_netspan_tempo=False (default None).
        use_netspan_tempo: If True (default), derives note duration from Net-Span ratio min(p, q)/max(p, q).
        sample_rate: Audio sampling rate in Hz (default 44100).
        f_min: Minimum frequency in Hz (default 33.0).
        f_max: Maximum frequency in Hz (default 3960.0).
        ana_is_high: Pitch mapping direction (default True).
        stereo: If True (default), spreads demon tracks across stereo field.
        skip_syzygy_and_occult: If True (default), skips [X] and [?] routes.

    Returns:
        Path to the generated MP3 or WAV audio file.
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
    total_samples = int(round((duration + 1.0) * sample_rate))

    master_l = np.zeros(total_samples, dtype=np.float32)
    master_r = np.zeros(total_samples, dtype=np.float32)

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

        # Calculate this demon's note tempo
        if use_netspan_tempo:
            demon_note_dur = calculate_demon_tempo(d, numogram_order=numogram_order)
        else:
            demon_note_dur = note_duration if note_duration is not None else 0.5

        # Phase offset to break artificial alignment at rational common multiples (e.g. 1.0s, 2.0s)
        if stagger_phase:
            offset_sec = (idx * 0.6180339887) % demon_note_dur
            start_sample_offset = int(round(offset_sec * sample_rate))
        else:
            start_sample_offset = 0

        active_tracks += 1
        track = np.zeros(total_samples, dtype=np.float32)

        if timbre in ("buzzer", "click"):
            # Minimalist clicking / buzzer blip ("滴一下")
            actual_click_dur = min(click_duration, demon_note_dur * 0.45)
            click_samples = max(16, int(round(actual_click_dur * sample_rate)))
            tt_click = np.arange(click_samples, dtype=np.float32) / float(sample_rate)
            fade_samples = min(int(0.002 * sample_rate), click_samples // 4)
            click_win = np.ones(click_samples, dtype=np.float32)
            if fade_samples > 0:
                click_win[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
                click_win[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, endpoint=False)

            cur_sample = start_sample_offset
            k = 0
            tempo_samples = max(1, int(round(demon_note_dur * sample_rate)))
            while cur_sample < total_samples:
                f = demon_freqs[k % len(demon_freqs)]
                two_pi_f = 2.0 * np.pi * f
                # Electronic buzzer tone: fundamental + subtle 3rd harmonic
                click_note = (
                    np.sin(two_pi_f * tt_click)
                    + 0.25 * np.sin(3.0 * two_pi_f * tt_click)
                ) * click_win

                end_idx = min(cur_sample + click_samples, total_samples)
                seg_len = end_idx - cur_sample
                track[cur_sample:end_idx] += click_note[:seg_len]
                cur_sample += tempo_samples
                k += 1

        else:
            # Piano tone model from music.py
            demon_note_len = max(1, int(round(demon_note_dur * sample_rate)))
            release_extra_sec = 0.6
            n_note_samples = int((demon_note_dur + release_extra_sec) * sample_rate)

            tt_note = np.arange(n_note_samples, dtype=np.float32) / float(sample_rate)
            att = np.minimum(1.0, tt_note / 0.005)
            rel = np.where(
                tt_note > demon_note_dur,
                np.exp(-(tt_note - demon_note_dur) * 12.0, dtype=np.float32),
                1.0,
            )
            env = att * rel * (80.0 / 127.0)

            decay_weights = [
                (hk, (a * np.exp(-tt_note * (1.5 + hk * 0.9), dtype=np.float32) * env).astype(np.float32))
                for hk, a in [(1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)]
            ]

            cur_sample = start_sample_offset
            k = 0
            while cur_sample < total_samples:
                f = demon_freqs[k % len(demon_freqs)]
                two_pi_f = 2.0 * np.pi * f
                note = sum(dw * np.sin(two_pi_f * hk * tt_note) for hk, dw in decay_weights)

                end_idx = min(cur_sample + n_note_samples, total_samples)
                seg_len = end_idx - cur_sample
                track[cur_sample:end_idx] += note[:seg_len]
                cur_sample += demon_note_len
                k += 1

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

    # Crop to exact duration and normalize to -0.9 dBFS
    master_l = master_l[:n_samples]
    master_r = master_r[:n_samples]
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
            "-i", "-",
            "-b:a", "192k",
            str(out_p),
        ]
        proc = subprocess.run(cmd, input=pcm_data.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding error: {proc.stderr.decode(errors='ignore')}")
    else:
        # Fallback to WAV format if ffmpeg is unavailable
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numogram Music Module")
    parser.add_argument("--audio", type=str, default="full_60s.mp3", help="Output audio file path")
    parser.add_argument("--demons", type=str, default=None, help="Comma-separated demon names/IDs to include")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60.0)")
    parser.add_argument("--fixed-tempo", action="store_true", help="Use fixed 0.5s tempo instead of Net-Span ratio")
    args = parser.parse_args()

    demon_list = [d.strip() for d in args.demons.split(",")] if args.demons else None
    out_file = generate_ritual_audio(
        demons=demon_list,
        output_path=args.audio,
        duration=args.duration,
        use_netspan_tempo=not args.fixed_tempo,
    )
    print(f"Generated ritual audio: {out_file}")
