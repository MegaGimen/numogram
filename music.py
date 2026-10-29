"""直接输入左右手的音符序列生成音乐：MIDI + 试听音频。

每一项写成 (偏移, 倍数, 拍数)，频率 = 偏移 × 倍数 × DO
  偏移：换八度用。1 = 表里的八度，2 = 高一个八度，0.5 = 低一个八度，0.25 = 低两个八度
  倍数：这个音是 do 的几倍（纯律）
        do 1    re 9/8    mi 5/4    fa 4/3    sol 3/2    la 5/3    si 15/8
  拍数：这个音持续几拍
和弦（几个音同时按）：偏移和倍数各写成一个元组，一一对应，比如
  ((0.5, 0.5, 0.5), (1, 5/4, 3/2), 4) = 低一个八度的 do mi sol 一起按 4 拍
休止：写 (0, 0, 拍数)，比如 (0, 0, 1) 是一拍不出声
"""
import math, struct, wave
import numpy as np

DO = 264        # do 的频率（Hz）
TEMPO = 96      # 每分钟拍数，4/4 拍
TPB = 480       # MIDI 每拍 tick 数

# ---------------- 输入：右手（旋律），每拍一个音，在高一个八度 ----------------
RIGHT = [
    (2, 1, 1), (2, 9/8, 1), (2, 5/4, 1), (2, 1, 1),         # 第 1 小节 | I   1 2 3 1
    (2, 4/3, 1), (2, 3/2, 1), (2, 5/3, 1), (2, 4/3, 1),     # 第 2 小节 | IV  4 5 6 4
    (2, 3/2, 1), (2, 5/3, 1), (2, 15/8, 1), (2, 3/2, 1),    # 第 3 小节 | V   5 6 7 5
    (2, 3/2, 4),                                            # 第 4 小节 | V   5 — — —（半终止）
    (2, 1, 1), (2, 9/8, 1), (2, 5/4, 1), (2, 1, 1),         # 第 5 小节 | I   1 2 3 1
    (2, 4/3, 1), (2, 3/2, 1), (2, 5/3, 1), (2, 4/3, 1),     # 第 6 小节 | IV  4 5 6 4
    (2, 3/2, 1), (2, 4/3, 1), (2, 9/8, 1), (1, 15/8, 1),    # 第 7 小节 | V   5 4 2 7̣
    (2, 1, 4),                                              # 第 8 小节 | I   1 — — —（完全终止）
]

# ---------------- 输入：左手（阿尔贝蒂低音），每个音半拍 ----------------
# 低、高、中、高，每小节重复两遍
LEFT = [
    (0.5, 1, 0.5), (0.5, 3/2, 0.5), (0.5, 5/4, 0.5), (0.5, 3/2, 0.5),      # 第 1 小节 | I   do sol mi sol
    (0.5, 1, 0.5), (0.5, 3/2, 0.5), (0.5, 5/4, 0.5), (0.5, 3/2, 0.5),
    (0.5, 4/3, 0.5), (1, 1, 0.5), (0.5, 5/3, 0.5), (1, 1, 0.5),            # 第 2 小节 | IV  fa do la do
    (0.5, 4/3, 0.5), (1, 1, 0.5), (0.5, 5/3, 0.5), (1, 1, 0.5),
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5), # 第 3 小节 | V   sol re si re
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5),
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5), # 第 4 小节 | V   sol re si re
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5),
    (0.5, 1, 0.5), (0.5, 3/2, 0.5), (0.5, 5/4, 0.5), (0.5, 3/2, 0.5),      # 第 5 小节 | I   do sol mi sol
    (0.5, 1, 0.5), (0.5, 3/2, 0.5), (0.5, 5/4, 0.5), (0.5, 3/2, 0.5),
    (0.5, 4/3, 0.5), (1, 1, 0.5), (0.5, 5/3, 0.5), (1, 1, 0.5),            # 第 6 小节 | IV  fa do la do
    (0.5, 4/3, 0.5), (1, 1, 0.5), (0.5, 5/3, 0.5), (1, 1, 0.5),
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5), # 第 7 小节 | V   sol re si re
    (0.25, 3/2, 0.5), (0.5, 9/8, 0.5), (0.25, 15/8, 0.5), (0.5, 9/8, 0.5),
    ((0.25, 0.5, 0.5, 0.5), (1, 1, 5/4, 3/2), 4),                          # 第 8 小节 | I   do do mi sol 一起按 4 拍
]

# ---------------- 把序列展开成音符事件 ----------------
def expand(seq, hand):
    events, t = [], 0.0
    for offsets, ratios, beats in seq:
        if not isinstance(offsets, tuple):
            offsets, ratios = (offsets,), (ratios,)
        freqs = [round(o * r * DO, 6) for o, r in zip(offsets, ratios)]
        if all(f == 0 for f in freqs):   # 休止：不发声，只往后数拍
            t += beats
            continue
        pos = t % 4
        if hand == "right":
            vel = 92 if pos in (0, 2) else 78        # 重拍强，轻拍弱
            dur = beats * 0.95
        else:
            vel = 70 if (t * 2) % 2 == 0 else 62     # 正拍稍强，后半拍稍弱
            dur = beats * (0.98 if len(freqs) > 1 else 0.96)
        for f in freqs:
            events.append((t, dur, f, vel))
        t += beats
    return events, t

right_ev, r_len = expand(RIGHT, "right")
left_ev, l_len = expand(LEFT, "left")
assert r_len == l_len == 32, (r_len, l_len)   # 两只手都是 8 小节 × 4 拍

# ---------------- 写 MIDI：频率 → 最近的琴键 + 弯音微调 ----------------
BEND_RANGE = 2   # 弯音范围 ±2 个半音

def freq_to_midi(f):
    x = 69 + 12 * math.log2(f / 440)
    key = round(x)
    cents = (x - key) * 100
    bend = int(round(8192 + cents / (BEND_RANGE * 100) * 8192))
    return key, max(0, min(16383, bend)), cents

def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)

def track(events, channels, name):
    # 每个音轮流用不同通道，这样每个音的弯音互不影响（和弦里的音也能各自调）
    msgs = []
    setup = b""
    for ch in channels:
        setup += bytes([0, 0xC0 | ch, 0])                                   # 钢琴音色
        setup += bytes([0, 0xB0 | ch, 101, 0, 0, 0xB0 | ch, 100, 0])        # 选择弯音范围参数
        setup += bytes([0, 0xB0 | ch, 6, BEND_RANGE, 0, 0xB0 | ch, 38, 0])  # 设为 ±2 半音
    for i, (start, dur, f, vel) in enumerate(events):
        ch = channels[i % len(channels)]
        key, bend, _ = freq_to_midi(f)
        on, off = int(round(start * TPB)), int(round((start + dur) * TPB))
        msgs.append((on, 1, bytes([0xE0 | ch, bend & 0x7F, bend >> 7])))   # 先调音高
        msgs.append((on, 2, bytes([0x90 | ch, key, vel])))                 # 再按下
        msgs.append((off, 0, bytes([0x80 | ch, key, 0])))                  # 松开
    msgs.sort(key=lambda m: (m[0], m[1]))
    data = b"\x00\xff\x03" + vlq(len(name.encode())) + name.encode() + setup
    last = 0
    for tick, _, m in msgs:
        data += vlq(tick - last) + m
        last = tick
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data

tempo_trk = b"\x00\xff\x51\x03" + struct.pack(">I", 60_000_000 // TEMPO)[1:]
tempo_trk += b"\x00\xff\x58\x04\x04\x02\x18\x08"
tempo_trk += b"\x00\xff\x2f\x00"
tempo_trk = b"MTrk" + struct.pack(">I", len(tempo_trk)) + tempo_trk

with open("classical_period_just.mid", "wb") as fh:
    fh.write(b"MThd" + struct.pack(">IHHH", 6, 1, 3, TPB))
    fh.write(tempo_trk)
    fh.write(track(right_ev, [0], "Right hand (melody)"))
    fh.write(track(left_ev, [1, 2, 3, 4, 5, 6, 7, 8], "Left hand (Alberti bass)"))

# ---------------- 合成试听音频：直接用频率发声 ----------------
SR = 44100
spb = 60 / TEMPO
buf = np.zeros(int((32 * spb + 2.0) * SR))

def tone(f, secs, vel):
    n = int((secs + 0.6) * SR)
    tt = np.arange(n) / SR
    w = sum(a * np.sin(2 * np.pi * f * k * tt) * np.exp(-tt * (1.5 + k * 0.9))
            for k, a in [(1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)])
    att = np.minimum(1, tt / 0.005)
    rel = np.where(tt > secs, np.exp(-(tt - secs) * 12), 1.0)
    return w * att * rel * (vel / 127)

for start, dur, f, vel in right_ev + left_ev:
    s = int(start * spb * SR)
    w = tone(f, dur * spb, vel)
    buf[s:s + len(w)] += w[: len(buf) - s]

buf = buf / np.max(np.abs(buf)) * 0.85
with wave.open("classical_period_just.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((buf * 32767).astype(np.int16).tobytes())

# 打印每个用到的频率和钢琴键差多少
print("频率(Hz)  最近琴键  偏差(音分)")
names = "C C# D D# E F F# G G# A A# B".split()
for f in sorted({e[2] for e in right_ev + left_ev}):
    key, _, cents = freq_to_midi(f)
    print(f"{f:>8g}  {names[key % 12] + str(key // 12 - 1):>6}  {cents:+6.1f}")
