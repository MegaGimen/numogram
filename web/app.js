/**
 * NUMOGRAM HOPFIELD ENERGY DESCENT & TEMPERATURE DYNAMICS
 * Exact 1:1 geometry aligned to reference diagram (864 x 1152)
 */

// Zone metadata
const ZONES = [
  { id: 0, demon: "Utunul", role: "plex", name: "The Abyss", gateVal: 0, dst: 0, syz: 9, tractor: 9 },
  { id: 1, demon: "Murrumur", role: "torque", name: "Initiation", gateVal: 1, dst: 1, syz: 8, tractor: 7 },
  { id: 2, demon: "Oddubb", role: "torque", name: "Centrifugal", gateVal: 3, dst: 3, syz: 7, tractor: 5 },
  { id: 3, demon: "Pabbuk", role: "warp", name: "Warp Alpha", gateVal: 6, dst: 6, syz: 6, tractor: 3 },
  { id: 4, demon: "Pupana", role: "torque", name: "Delta Incline", gateVal: 10, dst: 1, syz: 5, tractor: 1 },
  { id: 5, demon: "Bubbamu", role: "torque", name: "Torque Beta", gateVal: 15, dst: 6, syz: 4, tractor: 1 },
  { id: 6, demon: "Tchattuk", role: "warp", name: "Warp Omega", gateVal: 21, dst: 3, syz: 3, tractor: 3 },
  { id: 7, demon: "Puppo", role: "torque", name: "Hold Gravity", gateVal: 28, dst: 1, syz: 2, tractor: 5 },
  { id: 8, demon: "Minommo", role: "torque", name: "Abyssal Gate", gateVal: 36, dst: 9, syz: 1, tractor: 7 },
  { id: 9, demon: "Katak", role: "plex", name: "Barker's Spiral", gateVal: 45, dst: 9, syz: 0, tractor: 9 }
];

// The 5 CCRU Syzygies (sum = 9)
const SYZYGIES = [
  { hi: 9, lo: 0, tr: 9, kind: "plex" },
  { hi: 8, lo: 1, tr: 7, kind: "torque" },
  { hi: 7, lo: 2, tr: 5, kind: "torque" },
  { hi: 6, lo: 3, tr: 3, kind: "warp" },
  { hi: 5, lo: 4, tr: 1, kind: "torque" }
];

// Exact circle coordinates measured directly from reference image (864 x 1152)
const COORDS = {
  3: { x: 400, y: 212, r: 46, tri: "right" },
  6: { x: 514, y: 214, r: 46, tri: "left" },
  5: { x: 276, y: 435, r: 46, tri: "down-left" },
  4: { x: 207, y: 521, r: 46, tri: "up-left" },
  2: { x: 638, y: 362, r: 45, tri: "down-right" },
  7: { x: 700, y: 451, r: 45, tri: "up-left" },
  1: { x: 450, y: 703, r: 44, tri: "up-down" },
  8: { x: 447, y: 806, r: 43, tri: "up" },
  9: { x: 445, y: 932, r: 44, tri: "down" },
  0: { x: 442, y: 1038, r: 43, tri: "up" }
};

// Small Gate Circles (numbers on the small rings)
const GATES = {
  15: { x: 530, y: 59, r: 20, z: 5, dst: 6 },
  6:  { x: 458, y: 133, r: 18, z: 3, dst: 6 },
  21: { x: 435, y: 279, r: 20, z: 6, dst: 3 },
  3:  { x: 378, y: 284, r: 20, z: 2, dst: 3 },
  10: { x: 354, y: 693, r: 20, z: 4, dst: 1 },
  1:  { x: 475, y: 606, r: 20, z: 1, dst: 1 },
  28: { x: 534, y: 649, r: 20, z: 7, dst: 1 },
  36: { x: 502, y: 854, r: 20, z: 8, dst: 9 },
  45: { x: 380, y: 904, r: 20, z: 9, dst: 9 }
};

// Hopfield Parameters (exact match to hopfield_numogram.py)
const PARAMS = {
  lam_syz: 1.0,
  lam_curr: 0.35,
  lam_bin: 0.8,
  lam_chan: 0.45,
  gate_gain: 12.0,
  lr: 0.08
};

// Thresholds for Gates 1..9: thresh = 0.12 + 0.78 * (Gt / 45.0)
const THRESHOLDS = new Float32Array(10);
for (let z = 1; z <= 9; z++) {
  const gt = (z * (z + 1)) / 2;
  THRESHOLDS[z] = 0.12 + 0.78 * (gt / 45.0);
}

// App State
const state = {
  s: new Float32Array(10),
  gateOpen: new Float32Array(10),
  stepCount: 0,
  running: false,
  speedFps: 20,
  palette: "coolwarm",
  activePreset: "zone1",
  history: [],
  energy: 0,
  terms: { syz: 0, curr: 0, bin: 0, chan: 0 }
};

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// -------------------------------------------------------------
// Continuous Hopfield Relaxation Step
// -------------------------------------------------------------
function computeStep() {
  const { lam_syz, lam_curr, lam_bin, lam_chan, lr, gate_gain } = PARAMS;
  const s = state.s;
  const grad = new Float32Array(10);

  // 1. Gate conductances g[z]
  for (let z = 1; z <= 9; z++) {
    const mag = Math.abs(s[z]);
    state.gateOpen[z] = sigmoid(gate_gain * (mag - THRESHOLDS[z]));
  }

  // 2. Syzygy anti-phase gradient: E_syz = 0.5 * (sA + sB)^2
  for (const { hi, lo } of SYZYGIES) {
    const sum = s[hi] + s[lo];
    grad[hi] += lam_syz * sum;
    grad[lo] += lam_syz * sum;
  }

  // 3. Current tractor pump gradient: E_curr = - sT * (sA - sB)^2
  for (const { hi, lo, tr } of SYZYGIES) {
    const diff = s[hi] - s[lo];
    const sT = s[tr];
    grad[tr] += -lam_curr * (diff * diff);
    grad[hi] += -lam_curr * (2 * sT * diff);
    grad[lo] += -lam_curr * (2 * sT * diff);
  }

  // 4. Binary well gradient: E_bin = 0.25 * (s^2 - 1)^2
  for (let z = 0; z < 10; z++) {
    grad[z] += lam_bin * s[z] * (s[z] * s[z] - 1.0);
  }

  // 5. Channel ferromagnetic coupling: E_chan = - g * s_z * s_dst
  for (let z = 1; z <= 9; z++) {
    const dst = ZONES[z].dst;
    const g = state.gateOpen[z];
    grad[z] += -lam_chan * g * s[dst];
    grad[dst] += -lam_chan * g * s[z];
  }

  // Gradient update + clamp [-1, 1]
  for (let z = 0; z < 10; z++) {
    const nextVal = s[z] - lr * grad[z];
    s[z] = Math.max(-1.0, Math.min(1.0, nextVal));
  }

  state.stepCount++;
  recordSnapshot();
}

function computeEnergy() {
  const s = state.s;
  let e_syz = 0;
  for (const { hi, lo } of SYZYGIES) {
    const sum = s[hi] + s[lo];
    e_syz += 0.5 * (sum * sum);
  }

  let e_curr = 0;
  for (const { hi, lo, tr } of SYZYGIES) {
    const diff = s[hi] - s[lo];
    e_curr -= s[tr] * (diff * diff);
  }

  let e_bin = 0;
  for (let z = 0; z < 10; z++) {
    const sq = s[z] * s[z] - 1.0;
    e_bin += 0.25 * (sq * sq);
  }

  let e_chan = 0;
  for (let z = 1; z <= 9; z++) {
    const dst = ZONES[z].dst;
    const g = state.gateOpen[z];
    e_chan -= g * s[z] * s[dst];
  }

  state.terms = {
    syz: PARAMS.lam_syz * e_syz,
    curr: PARAMS.lam_curr * e_curr,
    bin: PARAMS.lam_bin * e_bin,
    chan: PARAMS.lam_chan * e_chan
  };
  state.energy = state.terms.syz + state.terms.curr + state.terms.bin + state.terms.chan;
}

function recordSnapshot() {
  computeEnergy();
  
  let openCount = 0;
  for (let z = 1; z <= 9; z++) {
    if (state.gateOpen[z] > 0.5) openCount++;
  }

  const snap = {
    step: state.stepCount,
    s: new Float32Array(state.s),
    gateOpen: new Float32Array(state.gateOpen),
    E: state.energy,
    terms: { ...state.terms },
    openCount
  };

  state.history.push(snap);
  if (state.history.length > 500) state.history.shift();

  updateUI();
}

// -------------------------------------------------------------
// Temperature Colormap
// -------------------------------------------------------------
const PALETTES = {
  // Coolwarm: -1 Cyan/Blue, 0 Slate Dark, +1 Hot Amber/Red
  coolwarm: (val) => {
    const u = (Math.max(-1, Math.min(1, val)) + 1) / 2;
    let r, g, b;
    if (u < 0.5) {
      const f = u / 0.5;
      r = Math.floor(0 + 40 * f);
      g = Math.floor(180 - 100 * f);
      b = Math.floor(255 - 130 * f);
    } else {
      const f = (u - 0.5) / 0.5;
      r = Math.floor(40 + 215 * f);
      g = Math.floor(80 - 40 * f);
      b = Math.floor(125 - 100 * f);
    }
    return [r, g, b];
  },
  // Inferno
  inferno: (val) => {
    const u = Math.max(0, Math.min(1, Math.abs(val)));
    let r, g, b;
    if (u < 0.3) {
      const f = u / 0.3;
      r = Math.floor(20 + 80 * f);
      g = Math.floor(5 * f);
      b = Math.floor(35 + 85 * f);
    } else if (u < 0.7) {
      const f = (u - 0.3) / 0.4;
      r = Math.floor(100 + 130 * f);
      g = Math.floor(5 + 90 * f);
      b = Math.floor(120 - 90 * f);
    } else {
      const f = (u - 0.7) / 0.3;
      r = 255;
      g = Math.floor(95 + 160 * f);
      b = Math.floor(30 + 120 * f);
    }
    return [r, g, b];
  },
  // Cyber Phosphor Green to Magenta
  cyber: (val) => {
    const u = (Math.max(-1, Math.min(1, val)) + 1) / 2;
    let r, g, b;
    if (u < 0.5) {
      const f = u / 0.5;
      r = Math.floor(10 + 20 * f);
      g = Math.floor(220 + 20 * f);
      b = Math.floor(255 - 160 * f);
    } else {
      const f = (u - 0.5) / 0.5;
      r = Math.floor(30 + 225 * f);
      g = Math.floor(240 - 210 * f);
      b = Math.floor(95 + 120 * f);
    }
    return [r, g, b];
  },
  // Solar Flare
  solar: (val) => {
    const u = Math.max(0, Math.min(1, Math.abs(val)));
    const r = Math.floor(30 + 225 * u);
    const g = Math.floor(20 + 210 * (u * u));
    const b = Math.floor(25 * (u * u * u));
    return [r, g, b];
  }
};

function getNodeColor(val) {
  const [r, g, b] = (PALETTES[state.palette] || PALETTES.coolwarm)(val);
  return `rgb(${r}, ${g}, ${b})`;
}

// -------------------------------------------------------------
// Thermal Canvas 2D Diffusion
// -------------------------------------------------------------
const thermalCanvas = document.getElementById("thermal-canvas");
const thermalCtx = thermalCanvas.getContext("2d");

const offW = 108;
const offH = 144;
const offCanvas = document.createElement("canvas");
offCanvas.width = offW;
offCanvas.height = offH;
const offCtx = offCanvas.getContext("2d");

function renderThermalField() {
  const imgData = offCtx.createImageData(offW, offH);
  const data = imgData.data;
  const sx = offW / 864;
  const sy = offH / 1152;

  const pts = [];
  for (let z = 0; z < 10; z++) {
    pts.push({
      x: COORDS[z].x * sx,
      y: COORDS[z].y * sy,
      val: state.s[z]
    });
  }

  const sigma2 = 2 * 14 * 14;
  const palFn = PALETTES[state.palette] || PALETTES.coolwarm;
  let p = 0;

  for (let y = 0; y < offH; y++) {
    for (let x = 0; x < offW; x++) {
      let fSigned = 0;
      let fMag = 0;

      for (let i = 0; i < 10; i++) {
        const dx = x - pts[i].x;
        const dy = y - pts[i].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 600) {
          const w = Math.exp(-d2 / sigma2);
          fSigned += pts[i].val * w;
          fMag += Math.abs(pts[i].val) * w;
        }
      }

      const inputVal = state.palette === "coolwarm" || state.palette === "cyber" ? fSigned : fMag;
      const [r, g, b] = palFn(inputVal);
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = Math.min(210, Math.floor(fMag * 200));
      p += 4;
    }
  }

  offCtx.putImageData(imgData, 0, 0);

  thermalCtx.clearRect(0, 0, thermalCanvas.width, thermalCanvas.height);
  thermalCtx.imageSmoothingEnabled = true;
  thermalCtx.drawImage(offCanvas, 0, 0, thermalCanvas.width, thermalCanvas.height);
}

// -------------------------------------------------------------
// Flow Particles (Along Curves & Channels)
// -------------------------------------------------------------
const particleCanvas = document.getElementById("particle-canvas");
const particleCtx = particleCanvas.getContext("2d");

class FluxParticle {
  constructor(curve, color, gateId = null) {
    this.curve = curve;
    this.color = color;
    this.gateId = gateId;
    this.t = Math.random();
    this.speed = 0.007 + Math.random() * 0.01;
    this.size = 1.6 + Math.random() * 1.8;
  }

  update() {
    this.t += this.speed;
    if (this.t > 1.0) this.t = 0;
  }

  draw(ctx) {
    if (this.gateId !== null && state.gateOpen[this.gateId] < 0.5) return;
    const pt = this.curve(this.t);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

let particles = [];

function initParticles() {
  particles = [];

  // Helper for quadratic bezier
  const qBezier = (p0, p1, p2) => (t) => ({
    x: (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * p1.x + t * t * p2.x,
    y: (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y
  });

  // 1. Current 7 -> 5 (Horizontal plume)
  const c75 = (t) => ({ x: 655 + (325 - 655) * t, y: 435 });
  for (let i = 0; i < 4; i++) particles.push(new FluxParticle(c75, "#15ff75"));

  // 2. Current 4/5 -> 1 (Diagonal sweeping plume)
  const c41 = qBezier({ x: 260, y: 495 }, { x: 340, y: 620 }, { x: 418, y: 675 });
  for (let i = 0; i < 4; i++) particles.push(new FluxParticle(c41, "#15ff75"));

  // 3. Current 1 -> 7 (Upward curve plume)
  const c17 = qBezier({ x: 480, y: 755 }, { x: 670, y: 720 }, { x: 700, y: 495 });
  for (let i = 0; i < 4; i++) particles.push(new FluxParticle(c17, "#15ff75"));

  // 4. Current 6 -> 3 (Warp curl)
  const c63 = qBezier({ x: 490, y: 255 }, { x: 445, y: 285 }, { x: 435, y: 235 });
  for (let i = 0; i < 3; i++) particles.push(new FluxParticle(c63, "#15ff75"));

  // 5. Current 9 -> 9 (Plex curl)
  const c99 = qBezier({ x: 465, y: 965 }, { x: 485, y: 1000 }, { x: 445, y: 975 });
  for (let i = 0; i < 3; i++) particles.push(new FluxParticle(c99, "#15ff75"));

  // Channels with Gates
  // Gate 15: 5 -> Gt15 -> 6
  const cG15 = qBezier({ x: 276, y: 389 }, { x: 300, y: 40 }, { x: 548, y: 178 });
  for (let i = 0; i < 3; i++) particles.push(new FluxParticle(cG15, "#ff2bd6", 5));

  // Gate 3: 2 -> Gt3 -> 3
  const cG3 = qBezier({ x: 638, y: 407 }, { x: 460, y: 420 }, { x: 385, y: 256 });
  for (let i = 0; i < 3; i++) particles.push(new FluxParticle(cG3, "#ff2bd6", 2));

  // Gate 36: 8 -> Gt36 -> 9
  const cG36 = qBezier({ x: 470, y: 840 }, { x: 515, y: 865 }, { x: 480, y: 900 });
  for (let i = 0; i < 3; i++) particles.push(new FluxParticle(cG36, "#ff2bd6", 8));
}

function renderParticles() {
  particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
  for (const p of particles) {
    p.update();
    p.draw(particleCtx);
  }
}

// -------------------------------------------------------------
// SVG Graph Builder (Strictly Matching a08cc2335503fcd657a2298ff5d4b118.jpg)
// -------------------------------------------------------------
const svg = document.getElementById("numogram-svg");

function buildSVG() {
  svg.innerHTML = `
    <defs>
      <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur1"/>
        <feGaussianBlur in="SourceGraphic" stdDeviation="10" result="blur2"/>
        <feMerge>
          <feMergeNode in="blur2"/>
          <feMergeNode in="blur1"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>

      <!-- Plume Gradients -->
      <linearGradient id="grad-plume-75" x1="1" y1="0" x2="0" y2="0">
        <stop offset="0%" stop-color="#15ff75" stop-opacity="0.1"/>
        <stop offset="60%" stop-color="#15ff75" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#15ff75" stop-opacity="0.95"/>
      </linearGradient>

      <linearGradient id="grad-plume-41" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#15ff75" stop-opacity="0.15"/>
        <stop offset="70%" stop-color="#15ff75" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#15ff75" stop-opacity="0.95"/>
      </linearGradient>

      <linearGradient id="grad-plume-17" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#15ff75" stop-opacity="0.15"/>
        <stop offset="70%" stop-color="#15ff75" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#15ff75" stop-opacity="0.95"/>
      </linearGradient>

      <linearGradient id="grad-plume-63" x1="1" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#15ff75" stop-opacity="0.2"/>
        <stop offset="100%" stop-color="#15ff75" stop-opacity="0.9"/>
      </linearGradient>

      <linearGradient id="grad-plume-99" x1="1" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#15ff75" stop-opacity="0.2"/>
        <stop offset="100%" stop-color="#15ff75" stop-opacity="0.9"/>
      </linearGradient>
    </defs>
  `;

  // 1. THE 5 MASSIVE GLOWING CURRENTS (PLUMES)
  const plumeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  plumeLayer.setAttribute("id", "plume-layer");
  plumeLayer.innerHTML = `
    <!-- Current 7 -> 5 (Horizontal Arrow Plume) -->
    <path id="plume-7-5" d="
      M 655 425
      L 380 425
      L 380 405
      L 322 435
      L 380 465
      L 380 445
      L 655 445
      Z
    " fill="url(#grad-plume-75)" opacity="0.85" filter="url(#neonGlow)"/>

    <!-- Current 4/5 -> 1 (Trumpet Plume into Zone 1) -->
    <path id="plume-4-1" d="
      M 285 470
      Q 330 535 415 670
      L 395 650
      L 415 678
      L 428 655
      L 412 666
      Q 325 585 240 525
      Z
    " fill="url(#grad-plume-41)" opacity="0.85" filter="url(#neonGlow)"/>

    <!-- Current 1 -> 7 (Upward Sweeping Plume into Zone 7) -->
    <path id="plume-1-7" d="
      M 480 750
      Q 610 740 690 515
      L 680 525
      L 700 496
      L 708 522
      L 698 515
      Q 640 765 480 770
      Z
    " fill="url(#grad-plume-17)" opacity="0.85" filter="url(#neonGlow)"/>

    <!-- Current 6 -> 3 (Warp Curl Plume) -->
    <path id="plume-6-3" d="
      M 495 255
      Q 470 290 435 270
      L 442 260
      L 435 240
      L 420 255
      L 430 258
      Q 460 275 485 255
      Z
    " fill="url(#grad-plume-63)" opacity="0.85" filter="url(#neonGlow)"/>

    <!-- Current 9 -> 9 (Plex Curl Plume) -->
    <path id="plume-9-9" d="
      M 465 965
      Q 490 990 475 1010
      Q 450 1005 445 980
      L 438 985
      L 448 970
      L 458 980
      L 450 982
      Q 465 995 475 985
      Q 475 975 465 965
      Z
    " fill="url(#grad-plume-99)" opacity="0.85" filter="url(#neonGlow)"/>
  `;
  svg.appendChild(plumeLayer);

  // 2. CHANNELS WITH GATES (Thin Green Lines + Gate Loops)
  const channelLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  channelLayer.setAttribute("id", "channel-layer");
  channelLayer.innerHTML = `
    <!-- Gate 15: From Zone 5 top, arching over top into Gate 15 and into Zone 6 -->
    <path id="chan-15" class="gate-line" d="
      M 276 389
      C 240 220, 310 35, 510 59
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 548 70 Q 565 110 546 174" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="546,174 540,162 553,165" fill="#15ff75"/>

    <!-- Gate 6: From Zone 3 top into Gate 6 and down into Zone 6 -->
    <path id="chan-6" class="gate-line" d="
      M 425 174 Q 440 145 443 138
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 474 138 Q 490 150 495 174" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="495,174 485,165 498,162" fill="#15ff75"/>

    <!-- Gate 21: From Zone 6 bottom-left into Gate 21 and into Zone 3 -->
    <path id="chan-21" class="gate-line" d="
      M 485 248 Q 460 268 448 274
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 422 274 Q 418 265 422 254" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="422,254 416,264 428,262" fill="#15ff75"/>

    <!-- Gate 3: Long sweep from Zone 2 bottom, under Warp into Gate 3 and into Zone 3 -->
    <path id="chan-3" class="gate-line" d="
      M 638 407
      C 600 440, 480 380, 396 295
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 378 264 L 385 256" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="385,256 376,263 388,266" fill="#15ff75"/>

    <!-- Gate 10: From Zone 4 bottom, down-right into Gate 10 and straight into Zone 1 -->
    <path id="chan-10" class="gate-line" d="
      M 225 560 Q 270 693 334 693
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 374 693 L 406 703" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="406,703 395,696 397,708" fill="#15ff75"/>

    <!-- Gate 1: Teardrop Loop on top of Zone 1 -->
    <path id="chan-1" class="gate-line" d="
      M 438 662
      C 438 620, 458 595, 470 595
      C 485 595, 498 625, 470 662
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="470,662 468,650 478,655" fill="#15ff75"/>

    <!-- Gate 28: From Zone 7 bottom-left, into Gate 28 and into Zone 1 top-right -->
    <path id="chan-28" class="gate-line" d="
      M 670 485 Q 580 580 545 635
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 522 660 L 485 678" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="485,678 497,675 493,686" fill="#15ff75"/>

    <!-- Gate 36: From Zone 8 bottom-right, into Gate 36 and into Zone 9 -->
    <path id="chan-36" class="gate-line" d="
      M 475 838 Q 490 845 496 850
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <path d="M 500 870 Q 495 885 480 900" fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="480,900 488,890 493,901" fill="#15ff75"/>

    <!-- Gate 45: Loop to the left of Zone 9 -->
    <path id="chan-45" class="gate-line" d="
      M 410 918
      C 370 890, 360 915, 395 922
      C 410 925, 415 935, 415 940
    " fill="none" stroke="#15ff75" stroke-width="2"/>
    <polygon points="415,940 405,932 415,928" fill="#15ff75"/>
  `;
  svg.appendChild(channelLayer);

  // 3. SMALL GATE CIRCLES (With green background and numbers inside)
  const gateCircleLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gateCircleLayer.setAttribute("id", "gate-circle-layer");
  for (const [id, g] of Object.entries(GATES)) {
    const grp = document.createElementNS("http://www.w3.org/2000/svg", "g");
    grp.setAttribute("class", "gate-circle-group");
    grp.setAttribute("id", `gt-grp-${id}`);

    // Outer circle
    const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circ.setAttribute("cx", g.x);
    circ.setAttribute("cy", g.y);
    circ.setAttribute("r", g.r);
    circ.setAttribute("fill", "#050a06");
    circ.setAttribute("stroke", "#15ff75");
    circ.setAttribute("stroke-width", "2");
    circ.setAttribute("id", `gt-circ-${id}`);
    grp.appendChild(circ);

    // Number text
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", g.x);
    txt.setAttribute("y", g.y + 1);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("dominant-baseline", "central");
    txt.setAttribute("font-family", "JetBrains Mono, monospace");
    txt.setAttribute("font-size", g.r > 19 ? "14" : "12");
    txt.setAttribute("font-weight", "600");
    txt.setAttribute("fill", "#15ff75");
    txt.textContent = id;
    grp.appendChild(txt);

    gateCircleLayer.appendChild(grp);
  }
  svg.appendChild(gateCircleLayer);

  // 4. THE 10 LARGE ZONE CIRCLES (With exact radii, internal triangles, and numbers)
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  nodeLayer.setAttribute("id", "node-layer");

  for (let z = 0; z < 10; z++) {
    const c = COORDS[z];
    const grp = document.createElementNS("http://www.w3.org/2000/svg", "g");
    grp.setAttribute("class", "zone-node-group");
    grp.setAttribute("data-zone", z);
    grp.setAttribute("transform", `translate(${c.x}, ${c.y})`);

    // Temperature glow halo
    const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    halo.setAttribute("class", "zone-halo");
    halo.setAttribute("r", String(c.r + 8));
    halo.setAttribute("fill", "#15ff75");
    halo.setAttribute("opacity", "0.15");
    halo.setAttribute("filter", "url(#neonGlow)");
    grp.appendChild(halo);

    // Large main circle
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "zone-circle");
    circle.setAttribute("r", String(c.r));
    circle.setAttribute("fill", "#040805");
    circle.setAttribute("stroke", "#15ff75");
    circle.setAttribute("stroke-width", "2.6");
    grp.appendChild(circle);

    // Internal Direction Triangle(s) matching reference image
    if (c.tri === "right") {
      // Zone 3: points right
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "18,-6 30,0 18,6");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "left") {
      // Zone 6: points left
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "-18,-6 -30,0 -18,6");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "down-left") {
      // Zone 5: points bottom-left
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "-16,24 -26,14 -24,26");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "up-left") {
      // Zone 4 & 7: points top-left
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "-14,-24 -24,-16 -24,-26");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "down-right") {
      // Zone 2: points bottom-right
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "16,24 26,14 24,26");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "up-down") {
      // Zone 1: Two triangles (one up, one down)
      const triUp = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      triUp.setAttribute("points", "-6,-22 0,-30 6,-22");
      triUp.setAttribute("fill", "#15ff75");
      grp.appendChild(triUp);

      const triDown = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      triDown.setAttribute("points", "-6,22 0,30 6,22");
      triDown.setAttribute("fill", "#15ff75");
      grp.appendChild(triDown);
    } else if (c.tri === "up") {
      // Zone 8, 0: points up
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "-6,-22 0,-30 6,-22");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    } else if (c.tri === "down") {
      // Zone 9: points down
      const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      tri.setAttribute("points", "-6,22 0,30 6,22");
      tri.setAttribute("fill", "#15ff75");
      grp.appendChild(tri);
    }

    // Zone Number in center
    const num = document.createElementNS("http://www.w3.org/2000/svg", "text");
    num.setAttribute("class", "zone-number");
    num.setAttribute("y", "0");
    num.setAttribute("text-anchor", "middle");
    num.setAttribute("dominant-baseline", "central");
    num.setAttribute("font-family", "JetBrains Mono, monospace");
    num.setAttribute("font-size", "22");
    num.setAttribute("font-weight", "700");
    num.setAttribute("fill", "#15ff75");
    num.textContent = z;
    grp.appendChild(num);

    // Knock Zone on Click!
    grp.addEventListener("click", () => {
      if (Math.abs(state.s[z]) < 0.2) {
        state.s[z] = 1.0;
      } else if (state.s[z] > 0) {
        state.s[z] = -1.0;
      } else {
        state.s[z] = 1.0;
      }
      recordSnapshot();
    });

    nodeLayer.appendChild(grp);
  }
  svg.appendChild(nodeLayer);
}

// -------------------------------------------------------------
// UI Updates & HUD
// -------------------------------------------------------------
function buildHUD() {
  const vGrid = document.getElementById("vector-grid");
  vGrid.innerHTML = "";
  for (let z = 0; z < 10; z++) {
    const cell = document.createElement("div");
    cell.className = "node-cell";
    cell.id = `vec-cell-${z}`;
    cell.innerHTML = `
      <span class="c-id">Z${z}</span>
      <span class="c-val" id="vec-val-${z}">0.00</span>
    `;
    cell.addEventListener("click", () => {
      state.s[z] = state.s[z] > 0 ? -1.0 : 1.0;
      recordSnapshot();
    });
    vGrid.appendChild(cell);
  }

  const syzList = document.getElementById("syzygy-compact");
  syzList.innerHTML = "";
  for (const { hi, lo, tr, kind } of SYZYGIES) {
    const row = document.createElement("div");
    row.className = `syz-compact-row ${kind}`;
    row.innerHTML = `
      <span class="p-title">${hi}::${lo}</span>
      <span class="p-tr">➔ Tr:${tr}</span>
      <span class="p-tens" id="syz-t-${hi}-${lo}">张力: 0.00</span>
    `;
    syzList.appendChild(row);
  }
}

function updateUI() {
  document.getElementById("val-step").textContent = state.stepCount;
  document.getElementById("val-energy").textContent = (state.energy >= 0 ? "+" : "") + state.energy.toFixed(3);

  // Energy terms
  document.getElementById("term-syz").textContent = (state.terms.syz >= 0 ? "+" : "") + state.terms.syz.toFixed(3);
  document.getElementById("term-curr").textContent = (state.terms.curr >= 0 ? "+" : "") + state.terms.curr.toFixed(3);
  document.getElementById("term-bin").textContent = (state.terms.bin >= 0 ? "+" : "") + state.terms.bin.toFixed(3);
  document.getElementById("term-chan").textContent = (state.terms.chan >= 0 ? "+" : "") + state.terms.chan.toFixed(3);

  // Nodes & Vector Grid
  for (let z = 0; z < 10; z++) {
    const v = state.s[z];
    const vStr = (v >= 0 ? "+" : "") + v.toFixed(2);
    const color = getNodeColor(v);

    // Vector grid in HUD
    const valEl = document.getElementById(`vec-val-${z}`);
    if (valEl) {
      valEl.textContent = vStr;
      valEl.style.color = color;
    }

    // SVG Node Circle & Halo
    const nodeG = document.querySelector(`.zone-node-group[data-zone="${z}"]`);
    if (nodeG) {
      const halo = nodeG.querySelector(".zone-halo");
      const circle = nodeG.querySelector(".zone-circle");
      const c = COORDS[z];

      if (halo) {
        halo.setAttribute("r", String(c.r + 6 + Math.abs(v) * 20));
        halo.setAttribute("fill", color);
        halo.setAttribute("opacity", String(0.12 + Math.abs(v) * 0.45));
      }
      if (circle) {
        circle.setAttribute("fill", color);
        circle.setAttribute("stroke", Math.abs(v) > 0.5 ? "#ffffff" : "#15ff75");
      }
    }
  }

  // Gates & Open Channels
  let openCount = 0;
  const chanStrip = document.getElementById("channels-strip");
  chanStrip.innerHTML = "";

  for (let z = 1; z <= 9; z++) {
    const isOpen = state.gateOpen[z] > 0.5;
    if (isOpen) openCount++;

    // Map z to gate circle id
    const gtId = ZONES[z].gateVal;
    const circEl = document.getElementById(`gt-circ-${gtId}`);
    if (circEl) {
      if (isOpen) {
        circEl.setAttribute("stroke", "#ff2bd6");
        circEl.setAttribute("fill", "#2b0024");
      } else {
        circEl.setAttribute("stroke", "#15ff75");
        circEl.setAttribute("fill", "#050a06");
      }
    }

    const badge = document.createElement("span");
    badge.className = `channel-badge ${isOpen ? "open" : ""}`;
    const dst = ZONES[z].dst;
    badge.textContent = `${z}➔${dst} (${state.gateOpen[z].toFixed(2)})`;
    chanStrip.appendChild(badge);
  }

  document.getElementById("val-open-count").textContent = `${openCount} / 9`;

  // Syzygy Tensions
  for (const { hi, lo } of SYZYGIES) {
    const el = document.getElementById(`syz-t-${hi}-${lo}`);
    if (el) {
      const tension = Math.abs(state.s[hi] - state.s[lo]);
      el.textContent = `张力: ${tension.toFixed(2)}`;
    }
  }

  // Plumes glowing by tension
  // Current 7 -> 5
  const t75 = Math.abs(state.s[7] - state.s[2]); // tension of 7::2
  const p75 = document.getElementById("plume-7-5");
  if (p75) p75.setAttribute("opacity", String(0.3 + t75 * 0.35));

  // Current 4/5 -> 1
  const t54 = Math.abs(state.s[5] - state.s[4]); // tension of 5::4
  const p41 = document.getElementById("plume-4-1");
  if (p41) p41.setAttribute("opacity", String(0.3 + t54 * 0.35));

  // Current 1 -> 7
  const t81 = Math.abs(state.s[8] - state.s[1]); // tension of 8::1
  const p17 = document.getElementById("plume-1-7");
  if (p17) p17.setAttribute("opacity", String(0.3 + t81 * 0.35));

  // Timeline slider
  const sliderTimeline = document.getElementById("slider-timeline");
  if (sliderTimeline && !sliderTimeline.dataset.dragging) {
    sliderTimeline.value = Math.min(400, state.stepCount);
  }

  // Avalanche text hint
  const annot = document.getElementById("label-avalanche");
  if (annot) {
    if (state.activePreset === "zone9") {
      if (state.stepCount < 30) annot.textContent = "t=0: 仅 9➔9 导通，其余静止";
      else if (state.stepCount < 150) annot.textContent = "t=100: Plex 孤立锁死 (s0=-1, s9=+1)";
      else annot.textContent = "t=200+: 全网点燃，雪崩通透！";
    } else {
      annot.textContent = state.stepCount >= 100 ? "已稳定于吸引子态" : "弛豫收敛中...";
    }
  }
}

// -------------------------------------------------------------
// Presets
// -------------------------------------------------------------
function loadPreset(preset) {
  state.activePreset = preset;
  state.s.fill(0);
  state.history = [];
  state.stepCount = 0;

  switch (preset) {
    case "zone1":
      state.s[1] = 1.0;
      break;
    case "zone9":
      state.s[9] = 1.0;
      break;
    case "zone8":
      state.s[8] = 1.0;
      break;
    case "noise":
      for (let z = 0; z < 10; z++) {
        state.s[z] = (Math.random() * 2 - 1) * 0.05;
      }
      break;
    case "random":
      for (let z = 0; z < 10; z++) {
        state.s[z] = Math.random() * 2 - 1;
      }
      break;
  }

  recordSnapshot();
}

// -------------------------------------------------------------
// Animation Loop
// -------------------------------------------------------------
let lastStepTime = 0;

function mainLoop(timestamp) {
  const interval = 1000 / state.speedFps;
  if (state.running && timestamp - lastStepTime >= interval) {
    lastStepTime = timestamp;
    if (state.stepCount < 500) {
      computeStep();
    } else {
      state.running = false;
      document.getElementById("btn-play").classList.remove("running");
      document.getElementById("btn-play-icon").textContent = "▶";
      document.getElementById("btn-play-text").textContent = "重新演化";
    }
  }

  renderThermalField();
  renderParticles();

  requestAnimationFrame(mainLoop);
}

// -------------------------------------------------------------
// Events
// -------------------------------------------------------------
function initEvents() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      loadPreset(btn.dataset.preset);
    });
  });

  const btnPlay = document.getElementById("btn-play");
  btnPlay.addEventListener("click", () => {
    state.running = !state.running;
    if (state.running) {
      btnPlay.classList.add("running");
      document.getElementById("btn-play-icon").textContent = "⏸";
      document.getElementById("btn-play-text").textContent = "暂停";
    } else {
      btnPlay.classList.remove("running");
      document.getElementById("btn-play-icon").textContent = "▶";
      document.getElementById("btn-play-text").textContent = "弛豫演化";
    }
  });

  document.getElementById("btn-step").addEventListener("click", () => {
    computeStep();
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    loadPreset(state.activePreset);
  });

  const sliderSpeed = document.getElementById("slider-speed");
  sliderSpeed.addEventListener("input", (e) => {
    state.speedFps = parseInt(e.target.value);
    document.getElementById("val-speed").textContent = `${state.speedFps} FPS`;
  });

  document.getElementById("select-palette").addEventListener("change", (e) => {
    state.palette = e.target.value;
    updateUI();
  });

  const sliderTimeline = document.getElementById("slider-timeline");
  sliderTimeline.addEventListener("mousedown", () => {
    sliderTimeline.dataset.dragging = "true";
  });
  sliderTimeline.addEventListener("mouseup", () => {
    delete sliderTimeline.dataset.dragging;
  });
  sliderTimeline.addEventListener("input", (e) => {
    const targetStep = parseInt(e.target.value);
    if (state.history.length > 0) {
      let closest = state.history[0];
      for (const h of state.history) {
        if (Math.abs(h.step - targetStep) < Math.abs(closest.step - targetStep)) {
          closest = h;
        }
      }
      state.stepCount = closest.step;
      state.s.set(closest.s);
      state.gateOpen.set(closest.gateOpen);
      state.energy = closest.E;
      state.terms = { ...closest.terms };
      updateUI();
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      document.getElementById("btn-play").click();
    } else if (e.code === "ArrowRight") {
      document.getElementById("btn-step").click();
    }
  });
}

function initApp() {
  buildSVG();
  buildHUD();
  initParticles();
  initEvents();
  loadPreset("zone1");
  requestAnimationFrame(mainLoop);
}

window.addEventListener("DOMContentLoaded", initApp);
