/**
 * CCRU PANDEMONIUM RITUAL AUDIO SYNTHESIZER
 * Client Application Logic & High-Precision Real-Time Web Audio Engine
 */

(function () {
  'use strict';

  // Global Application State
  const state = {
    manifest: null,
    demons: [],
    nonRouteDemons: [],
    selectedRouteKeys: new Set(), // Set of unique "DemonName:Rt-ID" strings
    routeMap: new Map(), // key -> route data object
    isPlaying: false,
    audioCtx: null,
    masterGainNode: null,
    analyserNode: null,
    visualizerAnimationId: null,
    schedulerTimerId: null,
    activeVoices: new Map(), // key -> voice tracking object
    volume: 0.8,
    timbre: 'buzzer', // Default minimalist piezo buzzer click
    engine: 'webaudio', // Default zero-latency Web Audio API lookahead engine
    audioStreamElement: null,
  };

  // DOM Elements Cache
  const dom = {
    container: document.getElementById('demons-matrix-container'),
    statsPill: document.getElementById('total-stats-pill'),
    btnToggleStream: document.getElementById('btn-toggle-stream'),
    streamIcon: document.getElementById('stream-icon'),
    streamBtnText: document.getElementById('stream-btn-text'),
    statusLamp: document.getElementById('status-lamp'),
    statusLabel: document.getElementById('status-label'),
    statusSub: document.getElementById('status-sub'),
    btnSelectAll: document.getElementById('btn-select-all'),
    btnDeselectAll: document.getElementById('btn-deselect-all'),
    btnExpandAll: document.getElementById('btn-expand-all'),
    btnCollapseAll: document.getElementById('btn-collapse-all'),
    btnExportWav: document.getElementById('btn-export-wav'),
    demonSearch: document.getElementById('demon-search'),
    typeFilter: document.getElementById('type-filter'),
    visualizerCanvas: document.getElementById('audio-visualizer'),
    occultHeader: document.getElementById('occult-toggle-header'),
    occultBody: document.getElementById('occult-body'),
    occultChevron: document.getElementById('occult-chevron'),
    nonRouteContainer: document.getElementById('non-route-container'),
  };

  /* ==========================================================================
     INITIALIZATION & DATA LOADING
     ========================================================================== */

  async function init() {
    setupEventListeners();
    setupVisualizerCanvas();

    try {
      const resp = await fetch('/api/rituals');
      if (!resp.ok) {
        throw new Error(`Failed to load rituals manifest (HTTP ${resp.status})`);
      }
      const data = await resp.json();
      state.manifest = data;
      state.demons = data.demons || [];
      state.nonRouteDemons = data.non_route_demons || [];

      // Build Route Lookup Map
      state.demons.forEach(demon => {
        (demon.routes || []).forEach(rt => {
          state.routeMap.set(rt.route_key, {
            ...rt,
            demonName: demon.name,
            tempo: demon.tempo,
            pitch: demon.pitch,
            net_span: demon.net_span,
            pair: demon.pair,
          });
        });
      });

      // Default initial state: select ALL ritual routes across all demons
      state.demons.forEach(d => {
        (d.routes || []).forEach(r => state.selectedRouteKeys.add(r.route_key));
      });

      renderDemonsMatrix();
      renderNonRouteDemons();
      updateGlobalStatusReadout();

    } catch (err) {
      console.error('Initialization error:', err);
      dom.container.innerHTML = `
        <div class="loading-indicator" style="color: var(--ccru-red);">
          ERROR LOADING PANDEMONIUM MATRIX: ${escapeHtml(err.message)}
        </div>`;
    }
  }

  /* ==========================================================================
     DOM RENDERING
     ========================================================================== */

  function renderDemonsMatrix() {
    const filterText = (dom.demonSearch.value || '').trim().toLowerCase();
    const filterType = dom.typeFilter.value;

    const filtered = state.demons.filter(d => {
      if (filterType !== 'all') {
        const dType = (d.demon_type || '') + ' ' + (d.order || '');
        if (!dType.toLowerCase().includes(filterType.toLowerCase())) {
          return false;
        }
      }
      if (filterText) {
        const hay = [
          d.name,
          d.title,
          d.net_span,
          d.pitch,
          d.door,
          ...(d.routes || []).map(r => `${r.id} ${r.description} ${r.path_str}`),
        ].join(' ').toLowerCase();
        if (!hay.includes(filterText)) {
          return false;
        }
      }
      return true;
    });

    dom.statsPill.textContent = `${filtered.length} / ${state.demons.length} Demons Visible (${state.selectedRouteKeys.size} Routes Selected)`;

    if (filtered.length === 0) {
      dom.container.innerHTML = `
        <div class="loading-indicator">
          NO DEMONS MATCHING SEARCH QUERY "${escapeHtml(filterText)}"
        </div>`;
      return;
    }

    const htmlParts = filtered.map(d => {
      const demonKey = d.name;
      const totalRoutes = d.routes.length;
      let selectedCount = 0;
      d.routes.forEach(r => {
        if (state.selectedRouteKeys.has(r.route_key)) selectedCount++;
      });

      const isChecked = selectedCount === totalRoutes && totalRoutes > 0;
      const pitchClass = d.pitch.startsWith('Ana')
        ? 'badge-pitch-ana'
        : d.pitch.startsWith('Cth')
        ? 'badge-pitch-cth'
        : 'badge-pitch-null';

      const routesHtml = d.routes.map(r => {
        const isRtSelected = state.selectedRouteKeys.has(r.route_key);
        
        // Zone Nodes Chips
        const pathChips = (r.path || []).map((z, idx) => {
          return `<span class="zone-node-chip">Zone ${z}</span>${
            idx < r.path.length - 1 ? '<span class="zone-arrow">&rarr;</span>' : ''
          }`;
        }).join(' ');

        // Calculated Frequency Chips
        const freqChips = (r.frequencies || []).map(fObj => {
          return `<div class="freq-step-chip">
            <span class="zone-idx">Z${fObj.zone}:</span>
            <span class="freq-val">${fObj.freq} Hz</span>
          </div>`;
        }).join('');

        return `
          <div class="rt-item-card ${isRtSelected ? 'is-selected' : ''}" 
               data-route-key="${escapeAttr(r.route_key)}">
            
            <div class="rt-header-row">
              <label class="rt-select-wrap">
                <input type="checkbox" 
                       class="retro-checkbox rt-checkbox" 
                       data-route-key="${escapeAttr(r.route_key)}" 
                       ${isRtSelected ? 'checked' : ''} />
                <span class="rt-id-label">${escapeHtml(r.id)}</span>
              </label>
              <span class="rt-desc">${escapeHtml(r.description || r.meaning || 'Ritual Traverse')}</span>
              <button class="rt-audition-btn" data-audition-key="${escapeAttr(r.route_key)}">
                &#9835; Audition
              </button>
            </div>

            <!-- Path Traversal Sequence -->
            <div class="rt-path-box">
              <span class="path-label">Path:</span>
              ${pathChips}
            </div>

            <!-- Computed Frequencies & Tonality -->
            <div class="rt-freq-grid-box">
              <div class="freq-meta-header">
                <span><strong class="freq-tonality-tag">Tonality:</strong> ${escapeHtml(d.pitch)} [${d.pitch_interval[0]} Hz - ${d.pitch_interval[1]} Hz]</span>
                <span><strong>Voice Tempo:</strong> ${d.tempo}s (${d.tempo_ms}ms)</span>
                <span><strong>Formula:</strong> f(z) = a + (b-a) &times; z/10</span>
              </div>
              <div class="freq-chips-list">
                ${freqChips}
              </div>
            </div>

          </div>
        `;
      }).join('');

      return `
        <div class="demon-card ${selectedCount > 0 ? 'has-selection' : ''}" 
             data-demon-name="${escapeAttr(demonKey)}">
          
          <div class="demon-header" data-toggle-demon="${escapeAttr(demonKey)}">
            <label class="demon-select-wrap" onclick="event.stopPropagation()">
              <input type="checkbox" 
                     class="retro-checkbox demon-master-checkbox" 
                     data-demon-name="${escapeAttr(demonKey)}"
                     ${isChecked ? 'checked' : ''} />
              <span class="demon-id-badge">${escapeHtml(d.id || `Mesh-${String(d.index).padStart(2, '0')}`)}</span>
            </label>

            <div class="demon-name-block">
              <span class="demon-name">${escapeHtml(d.name)}</span>
              <span class="demon-title">${escapeHtml(d.title || d.demon_type)}</span>
            </div>

            <div class="demon-badges-group">
              <span class="badge badge-netspan" title="Net-Span Pair">${escapeHtml(d.net_span)}</span>
              <span class="badge ${pitchClass}" title="Pitch Level">${escapeHtml(d.pitch)}</span>
              <span class="badge badge-tempo" title="Note Tempo Duration">${d.tempo}s (${d.tempo_ms}ms)</span>
              ${d.door ? `<span class="badge badge-door" title="Door">${escapeHtml(d.door)}</span>` : ''}
              <span class="demon-active-counter">
                [${selectedCount}/${totalRoutes} Rt]
              </span>
              <button class="demon-toggle-btn" type="button" data-toggle-action="toggle">EXPAND</button>
              <span class="chevron-icon">&#9660;</span>
            </div>
          </div>

          <div class="demon-body">
            ${routesHtml}
          </div>

        </div>
      `;
    });

    dom.container.innerHTML = htmlParts.join('');
    updateDomCheckboxStates();
  }

  function renderNonRouteDemons() {
    if (!state.nonRouteDemons || state.nonRouteDemons.length === 0) return;
    const cards = state.nonRouteDemons.map(d => {
      const routesStr = (d.raw_routes || []).map(r => `${r.id}:[${r.path}] ${r.description}`).join('; ');
      return `
        <div class="non-route-card">
          <div class="non-route-card-title">${escapeHtml(d.name)} (${escapeHtml(d.net_span)}) - ${escapeHtml(d.pitch)}</div>
          <div class="non-route-card-route">${escapeHtml(routesStr || 'Chaotic / Tangential')}</div>
        </div>
      `;
    }).join('');
    dom.nonRouteContainer.innerHTML = cards;
  }

  /* ==========================================================================
     UNIFIED CHECKBOX SYNCHRONIZATION
     ========================================================================== */

  /**
   * Scans all route checkboxes and demon master checkboxes in the DOM and
   * synchronizes their checked and indeterminate states strictly with state.selectedRouteKeys.
   */
  function updateDomCheckboxStates() {
    // 1. Sync every individual Rt checkbox
    const allRtCheckboxes = dom.container.querySelectorAll('.rt-checkbox');
    allRtCheckboxes.forEach(cb => {
      const key = cb.getAttribute('data-route-key');
      const isSelected = state.selectedRouteKeys.has(key);
      cb.checked = isSelected;
      const card = cb.closest('.rt-item-card');
      if (card) {
        card.classList.toggle('is-selected', isSelected);
      }
    });

    // 2. Sync every demon master card and checkbox
    const allDemonCards = dom.container.querySelectorAll('.demon-card');
    allDemonCards.forEach(dCard => {
      const demonName = dCard.getAttribute('data-demon-name');
      const demon = state.demons.find(d => d.name === demonName);
      if (!demon) return;

      const total = demon.routes.length;
      let count = 0;
      demon.routes.forEach(r => {
        if (state.selectedRouteKeys.has(r.route_key)) {
          count++;
        }
      });

      const masterCb = dCard.querySelector('.demon-master-checkbox');
      if (masterCb) {
        if (count === total && total > 0) {
          masterCb.checked = true;
          masterCb.indeterminate = false;
        } else if (count > 0 && count < total) {
          masterCb.checked = false;
          masterCb.indeterminate = true;
        } else {
          masterCb.checked = false;
          masterCb.indeterminate = false;
        }
      }

      dCard.classList.toggle('has-selection', count > 0);
      const counterEl = dCard.querySelector('.demon-active-counter');
      if (counterEl) {
        counterEl.textContent = `[${count}/${total} Rt]`;
      }
    });
  }

  function updateGlobalStatusReadout() {
    const totalSelected = state.selectedRouteKeys.size;
    const activeDemons = new Set();
    state.selectedRouteKeys.forEach(k => {
      const info = state.routeMap.get(k);
      if (info) activeDemons.add(info.demonName);
    });

    if (state.isPlaying) {
      dom.statusLamp.classList.add('active');
      dom.statusLabel.textContent = `STREAMING LIVE // ${totalSelected} POLYSYNTH VOICES ACTIVE`;
      dom.statusSub.textContent = `${activeDemons.size} Demons Active &bull; Minimalist Buzzer Click Engine (30ms Piezo Pulse)`;
    } else {
      dom.statusLamp.classList.remove('active');
      dom.statusLabel.textContent = `STANDBY // ${totalSelected} VOICES ARMED`;
      dom.statusSub.textContent = `${activeDemons.size} Demons Selected &bull; Click START RITUAL STREAM to play`;
    }

    if (dom.statsPill) {
      dom.statsPill.textContent = `${state.demons.length} Demons Total (${totalSelected} Routes Selected)`;
    }
  }

  /* ==========================================================================
     EVENT LISTENERS & USER INTERACTIONS
     ========================================================================== */

  function setupEventListeners() {
    // Play/Stop Toggle
    dom.btnToggleStream.addEventListener('click', toggleAudioStream);

    // Global Selection: Select All Rituals
    dom.btnSelectAll.addEventListener('click', () => {
      state.demons.forEach(d => {
        d.routes.forEach(r => state.selectedRouteKeys.add(r.route_key));
      });
      updateDomCheckboxStates();
      updateVoicesForActivePlayback();
      updateGlobalStatusReadout();
    });

    // Global Selection: Deselect All Rituals
    dom.btnDeselectAll.addEventListener('click', () => {
      state.selectedRouteKeys.clear();
      updateDomCheckboxStates();
      updateVoicesForActivePlayback();
      updateGlobalStatusReadout();
    });

    // Expand / Collapse All
    dom.btnExpandAll.addEventListener('click', () => {
      document.querySelectorAll('.demon-card').forEach(card => {
        card.classList.add('expanded');
        const btn = card.querySelector('.demon-toggle-btn');
        if (btn) btn.textContent = 'COLLAPSE';
      });
    });

    dom.btnCollapseAll.addEventListener('click', () => {
      document.querySelectorAll('.demon-card').forEach(card => {
        card.classList.remove('expanded');
        const btn = card.querySelector('.demon-toggle-btn');
        if (btn) btn.textContent = 'EXPAND';
      });
    });

    // Export WAV (30s)
    dom.btnExportWav.addEventListener('click', exportWavAudio);

    // Search and Filters
    dom.demonSearch.addEventListener('input', debounce(renderDemonsMatrix, 150));
    dom.typeFilter.addEventListener('change', renderDemonsMatrix);

    // Occult Reference Accordion
    dom.occultHeader.addEventListener('click', () => {
      const isHidden = dom.occultBody.style.display === 'none';
      dom.occultBody.style.display = isHidden ? 'block' : 'none';
      dom.occultChevron.innerHTML = isHidden ? '&#9650;' : '&#9660;';
    });

    // Delegated click handling on Demons Container
    dom.container.addEventListener('click', handleContainerClick);
    dom.container.addEventListener('change', handleContainerChange);
  }

  function handleContainerClick(e) {
    // 1. Accordion Header Toggle
    const headerEl = e.target.closest('[data-toggle-demon]');
    if (headerEl && !e.target.closest('.demon-select-wrap')) {
      const card = headerEl.closest('.demon-card');
      if (card) {
        const isNowExpanded = card.classList.toggle('expanded');
        const btn = card.querySelector('.demon-toggle-btn');
        if (btn) btn.textContent = isNowExpanded ? 'COLLAPSE' : 'EXPAND';
      }
      return;
    }

    // 2. Single Rt Audition Button
    const auditionBtn = e.target.closest('[data-audition-key]');
    if (auditionBtn) {
      e.stopPropagation();
      const rKey = auditionBtn.getAttribute('data-audition-key');
      auditionSingleRoute(rKey);
      return;
    }
  }

  function handleContainerChange(e) {
    // 1. Master Demon Checkbox
    if (e.target.classList.contains('demon-master-checkbox')) {
      const demonName = e.target.getAttribute('data-demon-name');
      const isChecked = e.target.checked;
      const demon = state.demons.find(d => d.name === demonName);
      if (demon) {
        demon.routes.forEach(r => {
          if (isChecked) {
            state.selectedRouteKeys.add(r.route_key);
          } else {
            state.selectedRouteKeys.delete(r.route_key);
          }
        });
      }
      updateDomCheckboxStates();
      updateVoicesForActivePlayback();
      updateGlobalStatusReadout();
      return;
    }

    // 2. Individual Rt Checkbox
    if (e.target.classList.contains('rt-checkbox')) {
      const rKey = e.target.getAttribute('data-route-key');
      const isChecked = e.target.checked;
      if (isChecked) {
        state.selectedRouteKeys.add(rKey);
      } else {
        state.selectedRouteKeys.delete(rKey);
      }
      updateDomCheckboxStates();
      updateVoicesForActivePlayback();
      updateGlobalStatusReadout();
      return;
    }
  }

  /* ==========================================================================
     REAL-TIME WEB AUDIO ENGINE (ZERO-DRIFT LOOKAHEAD SCHEDULER)
     ========================================================================== */

  function getOrCreateAudioContext() {
    if (!state.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AudioContextClass();

      state.masterGainNode = state.audioCtx.createGain();
      state.masterGainNode.gain.setValueAtTime(state.volume, state.audioCtx.currentTime);

      state.analyserNode = state.audioCtx.createAnalyser();
      state.analyserNode.fftSize = 256;
      state.analyserNode.smoothingTimeConstant = 0.8;

      state.masterGainNode.connect(state.analyserNode);
      state.analyserNode.connect(state.audioCtx.destination);
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
    return state.audioCtx;
  }

  function toggleAudioStream() {
    if (state.isPlaying) {
      stopAudioStream();
    } else {
      startAudioStream();
    }
  }

  function startAudioStream() {
    if (state.selectedRouteKeys.size === 0) {
      alert('Please select at least one ritual route (Rt) to play.');
      return;
    }

    state.isPlaying = true;
    dom.btnToggleStream.classList.remove('play');
    dom.btnToggleStream.classList.add('stop');
    dom.streamIcon.textContent = '⏹';
    dom.streamBtnText.textContent = 'STOP RITUAL STREAM';

    startWebAudioSequencer();
    startVisualizerLoop();
    updateGlobalStatusReadout();
  }

  function stopAudioStream() {
    state.isPlaying = false;
    dom.btnToggleStream.classList.remove('stop');
    dom.btnToggleStream.classList.add('play');
    dom.streamIcon.textContent = '▶';
    dom.streamBtnText.textContent = 'START RITUAL STREAM';

    if (state.schedulerTimerId) {
      clearInterval(state.schedulerTimerId);
      state.schedulerTimerId = null;
    }

    state.activeVoices.clear();
    stopVisualizerLoop();
    updateGlobalStatusReadout();
  }

  function startWebAudioSequencer() {
    const ctx = getOrCreateAudioContext();
    const now = ctx.currentTime;

    state.activeVoices.clear();
    let idx = 0;
    state.selectedRouteKeys.forEach(rKey => {
      const rInfo = state.routeMap.get(rKey);
      if (rInfo && rInfo.raw_frequencies && rInfo.raw_frequencies.length > 0) {
        state.activeVoices.set(rKey, {
          routeKey: rKey,
          demonName: rInfo.demonName,
          tempo: Math.max(0.05, rInfo.tempo || 0.5),
          freqs: rInfo.raw_frequencies,
          noteIdx: 0,
          nextNoteTime: now + 0.05,
          pan: -0.8 + 1.6 * (idx % 9) / 8.0,
        });
        idx++;
      }
    });

    const LOOKAHEAD_MS = 25;
    const SCHEDULE_AHEAD_SEC = 0.12;

    state.schedulerTimerId = setInterval(() => {
      if (!state.isPlaying) return;
      const curTime = ctx.currentTime;
      const targetTime = curTime + SCHEDULE_AHEAD_SEC;

      state.activeVoices.forEach(voice => {
        while (voice.nextNoteTime < targetTime) {
          if (voice.nextNoteTime >= curTime - 0.02) {
            scheduleBuzzerNote(voice, voice.nextNoteTime);
          }
          voice.noteIdx = (voice.noteIdx + 1) % voice.freqs.length;
          voice.nextNoteTime += voice.tempo;
        }
      });
    }, LOOKAHEAD_MS);
  }

  function updateVoicesForActivePlayback() {
    if (!state.isPlaying) return;
    const ctx = getOrCreateAudioContext();
    const now = ctx.currentTime;

    // Remove deselected
    for (const [key] of state.activeVoices) {
      if (!state.selectedRouteKeys.has(key)) {
        state.activeVoices.delete(key);
      }
    }

    // Add newly selected
    let idx = state.activeVoices.size;
    state.selectedRouteKeys.forEach(rKey => {
      if (!state.activeVoices.has(rKey)) {
        const rInfo = state.routeMap.get(rKey);
        if (rInfo && rInfo.raw_frequencies && rInfo.raw_frequencies.length > 0) {
          state.activeVoices.set(rKey, {
            routeKey: rKey,
            demonName: rInfo.demonName,
            tempo: Math.max(0.05, rInfo.tempo || 0.5),
            freqs: rInfo.raw_frequencies,
            noteIdx: 0,
            nextNoteTime: now + 0.02,
            pan: -0.8 + 1.6 * (idx % 9) / 8.0,
          });
          idx++;
        }
      }
    });
  }

  function scheduleBuzzerNote(voice, noteTime) {
    const ctx = state.audioCtx;
    if (!ctx) return;

    const freq = voice.freqs[voice.noteIdx % voice.freqs.length];
    const clickDuration = Math.min(0.03, voice.tempo * 0.45);
    const fadeDuration = 0.002;

    // Gain Envelope Node
    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0.0001, noteTime);
    gainNode.gain.linearRampToValueAtTime(0.25, noteTime + fadeDuration);
    gainNode.gain.setValueAtTime(0.25, noteTime + clickDuration - fadeDuration);
    gainNode.gain.linearRampToValueAtTime(0.0001, noteTime + clickDuration);

    // Stereo Panner Node
    if (ctx.createStereoPanner) {
      const pannerNode = ctx.createStereoPanner();
      pannerNode.pan.setValueAtTime(voice.pan, noteTime);
      gainNode.connect(pannerNode);
      pannerNode.connect(state.masterGainNode);
    } else {
      gainNode.connect(state.masterGainNode);
    }

    // CCRU Minimalist Buzzer Click: Fundamental Sine + 0.25 * 3rd Harmonic
    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(freq, noteTime);

    const osc3 = ctx.createOscillator();
    osc3.type = 'sine';
    osc3.frequency.setValueAtTime(freq * 3.0, noteTime);

    const osc3Gain = ctx.createGain();
    osc3Gain.gain.setValueAtTime(0.25, noteTime);

    osc1.connect(gainNode);
    osc3.connect(osc3Gain);
    osc3Gain.connect(gainNode);

    osc1.start(noteTime);
    osc3.start(noteTime);
    osc1.stop(noteTime + clickDuration + 0.005);
    osc3.stop(noteTime + clickDuration + 0.005);

    // Schedule synchronized UI flash
    const delayMs = Math.max(0, (noteTime - ctx.currentTime) * 1000);
    setTimeout(() => {
      triggerVisualFlash(voice.routeKey, voice.demonName);
    }, delayMs);
  }

  function triggerVisualFlash(routeKey, demonName) {
    const rtCard = dom.container.querySelector(`.rt-item-card[data-route-key="${CSS.escape(routeKey)}"]`);
    if (rtCard) {
      rtCard.classList.add('is-active-click');
      setTimeout(() => rtCard.classList.remove('is-active-click'), 60);
    }
    const demonCard = dom.container.querySelector(`.demon-card[data-demon-name="${CSS.escape(demonName)}"]`);
    if (demonCard) {
      demonCard.classList.add('is-clicking');
      setTimeout(() => demonCard.classList.remove('is-clicking'), 60);
    }
  }

  function auditionSingleRoute(rKey) {
    const rInfo = state.routeMap.get(rKey);
    if (!rInfo || !rInfo.raw_frequencies) return;

    const ctx = getOrCreateAudioContext();
    const now = ctx.currentTime + 0.02;
    const tempo = Math.max(0.05, rInfo.tempo || 0.5);

    // Play 3 full cycles of this single ritual
    const cycles = 3;
    const totalNotes = rInfo.raw_frequencies.length * cycles;
    for (let i = 0; i < totalNotes; i++) {
      const noteTime = now + i * tempo;
      const v = {
        routeKey: rKey,
        demonName: rInfo.demonName,
        tempo: tempo,
        freqs: rInfo.raw_frequencies,
        noteIdx: i % rInfo.raw_frequencies.length,
        pan: 0.0,
      };
      scheduleBuzzerNote(v, noteTime);
    }
  }

  async function exportWavAudio() {
    const selectedList = Array.from(state.selectedRouteKeys);
    if (selectedList.length === 0) {
      alert('Please select at least one ritual route to export.');
      return;
    }

    dom.btnExportWav.textContent = '[ EXPORTING... ]';
    try {
      const resp = await fetch('/api/export_audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rts: selectedList,
          duration: 30.0,
        }),
      });
      if (!resp.ok) throw new Error(`Export failed with HTTP ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pandemonium_rituals_${selectedList.length}_routes_30s.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export error: ${err.message}`);
    } finally {
      dom.btnExportWav.textContent = '[ EXPORT WAV (30s) ]';
    }
  }

  /* ==========================================================================
     OSCILLOSCOPE / AUDIO VISUALIZER
     ========================================================================== */

  function setupVisualizerCanvas() {
    const canvas = dom.visualizerCanvas;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0b0d03';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#383a15';
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }

  function startVisualizerLoop() {
    const canvas = dom.visualizerCanvas;
    const ctx = canvas.getContext('2d');
    const analyser = state.analyserNode;
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      state.visualizerAnimationId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'rgba(11, 13, 3, 0.3)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#68d338';
      ctx.beginPath();

      const sliceWidth = canvas.width / Number(bufferLength);
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2.0;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }

    draw();
  }

  function stopVisualizerLoop() {
    if (state.visualizerAnimationId) {
      cancelAnimationFrame(state.visualizerAnimationId);
      state.visualizerAnimationId = null;
    }
    setupVisualizerCanvas();
  }

  /* ==========================================================================
     UTILITY HELPERS
     ========================================================================== */

  function debounce(fn, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escapeAttr(str) {
    return escapeHtml(str);
  }

  // Start initialization on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
