(function() {
"use strict";

// ============================================================================
// MoE (Mixture of Experts) Performance Visualization Widget
// ============================================================================
// Interactive visualization for understanding MoE communication patterns,
// Double Buffer Overlap (DBO) pipelining, and DeepEP optimizations.
// ============================================================================

// ============================================================================
// SECTION 1: UTILITY FUNCTIONS
// ============================================================================

/**
 * Format a number with appropriate precision based on magnitude.
 * @param {number} value - The number to format
 * @param {number} decimalPlaces - Maximum decimal places (default: 3)
 * @returns {string} Formatted number string
 */
function formatNumber(value, decimalPlaces = 3) {
  if (!isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(decimalPlaces);
}

/**
 * Shorthand for document.getElementById with null safety.
 * @param {string} elementId - DOM element ID
 * @returns {HTMLElement|null}
 */
function getElement(elementId) {
  return document.getElementById(elementId);
}

// ============================================================================
// SECTION 2: MODEL CONSTANTS & CONFIGURATION
// ============================================================================

/** MoE architecture constants */
const MOE_CONFIG = {
  HIDDEN_DIMENSION: 6144,        // Hidden layer size (typical for large models)
  BYTES_PER_ELEMENT: 2,          // FP16/BF16 = 2 bytes
  TOP_K_EXPERTS: 2,              // Number of experts selected per token
};

/** Default slider/control values */
const DEFAULT_PARAMETERS = {
  tokensExponent: 10,    // 2^10 * 8 = 8192 tokens
  expertParallelism: 16, // EP ranks
  latencyMicroseconds: 25,
  bandwidthGBps: 25,
  computeMicrosecondsPerToken: 0.45,
  loadImbalanceSkew: 0.10,
  enableDBO: true,
  enableMicroBatchSplit: true,
  enableAnimation: true,
};

// ============================================================================
// SECTION 3: PERFORMANCE MODEL CALCULATIONS
// ============================================================================

/**
 * Calculate the straggler factor based on EP ranks and load imbalance.
 * Models the worst-case expert load due to routing imbalance.
 * 
 * @param {number} epRanks - Number of expert parallel ranks
 * @param {number} skew - Load imbalance factor (0 = perfect balance, 1 = severe)
 * @returns {number} Multiplicative straggler factor
 */
function calculateStragglerFactor(epRanks, skew) {
  const structuralImbalance = 1 + 2.2 * skew;
  const coordinationPenalty = 1 + 0.12 * Math.log2(epRanks);
  return structuralImbalance * coordinationPenalty;
}

/**
 * Calculate bytes transferred per token per dispatch/combine operation.
 * @returns {number} Bytes per token
 */
function calculateBytesPerToken() {
  return MOE_CONFIG.TOP_K_EXPERTS * MOE_CONFIG.HIDDEN_DIMENSION * MOE_CONFIG.BYTES_PER_ELEMENT;
}

/**
 * Calculate dispatch (all-to-all communication) time in milliseconds.
 * 
 * @param {number} tokenCount - Number of tokens in batch
 * @param {number} latencyMicroseconds - Network latency in µs
 * @param {number} bandwidthGBps - Effective bandwidth in GB/s
 * @returns {number} Dispatch time in milliseconds
 */
function calculateDispatchTimeMs(tokenCount, epRanks, latencyMicroseconds, bandwidthGBps) {
  const latencyMs = latencyMicroseconds / 1000.0;
  const bandwidthBytesPerSec = bandwidthGBps * 1e9;
  const totalBytes = (tokenCount * calculateBytesPerToken()) / epRanks;
  const transferTimeMs = (totalBytes / bandwidthBytesPerSec) * 1000.0;
  return latencyMs + transferTimeMs;
}

/**
 * Calculate expert compute time in milliseconds.
 * Models the time for the slowest (straggler) expert to complete.
 * 
 * @param {number} tokenCount - Number of tokens in batch
 * @param {number} epRanks - Number of expert parallel ranks
 * @param {number} computeUsPerToken - Compute time per token in µs
 * @param {number} skew - Load imbalance factor
 * @returns {number} Compute time in milliseconds
 */
function calculateComputeTimeMs(tokenCount, epRanks, computeUsPerToken, skew) {
  const routedTokens = tokenCount * MOE_CONFIG.TOP_K_EXPERTS;
  const meanTokensPerExpert = routedTokens / epRanks;
  const randomVariationFactor = 1 + 3.2 / Math.sqrt(Math.max(1, tokenCount));
  const stragglerFactor = calculateStragglerFactor(epRanks, skew);
  const maxExpertLoad = meanTokensPerExpert * randomVariationFactor * stragglerFactor;
  return (maxExpertLoad * computeUsPerToken) / 1000.0;
}

/**
 * Run the full MoE performance model.
 * 
 * @param {number} tokenCount - Tokens per step
 * @param {number} epRanks - Expert parallelism degree
 * @param {number} latencyUs - Network latency (µs)
 * @param {number} bandwidthGBps - Bandwidth (GB/s)
 * @param {number} computeUsPerToken - Compute time per token (µs)
 * @param {number} skew - Load imbalance (0-1)
 * @returns {Object} Timing breakdown
 */
function calculateMoETimings(tokenCount, epRanks, latencyUs, bandwidthGBps, computeUsPerToken, skew) {
  const dispatchMs = calculateDispatchTimeMs(tokenCount, epRanks, latencyUs, bandwidthGBps);
  const computeMs = calculateComputeTimeMs(tokenCount, epRanks, computeUsPerToken, skew);
  const combineMs = dispatchMs; // Symmetric all-to-all

  const sequentialTotalMs = dispatchMs + computeMs + combineMs;
  const dboOverlappedMs = Math.max(dispatchMs + combineMs, computeMs);

  return {
    dispatchMs,
    computeMs,
    combineMs,
    sequentialTotalMs,
    dboOverlappedMs,
  };
}

// ============================================================================
// SECTION 4: DEEPEP MODEL (Low-Latency vs High-Throughput)
// ============================================================================

const DeepEPModel = {
  /**
   * Get DeepEP parameters for a specific mode.
   * @param {'ll'|'ht'} mode - 'll' for low-latency, 'ht' for high-throughput
   * @param {number} baseLatencyUs - Base latency from slider
   * @param {number} baseBandwidthGBps - Base bandwidth from slider
   * @returns {Object} Mode parameters
   */
  getParameters(mode, baseLatencyUs, baseBandwidthGBps) {
    const baseLatencyMs = baseLatencyUs / 1000.0;

    if (mode === 'll') {
      return {
        name: 'Low‑latency',
        latencyMs: 0.55 * baseLatencyMs,
        bandwidthGBps: 0.70 * baseBandwidthGBps,
        description: 'smaller fixed cost; less BW‑optimized',
      };
    }
    return {
      name: 'High‑throughput',
      latencyMs: 1.15 * baseLatencyMs,
      bandwidthGBps: 1.35 * baseBandwidthGBps,
      description: 'more setup; higher BW_eff via hierarchy',
    };
  },

  /**
   * Calculate communication time for DeepEP mode.
   * @param {number} tokenCount - Number of tokens
   * @param {'ll'|'ht'} mode - DeepEP mode
   * @param {number} baseLatencyUs - Base latency
   * @param {number} baseBandwidthGBps - Base bandwidth
   * @returns {number} Round-trip communication time in ms
   */
  calculateCommTimeMs(tokenCount, epRanks, mode, baseLatencyUs, baseBandwidthGBps) {
    const totalBytes = (tokenCount * calculateBytesPerToken()) / epRanks;
    const params = this.getParameters(mode, baseLatencyUs, baseBandwidthGBps);
    const bandwidthBytesPerSec = params.bandwidthGBps * 1e9;
    const transferTimeMs = (totalBytes / bandwidthBytesPerSec) * 1000.0;
    return 2 * (params.latencyMs + transferTimeMs); // Round-trip
  },

  /**
   * Automatically pick the best mode based on batch size.
   * @param {number} tokenCount - Number of tokens
   * @param {number} baseLatencyUs - Base latency
   * @param {number} baseBandwidthGBps - Base bandwidth
   * @returns {'ll'|'ht'} Optimal mode
   */
  pickOptimalMode(tokenCount, epRanks, baseLatencyUs, baseBandwidthGBps) {
    const llTime = this.calculateCommTimeMs(tokenCount, epRanks, 'll', baseLatencyUs, baseBandwidthGBps);
    const htTime = this.calculateCommTimeMs(tokenCount, epRanks, 'ht', baseLatencyUs, baseBandwidthGBps);
    return llTime <= htTime ? 'll' : 'ht';
  },
};

// ============================================================================
// SECTION 5: THEME SYSTEM
// ============================================================================

const ThemeManager = {
  colors: {},

  /** Read CSS custom properties and populate color palette */
  loadFromCSS() {
    const css = getComputedStyle(document.documentElement);
    const get = (name) => css.getPropertyValue(name).trim() || null;

    this.colors = {
      foreground: get('--fg') || '#1a1a1a',
      muted: get('--muted') || '#555555',
      mutedSecondary: get('--muted-2') || '#888888',
      border: get('--border') || '#e5e2dc',
      surface: get('--surface') || '#ffffff',
      surfaceSecondary: get('--surface-2') || '#f5f4f1',
      compute: get('--compute') || '#059669',
      communication: get('--comm') || '#2563eb',
      combine: get('--combine') || '#dc2626',
      dboHighlight: get('--dbo') || '#7c3aed',
      sequentialHighlight: get('--no-overlap') || '#d97706',
      batchPrimary: '#94a3b8',
      batchSecondary: '#64748b',
      fontBody: get('--font-body') || '-apple-system, BlinkMacSystemFont, sans-serif',
      fontMono: get('--font-mono') || 'SF Mono, Monaco, monospace',
    };

    return this.colors;
  },

  /** Get surface color with alpha */
  getSurfaceWithAlpha(alpha = 0.95) {
    return this._colorWithAlpha(this.colors.surface, alpha) || `rgba(255, 255, 255, ${alpha})`;
  },

  /** Get subtle overlay color */
  getOverlayColor(alpha = 0.02) {
    return this._colorWithAlpha(this.colors.foreground, alpha) || `rgba(0, 0, 0, ${alpha})`;
  },

  _colorWithAlpha(color, alpha) {
    try {
      const c = d3.color(color);
      if (!c) return null;
      return `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
    } catch {
      return null;
    }
  },
};

// Initialize theme
ThemeManager.loadFromCSS();

// Listen for theme changes
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    ThemeManager.loadFromCSS();
    renderAll();
  });
}

// ============================================================================
// SECTION 6: ANIMATION STATE MANAGEMENT
// ============================================================================

const AnimationController = {
  lastFrameTime: performance.now(),
  elapsedSeconds: 0,
  requestId: null,
  equationsDirty: true,

  reset() {
    this.lastFrameTime = performance.now();
    this.elapsedSeconds = 0;
    this.equationsDirty = true;
  },

  start(tickCallback) {
    if (this.requestId) return;
    this.lastFrameTime = performance.now();
    const loop = (now) => {
      tickCallback(now);
      this.requestId = requestAnimationFrame(loop);
    };
    this.requestId = requestAnimationFrame(loop);
  },

  stop() {
    if (this.requestId) {
      cancelAnimationFrame(this.requestId);
      this.requestId = null;
    }
  },

  tick(now) {
    const deltaSeconds = (now - this.lastFrameTime) / 1000;
    this.lastFrameTime = now;
    this.elapsedSeconds += deltaSeconds;
    return this.elapsedSeconds;
  },
};

// ============================================================================
// SECTION 7: DOM ELEMENT REFERENCES
// ============================================================================

const DOMElements = {
  // Sliders
  tokensSlider: null,
  epSlider: null,
  latencySlider: null,
  bandwidthSlider: null,
  computeSlider: null,
  skewSlider: null,

  // Secondary Sliders (DeepEP)
  tokensSlider2: null,
  epSlider2: null,
  latencySlider2: null,
  bandwidthSlider2: null,
  computeSlider2: null,
  skewSlider2: null,

  // Checkboxes
  dboCheckbox: null,
  splitCheckbox: null,
  animateCheckbox: null,

  // Buttons
  resetButton: null,
  decodePresetButton: null,
  prefillPresetButton: null,

  // SVG containers
  timelineSvg: null,
  curvesSvg: null,

  // DeepEP elements
  deepepCurveSvg: null,
  deepepTopoSvg: null,
  deepepTimelineSvg: null,
  deepepAutoRadio: null,
  deepepLLRadio: null,
  deepepHTRadio: null,
  deepepAnimateCheckbox: null,

  // Tooltip
  timelineTooltip: null,
  timelineTooltipContainer: null,

  /** Initialize all DOM references */
  init() {
    this.tokensSlider = getElement('tokens');
    this.epSlider = getElement('ep');
    this.latencySlider = getElement('lat');
    this.bandwidthSlider = getElement('bw');
    this.computeSlider = getElement('ctok');
    this.skewSlider = getElement('skew');

    this.tokensSlider2 = getElement('tokens-deepep');
    this.epSlider2 = getElement('ep-deepep');
    this.latencySlider2 = getElement('lat-deepep');
    this.bandwidthSlider2 = getElement('bw-deepep');
    this.computeSlider2 = getElement('ctok-deepep');
    this.skewSlider2 = getElement('skew-deepep');

    this.dboCheckbox = getElement('dbo');
    this.splitCheckbox = getElement('split');
    this.animateCheckbox = getElement('animate');

    this.resetButton = getElement('reset');
    this.decodePresetButton = getElement('snapDecode');
    this.prefillPresetButton = getElement('snapPrefill');

    this.timelineSvg = getElement('timeline');
    this.curvesSvg = getElement('curves');

    this.deepepCurveSvg = getElement('deepepCurve');
    this.deepepTopoSvg = getElement('deepepTopo');
    this.deepepTimelineSvg = getElement('deepepTimeline');
    this.deepepAutoRadio = getElement('deepepAuto');
    this.deepepLLRadio = getElement('deepepLL');
    this.deepepHTRadio = getElement('deepepHT');

    this.timelineTooltip = getElement('timelineTip');
    this.timelineTooltipContainer = this.timelineTooltip?.parentElement || null;
  },

  getDeepepAnimateCheckbox() {
    if (!this.deepepAnimateCheckbox) {
      // Try finding the one in DeepEPMechanics first
      this.deepepAnimateCheckbox = getElement('mechAnim');
      
      if (!this.deepepAnimateCheckbox) {
          this.deepepAnimateCheckbox = getElement('dpAnimToggle');
      }
      
      // Fallback if ID lookup fails: find container via radio button
      if (!this.deepepAnimateCheckbox && this.deepepHTRadio) {
        const container = this.deepepHTRadio.closest('.viz-toggles');
        if (container) {
          // Try finding existing checkbox first
          this.deepepAnimateCheckbox = container.querySelector('input[type="checkbox"]');
          
          // If absolutely missing, create it programmatically
          if (!this.deepepAnimateCheckbox) {
             const label = document.createElement('label');
             label.className = 'viz-toggle';
             const input = document.createElement('input');
             input.type = 'checkbox';
             input.id = 'dpAnimToggle';
             input.checked = true;
             label.appendChild(input);
             label.appendChild(document.createTextNode(' Animate'));
             container.appendChild(label);
             this.deepepAnimateCheckbox = input;
             console.log('Created DeepEP checkbox programmatically');
          } else {
             console.log('Found DeepEP checkbox via fallback traversal');
          }
        }
      }
    }
    return this.deepepAnimateCheckbox;
  },
};

// ============================================================================
// SECTION 8: CONTROL STATE MANAGEMENT
// ============================================================================

const ControlState = {
  /** Apply a state object to all controls */
  applyToControls(state) {
    const el = DOMElements;
    if (!el.tokensSlider) return;

    el.tokensSlider.value = state.tokensExponent;
    el.epSlider.value = state.expertParallelism;
    el.latencySlider.value = state.latencyMicroseconds;
    el.bandwidthSlider.value = state.bandwidthGBps;
    el.computeSlider.value = state.computeMicrosecondsPerToken;
    el.skewSlider.value = state.loadImbalanceSkew;

    if (el.tokensSlider2) el.tokensSlider2.value = state.tokensExponent;
    if (el.epSlider2) el.epSlider2.value = state.expertParallelism;
    if (el.latencySlider2) el.latencySlider2.value = state.latencyMicroseconds;
    if (el.bandwidthSlider2) el.bandwidthSlider2.value = state.bandwidthGBps;
    if (el.computeSlider2) el.computeSlider2.value = state.computeMicrosecondsPerToken;
    if (el.skewSlider2) el.skewSlider2.value = state.loadImbalanceSkew;

    el.dboCheckbox.checked = state.enableDBO;
    el.splitCheckbox.checked = state.enableMicroBatchSplit;
    el.animateCheckbox.checked = state.enableAnimation;

    AnimationController.reset();
    onStateChange();
  },

  /** Read current token count from slider (exponential scale) */
  getTokenCount() {
    if (!DOMElements.tokensSlider) return 256;
    const exponent = parseInt(DOMElements.tokensSlider.value, 10);
    return Math.round(Math.pow(2, exponent) * 8);
  },

  /** Read all model parameters from controls */
  getModelParameters() {
    const el = DOMElements;
    if (!el.epSlider) {
      return {
        tokenCount: 256,
        epRanks: 16,
        latencyUs: 25,
        bandwidthGBps: 25,
        computeUsPerToken: 0.45,
        skew: 0.10,
      };
    }

    const params = {
      tokenCount: this.getTokenCount(),
      epRanks: parseInt(el.epSlider.value, 10),
      latencyUs: parseFloat(el.latencySlider.value),
      bandwidthGBps: parseFloat(el.bandwidthSlider.value),
      computeUsPerToken: parseFloat(el.computeSlider.value),
      skew: parseFloat(el.skewSlider.value),
    };

    // Sanitize NaN values
    if (isNaN(params.tokenCount)) params.tokenCount = 256;
    if (isNaN(params.epRanks)) params.epRanks = 16;
    if (isNaN(params.latencyUs)) params.latencyUs = 25;
    if (isNaN(params.bandwidthGBps)) params.bandwidthGBps = 25;
    if (isNaN(params.computeUsPerToken)) params.computeUsPerToken = 0.45;
    if (isNaN(params.skew)) params.skew = 0.10;

    return params;
  },

  /** Get current DeepEP mode selection */
  getDeepEPMode() {
    const el = DOMElements;
    if (el.deepepLLRadio?.checked) return 'll';
    if (el.deepepHTRadio?.checked) return 'ht';
    return 'auto';
  },
};

// ============================================================================
// SECTION 9: LABEL & KPI UPDATERS
// ============================================================================

const LabelUpdater = {
  /** Update all slider value labels and KPI displays */
  updateAll(timings) {
    const el = DOMElements;
    if (!el.tokensSlider) return;

    const params = ControlState.getModelParameters();

    // Slider value labels
    this._setText('tokensVal', `${params.tokenCount.toLocaleString()} tok`);
    this._setText('epVal', `${params.epRanks} ranks`);
    this._setText('latVal', `${params.latencyUs} µs`);
    this._setText('bwVal', `${params.bandwidthGBps} GB/s`);
    this._setText('ctokVal', `${formatNumber(params.computeUsPerToken, 2)} µs/tok`);
    this._setText('skewVal', formatNumber(params.skew, 2));

    this._setText('tokensVal-deepep', `${params.tokenCount.toLocaleString()} tok`);
    this._setText('epVal-deepep', `${params.epRanks} ranks`);
    this._setText('latVal-deepep', `${params.latencyUs} µs`);
    this._setText('bwVal-deepep', `${params.bandwidthGBps} GB/s`);
    this._setText('ctokVal-deepep', `${formatNumber(params.computeUsPerToken, 2)} µs/tok`);
    this._setText('skewVal-deepep', formatNumber(params.skew, 2));

    // KPI calculations
    const throughputSequential = params.tokenCount / (timings.sequentialTotalMs / 1000);
    const throughputDBO = params.tokenCount / (timings.dboOverlappedMs / 1000);
    const commFraction = (timings.dispatchMs + timings.combineMs) / timings.sequentialTotalMs;

    // KPI displays
    this._setText('summaryPill',
      `no‑overlap ${formatNumber(timings.sequentialTotalMs, 2)} ms • ` +
      `DBO ${formatNumber(timings.dboOverlappedMs, 2)} ms • ` +
      `comm ${(100 * commFraction).toFixed(0)}%`
    );
    this._setText('kpiLine',
      `Throughput ${Math.round(throughputSequential).toLocaleString()} tok/s → ` +
      `${Math.round(throughputDBO).toLocaleString()} tok/s (DBO)`
    );
    this._setText('timelineKpi',
      `dispatch ${formatNumber(timings.dispatchMs, 2)} ms • ` +
      `compute ${formatNumber(timings.computeMs, 2)} ms • ` +
      `combine ${formatNumber(timings.combineMs, 2)} ms`
    );
    this._setText('curveKpi', `Point: ${params.tokenCount.toLocaleString()} tokens/step`);

    // KaTeX equations
    this._updateEquations(timings);
  },

  _setText(elementId, text) {
    const element = getElement(elementId);
    if (element) element.textContent = text;
  },

  _updateEquations(timings) {
    if (!AnimationController.equationsDirty) return;

    if (window.katex) {
      const cComm = ThemeManager.colors.communication;
      const cComp = ThemeManager.colors.compute;
      const cComb = ThemeManager.colors.combine;
      const cDbo = ThemeManager.colors.dboHighlight;

      this._renderKatex('eqDispatch',
        `\\textcolor{${cComm}}{t_d} = \\textcolor{${cComm}}{L_{\\mathrm{eff}}} + \\frac{\\textcolor{${cComm}}{V}}{\\textcolor{${cComm}}{BW_{\\mathrm{eff}}}} = ${formatNumber(timings.dispatchMs, 3)}\\,\\mathrm{ms}`);
      this._renderKatex('eqCompute',
        `\\textcolor{${cComp}}{t_c} \\approx \\gamma \\cdot \\frac{B\\cdot k}{P} \\cdot \\textcolor{${cComp}}{c_{tok,\\mathrm{eff}}} = ${formatNumber(timings.computeMs, 3)}\\,\\mathrm{ms}`);
      this._renderKatex('eqNo',
        `t_{no} = 2\\textcolor{${cComm}}{t_d} + \\textcolor{${cComp}}{t_c} = ${formatNumber(timings.sequentialTotalMs, 3)}\\,\\mathrm{ms}`);
      this._renderKatex('eqDbo',
        `\\textcolor{${cDbo}}{t_{dbo}} \\approx \\max(2\\textcolor{${cComm}}{t_d},\\, \\textcolor{${cComp}}{t_c}) = ${formatNumber(timings.dboOverlappedMs, 3)}\\,\\mathrm{ms}`);
      AnimationController.equationsDirty = false;
    } else {
      // Fallback text
      this._setText('eqDispatch', `t_d = L_eff + V/BW_eff = ${formatNumber(timings.dispatchMs, 3)} ms`);
      this._setText('eqCompute', `t_c ≈ γ × (B×k/P) × c_tok,eff = ${formatNumber(timings.computeMs, 3)} ms`);
      this._setText('eqNo', `t_no = 2t_d + t_c = ${formatNumber(timings.sequentialTotalMs, 3)} ms`);
      this._setText('eqDbo', `t_dbo ≈ max(2t_d, t_c) = ${formatNumber(timings.dboOverlappedMs, 3)} ms`);
    }
  },

  _renderKatex(elementId, tex) {
    const element = getElement(elementId);
    if (!element || !window.katex) return;
    try {
      katex.render(tex, element, { throwOnError: false, displayMode: false, trust: true });
    } catch (err) {
      console.warn('KaTeX render error:', err);
      element.textContent = tex;
    }
  },

  /** Update DeepEP-specific labels */
  updateDeepEPLabels() {
    const params = ControlState.getModelParameters();
    const selectedMode = ControlState.getDeepEPMode();
    const effectiveMode = selectedMode === 'auto'
      ? DeepEPModel.pickOptimalMode(params.tokenCount, params.epRanks, params.latencyUs, params.bandwidthGBps)
      : selectedMode;

    const modeParams = DeepEPModel.getParameters(effectiveMode, params.latencyUs, params.bandwidthGBps);
    const commTime = DeepEPModel.calculateCommTimeMs(
      params.tokenCount, params.epRanks, effectiveMode, params.latencyUs, params.bandwidthGBps
    );
    const payloadBytes = (params.tokenCount * calculateBytesPerToken()) / params.epRanks;
    const payloadMiB = payloadBytes / (1024 * 1024);

    this._setText('deepepKpi', `${modeParams.name} selected • ${modeParams.description}`);
    this._setText('deepepPill',
      `one‑way payload ≈ ${formatNumber(payloadMiB, 2)} MiB • ` +
      `dispatch+combine ≈ ${formatNumber(commTime, 2)} ms`
    );
  },
};

// ============================================================================
// SECTION 10: SVG DRAWING UTILITIES
// ============================================================================

const SVGUtils = {
  /** Get dimensions of an SVG element */
  getDimensions(svgElement) {
    // Use parent container dimensions to avoid self-referential loops
    const container = svgElement.parentElement;
    if (!container) return { width: 100, height: 100 };
    
    // Use clientWidth/clientHeight to get the inner available space
    // This avoids measuring the SVG itself which might be in flux
    return {
      width: Math.max(1, container.clientWidth),
      height: Math.max(1, container.clientHeight),
    };
  },

  /** Prepare SVG for drawing (clear and set viewBox) */
  prepareSvg(svgElement) {
    if (!svgElement) return null;
    const { width, height } = this.getDimensions(svgElement);
    
    // Check if we actually need to update to avoid thrashing
    const currentViewBox = svgElement.getAttribute('viewBox');
    const newViewBox = `0 0 ${width} ${height}`;
    
    if (currentViewBox === newViewBox) {
        // If dimensions match, just clear content but don't touch attributes
        const svg = d3.select(svgElement);
        svg.selectAll('*').remove();
        return { svg, width, height };
    }

    const svg = d3.select(svgElement);
    // Only set viewBox, do NOT set width/height attributes to avoid layout thrashing
    // The CSS height: 100% will ensure it fills the container
    svg.attr('viewBox', newViewBox);
    svg.selectAll('*').remove();
    return { svg, width, height };
  },

  /** Generate log-scale tick values */
  generateLogTicks(min, max, targetCount) {
    const minExp = Math.floor(Math.log10(min));
    const maxExp = Math.ceil(Math.log10(max));
    const step = Math.max(1, Math.round((maxExp - minExp) / Math.max(1, targetCount - 1)));
    const values = [];
    for (let e = minExp; e <= maxExp; e += step) {
      values.push(Math.pow(10, e));
    }
    if (values[values.length - 1] < max) values.push(Math.pow(10, maxExp));
    return values.filter(v => v >= min && v <= max);
  },

  /** Prune tick array to max count */
  pruneTicks(values, maxTicks) {
    if (values.length <= maxTicks) return values;
    const step = Math.ceil(values.length / maxTicks);
    return values.filter((_, i) => i % step === 0);
  },
};

// ============================================================================
// SECTION 11: TOOLTIP CONTROLLER
// ============================================================================

const TooltipController = {
  hide() {
    const tooltip = DOMElements.timelineTooltip;
    if (!tooltip) return;
    tooltip.classList.remove('show');
    tooltip.setAttribute('aria-hidden', 'true');
  },

  show(blockInfo, clientX, clientY) {
    const tooltip = DOMElements.timelineTooltip;
    const container = DOMElements.timelineTooltipContainer;
    if (!tooltip || !container) return;

    const phaseMeta = {
      dispatch: { label: 'dispatch', term: 't_d' },
      compute: { label: 'compute', term: 't_c' },
      combine: { label: 'combine', term: 't_d (2nd comm leg)' },
    };
    const meta = phaseMeta[blockInfo.phase] || {
      label: blockInfo.phaseName || blockInfo.phase || 'phase',
      term: 't_phase',
    };

    const modelTimings = blockInfo.modelTimings || {};
    const dispatchMs = modelTimings.dispatchMs ?? 0;
    const combineMs = modelTimings.combineMs ?? 0;
    const computeMs = modelTimings.computeMs ?? 0;
    const sequentialMs = modelTimings.sequentialTotalMs ?? 0;
    const dboMs = modelTimings.dboOverlappedMs ?? 0;
    const fullPhaseMs = blockInfo.fullPhaseMs ?? blockInfo.durationMs;

    let laneLabel;
    if (blockInfo.enableDBO) {
      laneLabel = blockInfo.enableSplit ? `micro‑batch ${blockInfo.batchId}` : `batch ${blockInfo.batchId}`;
    } else {
      laneLabel = blockInfo.enableSplit ? 'single lane (split phase)' : 'single batch';
    }

    const endTime = blockInfo.startMs + blockInfo.durationMs;
    const phaseDurationText = blockInfo.enableSplit
      ? `${formatNumber(blockInfo.durationMs, 2)}ms (shown half of ${formatNumber(fullPhaseMs, 2)}ms)`
      : `${formatNumber(blockInfo.durationMs, 2)}ms`;
    const modelPhaseText = blockInfo.enableSplit
      ? `${meta.term}: full ${formatNumber(fullPhaseMs, 2)}ms • shown segment ${formatNumber(blockInfo.durationMs, 2)}ms`
      : `${meta.term}: ${formatNumber(fullPhaseMs, 2)}ms`;
    const modelStepText = blockInfo.enableDBO
      ? `DBO model: max(2t_d=${formatNumber(dispatchMs + combineMs, 2)}ms, t_c=${formatNumber(computeMs, 2)}ms) = ${formatNumber(dboMs, 2)}ms`
      : `No-overlap model: 2t_d + t_c = ${formatNumber(sequentialMs, 2)}ms`;

    tooltip.innerHTML =
      `<strong>${laneLabel}</strong> • ${meta.label}<br/>` +
      `<code>start ${formatNumber(blockInfo.startMs, 2)}ms • dur ${phaseDurationText} • end ${formatNumber(endTime, 2)}ms</code><br/>` +
      `<code>${modelPhaseText}</code><br/>` +
      `<code>${modelStepText}</code>`;

    const rect = container.getBoundingClientRect();
    const x = Math.max(10, Math.min(clientX - rect.left + 10, rect.width - 16));
    const y = Math.max(10, Math.min(clientY - rect.top + 10, rect.height - 16));

    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
    tooltip.classList.add('show');
    tooltip.setAttribute('aria-hidden', 'false');
  },
};

// ============================================================================
// SECTION 12: TIMELINE VISUALIZATION
// ============================================================================

const TimelineRenderer = {
  PHASE_COLORS: {
    dispatch: () => ThemeManager.colors.communication,
    compute: () => ThemeManager.colors.compute,
    combine: () => ThemeManager.colors.combine,
  },

  draw(timings, options) {
    try {
      const prep = SVGUtils.prepareSvg(DOMElements.timelineSvg);
      if (!prep) return;
      const { svg, width, height } = prep;

      // Global mouseleave to ensure tooltip hides when leaving the chart area
      svg.on('mouseleave', () => TooltipController.hide());

      const layout = this._calculateLayout(width, height, options.enableDBO);
      const schedule = this._buildSchedule(timings, options);
      const totalMs = this._calculateTotalDuration(schedule, options.enableDBO);
      const xScale = d3.scaleLinear()
        .domain([0, totalMs])
        .range([layout.chartX, layout.chartX + layout.chartWidth]);

      this._drawAxis(svg, xScale, layout);
      this._drawRows(svg, layout, options.enableDBO);
      this._drawBlocks(svg, schedule, xScale, layout, options.enableDBO, timings, options);

      if (options.enableAnimation) {
        this._drawPlayhead(svg, xScale, options.currentTimeMs, layout, options.enableDBO);
      }

      this._drawFooter(svg, layout, width, height, options.enableDBO, totalMs);
    } catch (error) {
      console.error("Error in TimelineRenderer.draw:", error);
      this._drawError(error.message);
    }
  },

  _calculateLayout(width, height, showParallel) {
    return {
      leftMargin: 70,
      rightMargin: 16,
      topMargin: 36,
      bottomMargin: 28,
      rowHeight: showParallel ? 38 : 32,
      rowGap: showParallel ? 16 : 0,
      blockHeight: showParallel ? 28 : 22,
      chartX: 70,
      chartWidth: Math.max(1, width - 70 - 16),
    };
  },

  _buildSchedule(timings, options) {
    const d = options.enableSplit ? timings.dispatchMs / 2 : timings.dispatchMs;
    const c = options.enableSplit ? timings.computeMs / 2 : timings.computeMs;
    const b = options.enableSplit ? timings.combineMs / 2 : timings.combineMs;

    const batchA = [];
    const batchB = [];

    if (!options.enableDBO) {
      // Sequential execution
      let t = 0;
      batchA.push({ phase: 'dispatch', start: t, duration: d }); t += d;
      batchA.push({ phase: 'compute', start: t, duration: c }); t += c;
      batchA.push({ phase: 'combine', start: t, duration: b });
    } else {
      // DBO pipelined execution
      const dispatchA = 0;
      const dispatchB = dispatchA + d;
      const computeA = dispatchA + d;
      const computeB = Math.max(dispatchB + d, computeA + c);
      const combineA = Math.max(computeA + c, dispatchB + d);
      const combineB = Math.max(computeB + c, combineA + b);

      batchA.push({ phase: 'dispatch', start: dispatchA, duration: d });
      batchA.push({ phase: 'compute', start: computeA, duration: c });
      batchA.push({ phase: 'combine', start: combineA, duration: b });

      batchB.push({ phase: 'dispatch', start: dispatchB, duration: d });
      batchB.push({ phase: 'compute', start: computeB, duration: c });
      batchB.push({ phase: 'combine', start: combineB, duration: b });
    }

    return { batchA, batchB };
  },

  _calculateTotalDuration(schedule, enableDBO) {
    if (!enableDBO) {
      const lastBlock = schedule.batchA[schedule.batchA.length - 1];
      return lastBlock.start + lastBlock.duration;
    }
    const lastA = schedule.batchA[schedule.batchA.length - 1];
    const lastB = schedule.batchB[schedule.batchB.length - 1];
    return Math.max(lastA.start + lastA.duration, lastB.start + lastB.duration);
  },

  _drawAxis(svg, xScale, layout) {
    const axisY = layout.topMargin - 12;
    const tickCount = Math.min(5, Math.max(3, Math.floor(layout.chartWidth / 120)));
    const axis = d3.axisTop(xScale)
      .ticks(tickCount)
      .tickFormat(d => `${formatNumber(d, 1)}ms`);

    svg.append('g')
      .attr('transform', `translate(0,${axisY})`)
      .call(axis);

    svg.selectAll('.domain').attr('stroke', ThemeManager.colors.border);
    svg.selectAll('.tick line').attr('stroke', ThemeManager.colors.border);
    svg.selectAll('.tick text')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-size', 9)
      .attr('font-family', ThemeManager.colors.fontMono)
      .attr('font-weight', 600);
  },

  _drawRows(svg, layout, showParallel) {
    const yA = layout.topMargin + layout.rowHeight / 2;
    const yB = yA + layout.rowHeight + layout.rowGap;

    this._drawRowBackground(svg, yA, 'Batch A', layout, false, !showParallel);
    if (showParallel) {
      this._drawRowBackground(svg, yB, 'Batch B', layout, true, false);
    }
  },

  _drawRowBackground(svg, yCenter, label, layout, isSecondary, singleBatch) {
    const y = yCenter - layout.rowHeight / 2;

    svg.append('rect')
      .attr('x', layout.chartX - 4)
      .attr('y', y)
      .attr('width', layout.chartWidth + 8)
      .attr('height', layout.rowHeight)
      .attr('rx', 8)
      .attr('fill', ThemeManager.getOverlayColor(isSecondary ? 0.015 : 0.025));

    svg.append('text')
      .attr('x', layout.chartX - 14)
      .attr('y', yCenter + 4)
      .attr('text-anchor', 'end')
      .attr('fill', ThemeManager.colors.foreground)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(singleBatch ? 'Batch' : label);
  },

  _drawBlocks(svg, schedule, xScale, layout, showParallel, timings, options) {
    const yA = layout.topMargin + layout.rowHeight / 2;
    const yB = yA + layout.rowHeight + layout.rowGap;
    const blockContext = {
      enableDBO: !!options.enableDBO,
      enableSplit: !!options.enableSplit,
      modelTimings: timings,
      phaseDurations: {
        dispatch: timings.dispatchMs,
        compute: timings.computeMs,
        combine: timings.combineMs,
      },
    };

    // Add stripe pattern for batch B
    const defs = svg.append('defs');
    defs.append('pattern')
      .attr('id', 'batch-b-stripes')
      .attr('width', 8)
      .attr('height', 8)
      .attr('patternUnits', 'userSpaceOnUse')
      .append('path')
      .attr('d', 'M0,0 L0,8')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.12);

    for (const block of schedule.batchA) {
      this._drawBlock(svg, 'A', block, yA, xScale, layout, false, blockContext);
    }

    if (showParallel) {
      for (const block of schedule.batchB) {
        this._drawBlock(svg, 'B', block, yB, xScale, layout, true, blockContext);
      }
    }
  },

  _drawBlock(svg, batchId, block, rowY, xScale, layout, isSecondaryBatch, blockContext) {
    const x = xScale(block.start);
    const w = Math.max(3, xScale(block.start + block.duration) - x);
    const h = layout.blockHeight;
    const y = rowY - h / 2;
    const color = this.PHASE_COLORS[block.phase]();
    const phaseName = block.phase;

    const rect = svg.append('rect')
      .attr('x', x)
      .attr('y', y)
      .attr('width', w)
      .attr('height', h)
      .attr('rx', 6)
      .attr('fill', color);

    if (isSecondaryBatch && w > 20) {
      svg.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', w)
        .attr('height', h)
        .attr('rx', 6)
        .attr('fill', 'url(#batch-b-stripes)');
    }

    if (w > 50) {
      svg.append('text')
        .attr('x', x + w / 2)
        .attr('y', rowY + 3.5)
        .attr('text-anchor', 'middle')
        .attr('fill', ThemeManager.getSurfaceWithAlpha(0.95))
        .attr('font-family', ThemeManager.colors.fontMono)
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .text(phaseName);
    }

    // Tooltip interaction
    const blockInfo = {
      batchId,
      phase: block.phase,
      phaseName,
      startMs: block.start,
      durationMs: block.duration,
      fullPhaseMs: blockContext?.phaseDurations?.[block.phase] ?? block.duration,
      enableDBO: blockContext?.enableDBO ?? false,
      enableSplit: blockContext?.enableSplit ?? false,
      modelTimings: blockContext?.modelTimings ?? null,
    };
    rect.on('mouseenter', (event) => TooltipController.show(blockInfo, event.clientX, event.clientY))
      .on('mousemove', (event) => TooltipController.show(blockInfo, event.clientX, event.clientY))
      .on('mouseleave', () => TooltipController.hide());
  },

  _drawPlayhead(svg, xScale, currentTimeMs, layout, showParallel) {
    const x = Math.round(xScale(currentTimeMs)) + 0.5;
    const yA = layout.topMargin + layout.rowHeight / 2;
    const yB = yA + layout.rowHeight + layout.rowGap;
    const bottomY = (showParallel ? yB : yA) + layout.rowHeight / 2 + 8;

    svg.append('line')
      .attr('x1', x)
      .attr('x2', x)
      .attr('y1', layout.topMargin - 20)
      .attr('y2', bottomY)
      .attr('stroke', 'rgba(225,29,72,0.6)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4 4');

    svg.append('circle')
      .attr('cx', x)
      .attr('cy', layout.topMargin - 24)
      .attr('r', 4)
      .attr('fill', 'rgba(225,29,72,0.8)');
  },

  _drawFooter(svg, layout, width, height, enableDBO, totalMs) {
    const yA = layout.topMargin + layout.rowHeight / 2;
    const yB = yA + layout.rowHeight + layout.rowGap;
    const contentBottom = enableDBO ? (yB + layout.rowHeight / 2) : (yA + layout.rowHeight / 2);
    const textY = Math.max(height - 14, contentBottom + 24);

    if (textY + 14 > height) {
      svg.attr('height', textY + 14).attr('viewBox', `0 0 ${width} ${textY + 14}`);
    }

    svg.append('text')
      .attr('x', layout.chartX)
      .attr('y', textY)
      .attr('fill', ThemeManager.colors.foreground)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(enableDBO ? '● DBO enabled — pipelined execution' : '○ Sequential — no overlap');

    svg.append('text')
      .attr('x', layout.chartX + layout.chartWidth)
      .attr('y', textY)
      .attr('text-anchor', 'end')
      .attr('fill', ThemeManager.colors.foreground)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(`Total: ${formatNumber(totalMs, 2)}ms`);
  },

  _drawError(message) {
    const svg = d3.select(DOMElements.timelineSvg);
    svg.selectAll('*').remove();
    svg.append('text')
      .attr('x', 10)
      .attr('y', 20)
      .attr('fill', 'red')
      .attr('font-size', 12)
      .text('Error: ' + message);
  },
};

// ============================================================================
// SECTION 13: SCALING CURVES VISUALIZATION
// ============================================================================

const CurvesRenderer = {
  draw() {
    const prep = SVGUtils.prepareSvg(DOMElements.curvesSvg);
    if (!prep) return;
    const { svg, width, height } = prep;

    const layout = {
      leftPad: 55, rightPad: 20, topPad: 20, bottomPad: 45,
      chartW: width - 55 - 20,
      chartH: height - 20 - 45,
    };

    // Define clip path
    svg.append('defs').append('clipPath')
      .attr('id', 'chart-clip')
      .append('rect')
      .attr('x', layout.leftPad)
      .attr('y', layout.topPad)
      .attr('width', layout.chartW)
      .attr('height', layout.chartH);

    const params = ControlState.getModelParameters();
    const dataPoints = this._generateDataPoints(params);
    const { xScale, yScale } = this._createScales(dataPoints, layout);

    this._drawBackground(svg, layout);
    this._drawGridAndAxes(svg, xScale, yScale, layout, dataPoints);
    
    // Create a group for curves with the clip path
    const curveGroup = svg.append('g').attr('clip-path', 'url(#chart-clip)');
    
    this._drawCurves(curveGroup, xScale, yScale, dataPoints);
    this._drawCurrentPoint(svg, xScale, yScale, params);
    this._drawLabels(svg, layout, width, height);
  },

  _generateDataPoints(params) {
    const MIN_EXP = -3, MAX_EXP = 20, SAMPLE_COUNT = 86;
    const points = [];
    let maxY = 0;

    for (let i = 0; i < SAMPLE_COUNT; i++) {
      const exp = MIN_EXP + (MAX_EXP - MIN_EXP) * (i / (SAMPLE_COUNT - 1));
      const tokens = Math.round(Math.pow(2, exp) * 8);
      const timings = calculateMoETimings(
        tokens, params.epRanks, params.latencyUs,
        params.bandwidthGBps, params.computeUsPerToken, params.skew
      );
      maxY = Math.max(maxY, timings.sequentialTotalMs, timings.dboOverlappedMs,
        timings.dispatchMs + timings.combineMs, timings.computeMs);
      points.push({ tokens, timings });
    }

    return { points, maxY };
  },

  _createScales(data, layout) {
    const xScale = d3.scaleLog()
      .domain([data.points[0].tokens, data.points[data.points.length - 1].tokens])
      .range([layout.leftPad, layout.leftPad + layout.chartW]);

    const yScale = d3.scaleLog()
      .domain([0.02, data.maxY * 1.15])
      .range([layout.topPad + layout.chartH, layout.topPad]);

    return { xScale, yScale };
  },

  _drawBackground(svg, layout) {
    svg.append('rect')
      .attr('x', layout.leftPad)
      .attr('y', layout.topPad)
      .attr('width', layout.chartW)
      .attr('height', layout.chartH)
      .attr('fill', ThemeManager.getOverlayColor(0.015));
  },

  _drawGridAndAxes(svg, xScale, yScale, layout, data) {
    const xTicks = SVGUtils.generateLogTicks(
      data.points[0].tokens,
      data.points[data.points.length - 1].tokens,
      Math.max(3, Math.floor(layout.chartW / 50))
    );

    const yTickBase = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
      .filter(v => v <= data.maxY * 1.15);
    const yTicks = SVGUtils.pruneTicks(yTickBase, Math.max(4, Math.floor(layout.chartH / 45)));

    // Y grid
    const yGrid = d3.axisLeft(yScale).tickValues(yTicks).tickSize(-layout.chartW).tickFormat('');
    svg.append('g')
      .attr('transform', `translate(${layout.leftPad},0)`)
      .call(yGrid)
      .selectAll('line').attr('stroke', ThemeManager.colors.border);

    // X grid
    const xGrid = d3.axisBottom(xScale).tickValues(xTicks).tickSize(-layout.chartH).tickFormat('');
    svg.append('g')
      .attr('transform', `translate(0,${layout.topPad + layout.chartH})`)
      .call(xGrid)
      .selectAll('line').attr('stroke', ThemeManager.colors.border);

    svg.selectAll('.domain').attr('stroke', ThemeManager.colors.border);

    // X axis
    const formatTokens = d => d >= 1e6 ? `${(d / 1e6).toFixed(0)}M` : d >= 1e3 ? `${(d / 1e3).toFixed(0)}k` : `${d}`;
    svg.append('g')
      .attr('transform', `translate(0,${layout.topPad + layout.chartH})`)
      .call(d3.axisBottom(xScale).tickValues(xTicks).tickFormat(formatTokens))
      .selectAll('text')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontMono)
      .attr('font-size', 9);

    // Y axis
    const formatMs = d => `${d >= 1 ? d3.format('~s')(d) : d.toFixed(2)}ms`;
    svg.append('g')
      .attr('transform', `translate(${layout.leftPad},0)`)
      .call(d3.axisLeft(yScale).tickValues(yTicks).tickFormat(formatMs))
      .selectAll('text')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontMono)
      .attr('font-size', 9);

    svg.selectAll('.tick line').attr('stroke', ThemeManager.colors.border);
  },

  _drawCurves(svg, xScale, yScale, data) {
    const makeLine = (getValue) => d3.line()
      .defined(p => isFinite(getValue(p.timings)))
      .x(p => xScale(p.tokens))
      .y(p => yScale(getValue(p.timings)));

    // Communication curve (dispatch + combine) - solid blue, thick
    svg.append('path')
      .attr('d', makeLine(t => t.dispatchMs + t.combineMs)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.communication)
      .attr('stroke-width', 3);

    // Compute curve - solid green, medium with markers
    svg.append('path')
      .attr('d', makeLine(t => t.computeMs)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.compute)
      .attr('stroke-width', 2.5);
    // Add circle markers for compute line
    const computeMarkerPoints = data.points.filter((_, i) => i % 8 === 0);
    svg.selectAll('.compute-marker')
      .data(computeMarkerPoints)
      .join('circle')
      .attr('class', 'compute-marker')
      .attr('cx', d => xScale(d.tokenCount))
      .attr('cy', d => yScale(d.computeMs))
      .attr('r', 3)
      .attr('fill', ThemeManager.colors.compute);

    // Sequential total (no-overlap) - dashed amber
    svg.append('path')
      .attr('d', makeLine(t => t.sequentialTotalMs)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.sequentialHighlight)
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '8 4');

    // DBO overlapped - dotted violet, thick
    svg.append('path')
      .attr('d', makeLine(t => t.dboOverlappedMs)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.dboHighlight)
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', '2 4');
    // Add diamond markers for DBO line
    const dboMarkerPoints = data.points.filter((_, i) => i % 8 === 4);
    svg.selectAll('.dbo-marker')
      .data(dboMarkerPoints)
      .join('path')
      .attr('class', 'dbo-marker')
      .attr('d', d3.symbol().type(d3.symbolDiamond).size(40))
      .attr('transform', d => {
         const x = xScale(d.tokenCount);
         const y = yScale(d.dboOverlappedMs);
         if (!isFinite(x) || !isFinite(y)) return null;
         return `translate(${x},${y})`;
      })
      .attr('fill', ThemeManager.colors.dboHighlight);
  },

  _drawCurrentPoint(svg, xScale, yScale, params) {
    const timings = calculateMoETimings(
      params.tokenCount, params.epRanks, params.latencyUs,
      params.bandwidthGBps, params.computeUsPerToken, params.skew
    );
    const isDBO = DOMElements.dboCheckbox?.checked;
    const yValue = isDBO ? timings.dboOverlappedMs : timings.sequentialTotalMs;
    const px = xScale(params.tokenCount);
    const py = yScale(yValue);

    svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 6)
      .attr('fill', ThemeManager.colors.foreground);
    svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 4)
      .attr('fill', isDBO ? ThemeManager.colors.dboHighlight : ThemeManager.colors.sequentialHighlight);
  },

  _drawLabels(svg, layout, width, height) {
    svg.append('text')
      .attr('x', layout.leftPad + layout.chartW / 2)
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 10)
      .text('batch tokens (log scale)');

    svg.append('text')
      .attr('transform', `translate(14,${layout.topPad + layout.chartH / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 10)
      .text('time (ms, log scale)');
  },
};

// ============================================================================
// SECTION 14: DEEPEP VISUALIZATIONS
// ============================================================================

const DeepEPRenderer = {
  drawCurve() {
    const prep = SVGUtils.prepareSvg(DOMElements.deepepCurveSvg);
    if (!prep) return;
    const { svg, width, height } = prep;

    const layout = { leftPad: 50, rightPad: 20, topPad: 20, bottomPad: 40 };
    layout.chartW = width - layout.leftPad - layout.rightPad;
    layout.chartH = height - layout.topPad - layout.bottomPad;

    const params = ControlState.getModelParameters();
    const data = this._generateCurveData(params);
    const { xScale, yScale } = this._createCurveScales(data, layout);

    this._drawCurveBackground(svg, layout);
    this._drawCurveAxes(svg, xScale, yScale, layout, data);
    this._drawCurveLines(svg, xScale, yScale, data);
    this._drawCurveCurrentPoint(svg, xScale, yScale, params);
    this._updateCurveKPI(params);
  },

  _generateCurveData(params) {
    const MIN_EXP = -3, MAX_EXP = 19, SAMPLES = 150;
    const points = [];
    let maxY = 0;

    for (let i = 0; i < SAMPLES; i++) {
      const exp = MIN_EXP + (MAX_EXP - MIN_EXP) * (i / (SAMPLES - 1));
      const tokens = Math.max(1, Math.round(Math.pow(2, exp) * 8));
      const llTime = Math.max(0.001, DeepEPModel.calculateCommTimeMs(tokens, params.epRanks, 'll', params.latencyUs, params.bandwidthGBps));
      const htTime = Math.max(0.001, DeepEPModel.calculateCommTimeMs(tokens, params.epRanks, 'ht', params.latencyUs, params.bandwidthGBps));
      maxY = Math.max(maxY, llTime, htTime);
      points.push({ tokens, llTime, htTime });
    }

    return { points, maxY };
  },

  _createCurveScales(data, layout) {
    return {
      xScale: d3.scaleLog()
        .domain([data.points[0].tokens, data.points[data.points.length - 1].tokens])
        .range([layout.leftPad, layout.leftPad + layout.chartW]),
      yScale: d3.scaleLog()
        .domain([0.005, data.maxY * 1.2])
        .range([layout.topPad + layout.chartH, layout.topPad]),
    };
  },

  _drawCurveBackground(svg, layout) {
    svg.append('rect')
      .attr('x', layout.leftPad)
      .attr('y', layout.topPad)
      .attr('width', layout.chartW)
      .attr('height', layout.chartH)
      .attr('fill', ThemeManager.getOverlayColor(0.015));
  },

  _drawCurveAxes(svg, xScale, yScale, layout, data) {
    const xTicks = SVGUtils.generateLogTicks(
      data.points[0].tokens,
      data.points[data.points.length - 1].tokens,
      Math.max(4, Math.floor(layout.chartW / 35))
    );
    const yTickBase = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
      .filter(v => v <= data.maxY * 1.2 && v >= 0.005);
    const yTicks = SVGUtils.pruneTicks(yTickBase, Math.max(4, Math.floor(layout.chartH / 45)));

    const formatMs = d => `${d >= 1 ? d3.format('~s')(d) : d.toFixed(2)}ms`;
    svg.append('g').attr('transform', `translate(${layout.leftPad},0)`)
      .call(d3.axisLeft(yScale).tickValues(yTicks).tickFormat(formatMs))
      .selectAll('text').attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontMono).attr('font-size', 9);

    const formatTokens = d => d >= 1e6 ? `${Math.round(d / 1e6)}M` : d >= 1e3 ? `${Math.round(d / 1e3)}k` : `${d}`;
    svg.append('g').attr('transform', `translate(0,${layout.topPad + layout.chartH})`)
      .call(d3.axisBottom(xScale).tickValues(xTicks).tickFormat(formatTokens))
      .selectAll('text').attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontMono).attr('font-size', 9);

    svg.selectAll('.domain').attr('stroke', ThemeManager.colors.border);
    svg.selectAll('.tick line').attr('stroke', ThemeManager.colors.border);
  },

  _drawCurveLines(svg, xScale, yScale, data) {
    const makeLine = getValue => d3.line()
      .defined(p => isFinite(getValue(p)))
      .x(p => xScale(p.tokens))
      .y(p => yScale(getValue(p)));

    // Find crossover
    let crossoverTokens = null;
    for (let i = 0; i < data.points.length - 1; i++) {
      const p1 = data.points[i];
      const p2 = data.points[i+1];
      const diff1 = p1.llTime - p1.htTime;
      const diff2 = p2.llTime - p2.htTime;
      if (diff1 < 0 && diff2 >= 0) {
        // Linear interpolation for X
        const t = Math.abs(diff1) / (Math.abs(diff1) + Math.abs(diff2));
        // Log scale interpolation
        const logT1 = Math.log(p1.tokens);
        const logT2 = Math.log(p2.tokens);
        const logTx = logT1 + t * (logT2 - logT1);
        crossoverTokens = Math.exp(logTx);
        break;
      }
    }

    // Draw regions
    if (crossoverTokens) {
      const xX = xScale(crossoverTokens);
      const h = yScale.range()[0] - yScale.range()[1];
      
      // LL Wins Region (Left)
      svg.append('rect')
        .attr('x', xScale.range()[0])
        .attr('y', yScale.range()[1])
        .attr('width', xX - xScale.range()[0])
        .attr('height', h)
        .attr('fill', ThemeManager.colors.dboHighlight)
        .attr('opacity', 0.05);

      // HT Wins Region (Right)
      svg.append('rect')
        .attr('x', xX)
        .attr('y', yScale.range()[1])
        .attr('width', xScale.range()[1] - xX)
        .attr('height', h)
        .attr('fill', ThemeManager.colors.communication)
        .attr('opacity', 0.05);

      // Crossover Line
      svg.append('line')
        .attr('x1', xX).attr('x2', xX)
        .attr('y1', yScale.range()[0])
        .attr('y2', yScale.range()[1])
        .attr('stroke', ThemeManager.colors.foreground)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 4')
        .attr('opacity', 0.4);

      // Label
      svg.append('text')
        .attr('x', xX + 4).attr('y', yScale.range()[1] + 10)
        .attr('fill', ThemeManager.colors.muted)
        .attr('font-family', ThemeManager.colors.fontMono)
        .attr('font-size', 9)
        .text('Crossover');
    }

    svg.append('path')
      .attr('d', makeLine(p => p.llTime)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.dboHighlight)
      .attr('stroke-width', 2.5);

    svg.append('path')
      .attr('d', makeLine(p => p.htTime)(data.points))
      .attr('fill', 'none')
      .attr('stroke', ThemeManager.colors.communication)
      .attr('stroke-width', 2.5);
  },

  _drawCurveCurrentPoint(svg, xScale, yScale, params) {
    const selectedMode = ControlState.getDeepEPMode();
    const effectiveMode = selectedMode === 'auto'
      ? DeepEPModel.pickOptimalMode(params.tokenCount, params.epRanks, params.latencyUs, params.bandwidthGBps)
      : selectedMode;

    const yValue = DeepEPModel.calculateCommTimeMs(params.tokenCount, params.epRanks, effectiveMode, params.latencyUs, params.bandwidthGBps);
    const px = xScale(params.tokenCount);
    const py = yScale(yValue);
    const color = effectiveMode === 'll' ? ThemeManager.colors.dboHighlight : ThemeManager.colors.communication;

    svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 6).attr('fill', ThemeManager.colors.foreground);
    svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 4).attr('fill', color);
  },

  _updateCurveKPI(params) {
    const llTime = DeepEPModel.calculateCommTimeMs(params.tokenCount, params.epRanks, 'll', params.latencyUs, params.bandwidthGBps);
    const htTime = DeepEPModel.calculateCommTimeMs(params.tokenCount, params.epRanks, 'ht', params.latencyUs, params.bandwidthGBps);
    const kpiEl = getElement('deepepCurveKpi');
    if (kpiEl) {
      kpiEl.textContent = `LL ${formatNumber(llTime, 2)} ms • HT ${formatNumber(htTime, 2)} ms @ ${params.tokenCount.toLocaleString()} tok`;
    }
    this._updateMetrics(params, llTime, htTime);
  },

  _updateMetrics(params, llComm, htComm) {
    const computeMs = calculateComputeTimeMs(
      params.tokenCount, params.epRanks, params.computeUsPerToken, params.skew
    );

    // DBO on: step time is max(comm, compute)
    // DBO off: step time is comm + compute
    const llTotalDBO = Math.max(llComm, computeMs);
    const htTotalDBO = Math.max(htComm, computeMs);
    const llTotalNoDBO = llComm + computeMs;
    const htTotalNoDBO = htComm + computeMs;

    const llTpsDBO = params.tokenCount / (llTotalDBO / 1000);
    const htTpsDBO = params.tokenCount / (htTotalDBO / 1000);
    const llTpsNoDBO = params.tokenCount / (llTotalNoDBO / 1000);
    const htTpsNoDBO = params.tokenCount / (htTotalNoDBO / 1000);

    const setText = (id, val) => {
      const el = getElement(id);
      if (el) el.textContent = val;
    };

    // Comm time is the same regardless of DBO
    setText('metric-ll-comm', `${formatNumber(llComm, 2)} ms`);
    setText('metric-ll-comm-nodbo', `${formatNumber(llComm, 2)} ms`);
    setText('metric-ht-comm', `${formatNumber(htComm, 2)} ms`);
    setText('metric-ht-comm-nodbo', `${formatNumber(htComm, 2)} ms`);

    setText('metric-ll-lat', `${formatNumber(llTotalDBO, 2)} ms`);
    setText('metric-ll-lat-nodbo', `${formatNumber(llTotalNoDBO, 2)} ms`);
    setText('metric-ht-lat', `${formatNumber(htTotalDBO, 2)} ms`);
    setText('metric-ht-lat-nodbo', `${formatNumber(htTotalNoDBO, 2)} ms`);

    setText('metric-ll-tps', `${Math.round(llTpsDBO).toLocaleString()}`);
    setText('metric-ll-tps-nodbo', `${Math.round(llTpsNoDBO).toLocaleString()}`);
    setText('metric-ht-tps', `${Math.round(htTpsDBO).toLocaleString()}`);
    setText('metric-ht-tps-nodbo', `${Math.round(htTpsNoDBO).toLocaleString()}`);
  },

  drawTopology(animationPhase = 0) {
    const prep = SVGUtils.prepareSvg(DOMElements.deepepTopoSvg);
    if (!prep) return;
    const { svg, width, height } = prep;

    const params = ControlState.getModelParameters();
    const selectedMode = ControlState.getDeepEPMode();
    const effectiveMode = selectedMode === 'auto'
      ? DeepEPModel.pickOptimalMode(params.tokenCount, params.epRanks, params.latencyUs, params.bandwidthGBps)
      : selectedMode;

    const topoKpi = getElement('deepepTopoKpi');
    if (topoKpi) {
      topoKpi.textContent = effectiveMode === 'll'
        ? 'Low‑latency: direct RDMA'
        : 'High‑throughput: NVLink → RDMA';
    }

    const pad = 16, topPad = 32;
    const nodeGap = 24;
    const nodeW = (width - 2 * pad - nodeGap) / 2;
    const nodeH = height - topPad - pad;

    // Background
    svg.append('rect')
      .attr('x', pad - 4).attr('y', topPad - 24)
      .attr('width', width - 2 * pad + 8).attr('height', height - topPad + 20)
      .attr('rx', 12).attr('fill', ThemeManager.getOverlayColor(0.01));

    // Mode label pill
    const modeLabel = effectiveMode === 'ht' ? 'RDMA via coordinators' : 'Direct RDMA (low latency)';
    const labelW = modeLabel.length * 6 + 24;
    
    svg.append('rect')
      .attr('x', width / 2 - labelW / 2).attr('y', topPad - 20)
      .attr('width', labelW).attr('height', 20)
      .attr('rx', 10)
      .attr('fill', ThemeManager.getSurfaceWithAlpha(0.92));

    svg.append('text')
      .attr('x', width / 2).attr('y', topPad - 7)
      .attr('text-anchor', 'middle')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 10).attr('font-weight', 600)
      .text(modeLabel);

    const nodeAGPUs = this._drawNode(svg, pad, topPad + 4, nodeW, nodeH, 'Node 0', effectiveMode);
    const nodeBGPUs = this._drawNode(svg, pad + nodeW + nodeGap, topPad + 4, nodeW, nodeH, 'Node 1', effectiveMode);

    this._drawConnections(svg, nodeAGPUs, nodeBGPUs, effectiveMode, animationPhase);
  },

  _drawNode(svg, x, y, w, h, label, mode) {
    // Node container
    svg.append('rect')
      .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
      .attr('rx', 12)
      .attr('fill', ThemeManager.getSurfaceWithAlpha(0.62))
      .attr('stroke', 'none');

    // Node header
    const headerH = 32;
    svg.append('path')
      .attr('d', `M${x},${y+headerH} h${w} v-${headerH-12} a12,12 0 0 0 -12,-12 h-${w-24} a12,12 0 0 0 -12,12 z`)
      .attr('fill', ThemeManager.getOverlayColor(0.02));

    svg.append('text')
      .attr('x', x + w / 2).attr('y', y + 20)
      .attr('text-anchor', 'middle')
      .attr('fill', ThemeManager.colors.foreground)
      .attr('font-family', ThemeManager.colors.fontBody)
      .attr('font-size', 11).attr('font-weight', 600)
      .attr('letter-spacing', '0.02em')
      .text(label.toUpperCase());

    // Grid layout calculation
    const gpuSize = 40; // Slightly smaller than 44 for better fit
    const cols = 4, rows = 2;
    
    // Calculate vertical spacing to center the grid in the remaining space
    const contentYStart = y + headerH;
    const contentH = h - headerH;
    const gridH = rows * gpuSize;
    const totalGapY = contentH - gridH;
    const marginY = totalGapY / 3; // Equal spacing: top, middle, bottom
    
    // Calculate horizontal spacing
    const gridW = cols * gpuSize;
    const totalGapX = w - gridW;
    const marginX = totalGapX / (cols + 1); // Equal spacing around all columns
    
    const gpus = [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Center coordinates
        const cx = x + marginX * (c + 1) + gpuSize * (c + 0.5);
        const cy = contentYStart + marginY * (r + 1) + gpuSize * (r + 0.5);
        
        const idx = r * cols + c;
        gpus.push({ cx, cy, idx, r: gpuSize / 2 });

        const isCoordinator = idx === 0;
        // In LL mode, rank 0 is just a normal rank, not a special coordinator
        const showAsCoordinator = isCoordinator && mode === 'ht';
        
        // Only highlight if it's a coordinator in HT mode. 
        // In LL mode, rank 0 is just a regular node (surface color).
        const gpuColor = showAsCoordinator
            ? ThemeManager.colors.communication
            : ThemeManager.getSurfaceWithAlpha(0.9);

        // GPU Chip Group
        const g = svg.append('g');
        
        // Main chip body
        g.append('rect')
          .attr('x', cx - gpuSize / 2).attr('y', cy - gpuSize / 2)
          .attr('width', gpuSize).attr('height', gpuSize)
          .attr('rx', 6)
          .attr('fill', gpuColor)
          .attr('stroke', 'none');

        // Inner detail (die)
        if (!showAsCoordinator) {
           g.append('rect')
            .attr('x', cx - gpuSize/2 + 6).attr('y', cy - gpuSize/2 + 6)
            .attr('width', gpuSize - 12).attr('height', gpuSize - 12)
            .attr('rx', 3)
            .attr('fill', ThemeManager.getOverlayColor(0.03));
        } else {
          // Coordinator indicator ring
          g.append('circle')
            .attr('cx', cx).attr('cy', cy)
            .attr('r', gpuSize/2 - 4)
            .attr('fill', 'none')
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1)
            .attr('opacity', 0.2);
        }

        g.append('text')
          .attr('x', cx).attr('y', cy + 4)
          .attr('text-anchor', 'middle')
          .attr('fill', showAsCoordinator ? '#ffffff' : ThemeManager.colors.mutedSecondary)
          .attr('font-family', ThemeManager.colors.fontMono)
          .attr('font-size', 12)
          .attr('font-weight', showAsCoordinator ? 700 : 500)
          .text(idx);
      }
    }

    return gpus;
  },

  _drawConnections(svg, nodeA, nodeB, mode, animPhase) {
    if (mode === 'ht') {
      // High-throughput: coordinator-to-coordinator (INTER-NODE, SLOW)
      const coordA = nodeA[0], coordB = nodeB[0];
      this._drawArrow(svg, coordA.cx + coordA.r, coordA.cy, coordB.cx - coordB.r, coordB.cy,
        ThemeManager.colors.communication, false, animPhase, 1.0); // Speed 1.0

      // Faint gather/scatter lines (INTRA-NODE, FAST)
      // Gather on Node A
      nodeA.slice(1).forEach(g => {
        this._drawArrow(svg, g.cx, g.cy, coordA.cx, coordA.cy,
          ThemeManager.getOverlayColor(0.2), false, animPhase, 3.0); // Speed 3.0
      });
      
      // Scatter on Node B
      nodeB.slice(1).forEach(g => {
        this._drawArrow(svg, coordB.cx, coordB.cy, g.cx, g.cy,
          ThemeManager.getOverlayColor(0.2), false, animPhase, 3.0); // Speed 3.0
      });
    } else {
      // Low-latency: direct all-to-all (INTER-NODE, SLOW)
      const pairs = [[1, 5], [2, 6], [4, 0], [7, 3]];
      for (const [ai, bi] of pairs) {
        if (nodeA[ai] && nodeB[bi]) {
          const p1 = nodeA[ai], p2 = nodeB[bi];
          const useCurve = Math.abs(p1.cy - p2.cy) > 10;
          this._drawArrow(svg, p1.cx + p1.r, p1.cy, p2.cx - p2.r, p2.cy,
            ThemeManager.colors.dboHighlight, useCurve, animPhase, 1.0); // Speed 1.0
        }
      }
    }
  },

  _drawArrow(svg, x1, y1, x2, y2, color, useCurve, animPhase, speed = 1.0) {
    const path = d3.path();
    path.moveTo(x1, y1);

    let mx, my;
    if (useCurve) {
      mx = (x1 + x2) / 2;
      my = Math.min(y1, y2) - 20;
      path.quadraticCurveTo(mx, my, x2, y2);
    } else {
      path.lineTo(x2, y2);
    }

    svg.append('path')
      .attr('d', path.toString())
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', speed > 1.5 ? 1 : 2) // Thinner lines for faint intra-node
      .attr('opacity', speed > 1.5 ? 0.6 : 1.0)
      .attr('stroke-dasharray', '4 4');

    // Arrowhead (only for main paths)
    if (speed <= 1.5) {
        const angle = useCurve ? Math.atan2(y2 - my, x2 - mx) : Math.atan2(y2 - y1, x2 - x1);
        const headLen = 6;
        const tip = d3.path();
        tip.moveTo(x2, y2);
        tip.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
        tip.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
        tip.closePath();
        svg.append('path').attr('d', tip.toString()).attr('fill', color);
    }

    // Animated packet
    if (animPhase > 0) {
      const t = (animPhase * speed) % 1;
      let px, py;
      if (useCurve) {
        const mt = 1 - t;
        px = mt * mt * x1 + 2 * mt * t * mx + t * t * x2;
        py = mt * mt * y1 + 2 * mt * t * my + t * t * y2;
      } else {
        px = x1 + (x2 - x1) * t;
        py = y1 + (y2 - y1) * t;
      }
      svg.append('circle')
        .attr('cx', px).attr('cy', py).attr('r', speed > 1.5 ? 2 : 3)
        .attr('fill', ThemeManager.colors.surface)
        .attr('stroke', color).attr('stroke-width', 1.25);
    }
  },

  drawTimeline() {
    const prep = SVGUtils.prepareSvg(DOMElements.deepepTimelineSvg);
    if (!prep) return;
    const { svg, width, height } = prep;

    const params = ControlState.getModelParameters();
    const selectedMode = ControlState.getDeepEPMode();
    const effectiveMode = selectedMode === 'auto'
      ? DeepEPModel.pickOptimalMode(params.tokenCount, params.epRanks, params.latencyUs, params.bandwidthGBps)
      : selectedMode;
    const modeParams = DeepEPModel.getParameters(effectiveMode, params.latencyUs, params.bandwidthGBps);
    const commTime = DeepEPModel.calculateCommTimeMs(
      params.tokenCount, params.epRanks, effectiveMode, params.latencyUs, params.bandwidthGBps
    );
    const timings = calculateMoETimings(
      params.tokenCount, params.epRanks, params.latencyUs,
      params.bandwidthGBps, params.computeUsPerToken, params.skew
    );
    const computeTime = timings.computeMs;

    const layout = { leftPad: 70, rightPad: 16, topPad: 32, bottomPad: 24 };
    layout.chartW = width - layout.leftPad - layout.rightPad;

    const stageTime = Math.max(commTime, computeTime);
    const totalTime = Math.max(stageTime * 2, commTime + computeTime) * 1.1;
    const xScale = d3.scaleLinear().domain([0, totalTime]).range([layout.leftPad, layout.leftPad + layout.chartW]);

    const rowH = 28, rowGap = 14;
    const ySlot0 = layout.topPad + rowH / 2;
    const ySlot1 = ySlot0 + rowH + rowGap;
    const yCompute = ySlot1 + rowH + rowGap;

    // Stripe pattern
    const defs = svg.append('defs');
    defs.append('pattern')
      .attr('id', 'overlap-stripes')
      .attr('width', 6).attr('height', 6)
      .attr('patternUnits', 'userSpaceOnUse')
      .append('path')
      .attr('d', 'M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2')
      .attr('stroke', ThemeManager.colors.compute)
      .attr('stroke-width', 1).attr('opacity', 0.2);

    // Axis
    svg.append('g').attr('transform', `translate(0,${layout.topPad - 10})`)
      .call(d3.axisTop(xScale).ticks(5).tickFormat(d => `${formatNumber(d, 1)}ms`));
    svg.selectAll('.domain')
      .attr('stroke', ThemeManager.colors.border)
      .attr('opacity', 0.45);
    svg.selectAll('.tick line')
      .attr('stroke', ThemeManager.colors.border)
      .attr('opacity', 0.35);
    svg.selectAll('.tick text')
      .attr('fill', ThemeManager.colors.muted)
      .attr('font-family', ThemeManager.colors.fontMono)
      .attr('font-size', 12).attr('font-weight', 600);

    // Row labels
    const drawRowLabel = (text, y) => {
      svg.append('text')
        .attr('x', layout.leftPad - 12).attr('y', y + 4)
        .attr('text-anchor', 'end')
        .attr('fill', ThemeManager.colors.foreground)
        .attr('font-family', ThemeManager.colors.fontBody)
        .attr('font-size', 12).attr('font-weight', 600)
        .text(text);
    };
    drawRowLabel('Slot 0', ySlot0);
    drawRowLabel('Slot 1', ySlot1);
    drawRowLabel('Compute', yCompute);

    // Row backgrounds
    [ySlot0, ySlot1, yCompute].forEach(y => {
      svg.append('rect')
        .attr('x', layout.leftPad).attr('y', y - rowH / 2)
        .attr('width', layout.chartW).attr('height', rowH)
        .attr('rx', 4).attr('fill', ThemeManager.getOverlayColor(0.015));
    });

    // Blocks
    const wComm = Math.max(3, xScale(commTime) - xScale(0));
    const wComp = Math.max(3, xScale(computeTime) - xScale(0));
    const x0 = layout.leftPad;

    // Double-buffer schedule
    this._drawKernelCommBlock(svg, x0, ySlot0, wComm, rowH, 'A', effectiveMode);
    this._drawTimelineBlock(svg, xScale(commTime), yCompute, wComp, rowH, ThemeManager.colors.compute, 'Comp A');
    this._drawKernelCommBlock(svg, xScale(commTime), ySlot1, wComm, rowH, 'B', effectiveMode);

    const timeKpi = getElement('deepepTimeKpi');
    if (timeKpi) {
      timeKpi.textContent =
        `${modeParams.name} comm ${formatNumber(commTime, 2)}ms • compute ${formatNumber(computeTime, 2)}ms`;
    }
  },

  _drawKernelCommBlock(svg, x, y, w, h, batchId, mode) {
    if (mode === 'll') {
      this._drawTimelineBlock(
        svg, x, y, w, h, ThemeManager.colors.dboHighlight, `RDMA ${batchId}`
      );
      return;
    }

    // HT kernel: depict hierarchical communication (gather -> RDMA -> scatter)
    const gatherFrac = 0.24;
    const rdmaFrac = 0.52;

    const wGather = Math.max(2, w * gatherFrac);
    const wRdma = Math.max(2, w * rdmaFrac);
    const wScatter = Math.max(2, w - wGather - wRdma);

    const cGatherScatter = ThemeManager.colors.batchSecondary;
    const cRdma = ThemeManager.colors.communication;

    this._drawTimelineBlock(svg, x, y, wGather, h, cGatherScatter, `G ${batchId}`, ThemeManager.getSurfaceWithAlpha(0.98), 54);
    this._drawTimelineBlock(svg, x + wGather, y, wRdma, h, cRdma, `RDMA ${batchId}`, ThemeManager.getSurfaceWithAlpha(0.98), 68);
    this._drawTimelineBlock(svg, x + wGather + wRdma, y, wScatter, h, cGatherScatter, `S ${batchId}`, ThemeManager.getSurfaceWithAlpha(0.98), 54);
  },

  _drawTimelineBlock(svg, x, y, w, h, color, label, labelColor = ThemeManager.getSurfaceWithAlpha(0.98), minLabelWidth = 40) {
    const blockH = h - 2;
    svg.append('rect')
      .attr('x', x).attr('y', y - blockH / 2)
      .attr('width', w).attr('height', blockH)
      .attr('rx', 4).attr('fill', color);

    if (label && w > minLabelWidth) {
      svg.append('text')
        .attr('x', x + w / 2).attr('y', y + 3.5)
        .attr('text-anchor', 'middle')
        .attr('fill', labelColor)
        .attr('paint-order', 'stroke')
        .attr('stroke', ThemeManager.getOverlayColor(0.35))
        .attr('stroke-width', 1.1)
        .attr('stroke-linejoin', 'round')
        .attr('font-family', ThemeManager.colors.fontMono)
        .attr('font-size', 12).attr('font-weight', 700)
        .attr('letter-spacing', '0.02em')
        .text(label);
    }
  },
};

// ============================================================================
// SECTION 15: SELF-TESTS
// ============================================================================

function runSelfTests() {
  const testContainer = getElement('tests');
  if (!testContainer) return;

  const results = [];
  const pass = (name) => results.push(`<span class="ok">✓</span> <span class="dim">${name}</span>`);
  const fail = (name, msg) => results.push(`<span class="bad">✗</span> <span class="dim">${name}</span> — ${msg}`);

  // Test 1: Model produces finite positive times
  const timings1 = calculateMoETimings(1024, 16, 25, 25, 0.45, 0.1);
  const allFinitePositive = [timings1.dispatchMs, timings1.computeMs, timings1.combineMs,
    timings1.sequentialTotalMs, timings1.dboOverlappedMs].every(x => isFinite(x) && x > 0);
  if (allFinitePositive) {
    pass('Model produces finite positive times');
  } else {
    fail('Model produces finite positive times', 'non-finite or non-positive values');
  }

  // Test 2: DBO time <= sequential time
  const timings2 = calculateMoETimings(8192, 32, 25, 50, 0.6, 0.05);
  if (timings2.dboOverlappedMs <= timings2.sequentialTotalMs + 1e-9) {
    pass('DBO time ≤ sequential time');
  } else {
    fail('DBO time ≤ sequential time', `dbo(${timings2.dboOverlappedMs}) > seq(${timings2.sequentialTotalMs})`);
  }

  // Test 3: Comm share decreases with larger batch
  const small = calculateMoETimings(256, 16, 25, 25, 0.45, 0.1);
  const large = calculateMoETimings(1048576, 16, 25, 25, 0.45, 0.1);
  const commShareSmall = (small.dispatchMs + small.combineMs) / small.sequentialTotalMs;
  const commShareLarge = (large.dispatchMs + large.combineMs) / large.sequentialTotalMs;
  if (commShareLarge <= commShareSmall + 0.01) {
    pass('Comm share decreases with larger batch');
  } else {
    fail('Comm share decreases with larger batch', `${commShareSmall.toFixed(3)} → ${commShareLarge.toFixed(3)}`);
  }

  // Test 4: Compute increases with skew
  const lowSkew = calculateMoETimings(4096, 16, 25, 25, 0.45, 0.0);
  const highSkew = calculateMoETimings(4096, 16, 25, 25, 0.45, 1.0);
  if (highSkew.computeMs >= lowSkew.computeMs) {
    pass('Compute time increases with skew');
  } else {
    fail('Compute time increases with skew', `${lowSkew.computeMs.toFixed(3)} → ${highSkew.computeMs.toFixed(3)}`);
  }

  testContainer.innerHTML = results.join('<br/>');
}

// ============================================================================
// SECTION 16: MAIN RENDER LOOP & EVENT HANDLING
// ============================================================================

let cachedTimings = null;

function computeCurrentTimings() {
  const params = ControlState.getModelParameters();
  return calculateMoETimings(
    params.tokenCount, params.epRanks, params.latencyUs,
    params.bandwidthGBps, params.computeUsPerToken, params.skew
  );
}

function renderStaticContent() {
  cachedTimings = computeCurrentTimings();
  LabelUpdater.updateAll(cachedTimings);
  CurvesRenderer.draw();
  LabelUpdater.updateDeepEPLabels();
  DeepEPRenderer.drawCurve();
  DeepEPRenderer.drawTopology();
  DeepEPRenderer.drawTimeline();
}

function renderTimeline(currentTimeMs) {
  if (!DOMElements.timelineSvg) return;
  const timings = cachedTimings || computeCurrentTimings();
  TimelineRenderer.draw(timings, {
    enableDBO: DOMElements.dboCheckbox?.checked ?? true,
    enableSplit: DOMElements.splitCheckbox?.checked ?? true,
    enableAnimation: DOMElements.animateCheckbox?.checked ?? true,
    currentTimeMs,
  });
}

function renderAll() {
  renderStaticContent();
  renderTimeline(0);
}

function animationTick(now) {
  const mainAnimEnabled = DOMElements.animateCheckbox?.checked;
  const deepepAnimEnabled = DOMElements.getDeepepAnimateCheckbox()?.checked;

  if (!mainAnimEnabled && !deepepAnimEnabled) {
    AnimationController.stop();
    return;
  }

  const elapsed = AnimationController.tick(now);
  const timings = cachedTimings || computeCurrentTimings();

  if (mainAnimEnabled) {
    const enableSplit = DOMElements.splitCheckbox?.checked;
    const enableDBO = DOMElements.dboCheckbox?.checked;
    const d = enableSplit ? timings.dispatchMs / 2 : timings.dispatchMs;
    const c = enableSplit ? timings.computeMs / 2 : timings.computeMs;
    const b = enableSplit ? timings.combineMs / 2 : timings.combineMs;

    let cycleDuration;
    if (!enableDBO) cycleDuration = (d + c + b) * 2;
    else cycleDuration = (d + c + b) + 2 * Math.max(d + b, c) + (d + b);

    const currentTimeMs = (elapsed * 1000) % cycleDuration;
    renderTimeline(currentTimeMs);
  } else {
    renderTimeline(0);
  }

  if (deepepAnimEnabled) {
    const topoPhase = (elapsed % 1.5) / 1.5;
    DeepEPRenderer.drawTopology(topoPhase);
  } else {
    DeepEPRenderer.drawTopology(0);
  }
}

function onStateChange() {
  try {
    renderStaticContent();
    const mainAnimEnabled = DOMElements.animateCheckbox?.checked;
    const deepepAnimEnabled = DOMElements.getDeepepAnimateCheckbox()?.checked;

    if (mainAnimEnabled || deepepAnimEnabled) {
      AnimationController.start(animationTick);
    } else {
      AnimationController.stop();
      renderTimeline(0);
      DeepEPRenderer.drawTopology(0);
    }
  } catch (error) {
    console.error('Error in onStateChange:', error);
  }
}

// ============================================================================
// SECTION 17: EVENT WIRING & INITIALIZATION
// ============================================================================

function wireEventListeners() {
  const el = DOMElements;

  // Preset buttons
  el.resetButton?.addEventListener('click', () => ControlState.applyToControls(DEFAULT_PARAMETERS));
  el.decodePresetButton?.addEventListener('click', () => ControlState.applyToControls({
    tokensExponent: 6, expertParallelism: 16, latencyMicroseconds: 35, bandwidthGBps: 25,
    computeMicrosecondsPerToken: 0.45, loadImbalanceSkew: 0.10,
    enableDBO: true, enableMicroBatchSplit: true, enableAnimation: true,
  }));
  el.prefillPresetButton?.addEventListener('click', () => ControlState.applyToControls({
    tokensExponent: 16, expertParallelism: 32, latencyMicroseconds: 25, bandwidthGBps: 50,
    computeMicrosecondsPerToken: 0.60, loadImbalanceSkew: 0.06,
    enableDBO: true, enableMicroBatchSplit: true, enableAnimation: true,
  }));

  // Sliders and checkboxes
  const controls = [
    el.tokensSlider, el.epSlider, el.latencySlider, el.bandwidthSlider,
    el.computeSlider, el.skewSlider, el.dboCheckbox, el.splitCheckbox,
    el.tokensSlider2, el.epSlider2, el.latencySlider2, el.bandwidthSlider2,
    el.computeSlider2, el.skewSlider2
  ];

  const syncHandler = (e) => {
    const target = e.target;
    const id = target.id;
    const val = target.value;

    // Sync logic
    if (id.startsWith('tokens')) {
      if (el.tokensSlider) el.tokensSlider.value = val;
      if (el.tokensSlider2) el.tokensSlider2.value = val;
    } else if (id.startsWith('ep')) {
      if (el.epSlider) el.epSlider.value = val;
      if (el.epSlider2) el.epSlider2.value = val;
    } else if (id.startsWith('lat')) {
      if (el.latencySlider) el.latencySlider.value = val;
      if (el.latencySlider2) el.latencySlider2.value = val;
    } else if (id.startsWith('bw')) {
      if (el.bandwidthSlider) el.bandwidthSlider.value = val;
      if (el.bandwidthSlider2) el.bandwidthSlider2.value = val;
    } else if (id.startsWith('ctok')) {
      if (el.computeSlider) el.computeSlider.value = val;
      if (el.computeSlider2) el.computeSlider2.value = val;
    } else if (id.startsWith('skew')) {
      if (el.skewSlider) el.skewSlider.value = val;
      if (el.skewSlider2) el.skewSlider2.value = val;
    }

    onStateChange();
  };

  controls.forEach(ctrl => {
    if (ctrl) {
      ctrl.addEventListener('input', syncHandler);
      ctrl.addEventListener('change', syncHandler);
    }
  });

  el.animateCheckbox?.addEventListener('change', onStateChange);
  el.getDeepepAnimateCheckbox()?.addEventListener('change', onStateChange);

  // DeepEP mode radios
  [el.deepepAutoRadio, el.deepepLLRadio, el.deepepHTRadio].forEach(radio => {
    radio?.addEventListener('change', onStateChange);
  });

  // Mechanics widget sync
  const mechLL = getElement('mechLL');
  const mechHT = getElement('mechHT');
  const mechAnim = getElement('mechAnim');

  // Sync Mechanics -> Analysis (LL/HT)
  const syncToAnalysis = (mode) => {
    if (mode === 'll' && el.deepepLLRadio) {
      el.deepepLLRadio.checked = true;
      el.deepepLLRadio.dispatchEvent(new Event('change'));
    } else if (mode === 'ht' && el.deepepHTRadio) {
      el.deepepHTRadio.checked = true;
      el.deepepHTRadio.dispatchEvent(new Event('change'));
    }
  };

  mechLL?.addEventListener('change', () => syncToAnalysis('ll'));
  mechHT?.addEventListener('change', () => syncToAnalysis('ht'));

  // Sync Analysis -> Mechanics (LL/HT)
  const syncToMechanics = () => {
    if (el.deepepLLRadio?.checked && mechLL) mechLL.checked = true;
    if (el.deepepHTRadio?.checked && mechHT) mechHT.checked = true;
  };
  el.deepepLLRadio?.addEventListener('change', syncToMechanics);
  el.deepepHTRadio?.addEventListener('change', syncToMechanics);

  // Sync Animation
  // The global animation loop checks getDeepepAnimateCheckbox().
  // We need to make sure that returns the active one or they stay synced.
  // Let's just sync them if both exist.
  const dpAnim = el.getDeepepAnimateCheckbox(); // This might find the one in Analysis if it exists? No, getDeepepAnimateCheckbox looks for 'dpAnimToggle'
  
  // Wait, in DeepEPMechanics I named it 'mechAnim'. In DeepEPAnalysis I removed it.
  // So getDeepepAnimateCheckbox needs to find 'mechAnim'.
  
  if (mechAnim) {
      mechAnim.addEventListener('change', onStateChange);
  }
}

const resizeObserver = new ResizeObserver(() => {
  clearTimeout(resizeObserver._debounceTimer);
  resizeObserver._debounceTimer = setTimeout(() => {
    renderAll();
    if (DOMElements.animateCheckbox?.checked) {
      AnimationController.start(animationTick);
    }
  }, 50);
});

function setupResizeObservers() {
  const containers = [
    DOMElements.timelineSvg, DOMElements.curvesSvg,
    DOMElements.deepepCurveSvg, DOMElements.deepepTopoSvg, DOMElements.deepepTimelineSvg,
  ].filter(Boolean).map(svg => svg.parentElement).filter(Boolean);

  containers.forEach(container => resizeObserver.observe(container));
}

function waitForKaTeX() {
  const katexReady = window.katex && typeof window.katex.render === 'function';
  const autoRenderReady = typeof renderMathInElement === 'function' || typeof window.renderMathInElement === 'function';

  if (katexReady && autoRenderReady) {
    const renderFn = renderMathInElement || window.renderMathInElement;
    try {
      renderFn(document.body, {
        delimiters: [
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
        trust: true,
      });
    } catch (err) {
      console.warn('KaTeX auto-render error:', err);
    }
    AnimationController.equationsDirty = true;
    renderAll();
  } else {
    const retryCount = (waitForKaTeX._retryCount || 0) + 1;
    if (retryCount < 30) {
      waitForKaTeX._retryCount = retryCount;
      setTimeout(waitForKaTeX, 100);
    } else {
      console.warn('KaTeX failed to load after 3 seconds, continuing without it');
      renderAll();
    }
  }
}

function initialize(retryCount = 0) {
  DOMElements.init();

  // Check for critical elements from BOTH widgets to ensure full load
  const deepepCheckbox = DOMElements.getDeepepAnimateCheckbox();

  if (!DOMElements.timelineSvg || !DOMElements.curvesSvg || !DOMElements.tokensSlider || !deepepCheckbox) {
    if (retryCount < 50) { // Max 2.5 seconds
        if (retryCount % 10 === 0) { // Log every 10th retry
            console.warn(`Required DOM elements not found (attempt ${retryCount}), retrying...`, {
              timelineSvg: !!DOMElements.timelineSvg,
              curvesSvg: !!DOMElements.curvesSvg,
              tokensSlider: !!DOMElements.tokensSlider,
              deepepCheckbox: !!deepepCheckbox
            });
        }
        setTimeout(() => initialize(retryCount + 1), 50);
        return;
    } else {
        console.error("Giving up on finding elements. Some features may be broken.");
    }
  }

  wireEventListeners();
  ControlState.applyToControls(DEFAULT_PARAMETERS);
  runSelfTests();
  setupResizeObservers();

  setTimeout(() => {
    renderAll();
    waitForKaTeX();
    const mainAnimEnabled = DOMElements.animateCheckbox?.checked;
    const deepepAnimEnabled = DOMElements.getDeepepAnimateCheckbox()?.checked;
    if (mainAnimEnabled || deepepAnimEnabled) {
      AnimationController.start(animationTick);
    }
  }, 50);
}

// Start initialization when DOM is ready (guard against SSR)
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => initialize());
  } else {
    initialize();
  }
}

})();
