/**
 * IPL Chase Pressure Predictor — Frontend Logic
 * -----------------------------------------------
 * Handles form interaction, validation, API calls,
 * and animated result rendering.
 */

/* ══════════════════════════════════════════════
   DOM References
══════════════════════════════════════════════ */
const form           = document.getElementById('predict-form');
const predictBtn     = document.getElementById('predict-btn');
const btnText        = predictBtn.querySelector('.btn-text');
const btnSpinner     = predictBtn.querySelector('.btn-spinner');
const globalError    = document.getElementById('global-error');
const globalErrorMsg = document.getElementById('global-error-msg');

const emptyState     = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');

// Gauge
const gaugeArc  = document.getElementById('gauge-arc');
const gaugePct  = document.getElementById('gauge-pct');
const vsBarWin  = document.getElementById('vs-bar-win');
const vsWinLbl  = document.getElementById('vs-win-label');
const vsLoseLbl = document.getElementById('vs-lose-label');
const explanTxt = document.getElementById('explanation-text');

// Stats
const statRuns    = document.getElementById('stat-runs');
const statOvers   = document.getElementById('stat-overs');
const statWickets = document.getElementById('stat-wickets');
const statPhase   = document.getElementById('stat-phase');

// Rate
const crrVal      = document.getElementById('crr-val');
const rrrVal      = document.getElementById('rrr-val');
const rateGapFill = document.getElementById('rate-gap-fill');
const rateGapNote = document.getElementById('rate-gap-note');

// Pressure
const pressureScore   = document.getElementById('pressure-score');
const pressureZoneBdg = document.getElementById('pressure-zone-badge');
const pressureZoneName= document.getElementById('pressure-zone-name');
const pressureFill    = document.getElementById('pressure-fill');
const compRateRatio   = document.getElementById('comp-rate-ratio');
const compWicketFact  = document.getElementById('comp-wicket-factor');
const compTimeFact    = document.getElementById('comp-time-factor');
const compPressure    = document.getElementById('comp-pressure');

const overPreview     = document.getElementById('over-preview');
const wicketSelector  = document.getElementById('wicket-selector');

/* ══════════════════════════════════════════════
   Wicket selector (0–9 buttons)
══════════════════════════════════════════════ */
let selectedWickets = 0;

(function buildWicketSelector() {
  for (let i = 0; i <= 9; i++) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'wicket-btn' + (i === 0 ? ' active' : '');
    btn.textContent = i;
    btn.dataset.value = i;
    btn.addEventListener('click', () => selectWicket(i));
    wicketSelector.appendChild(btn);
  }
})();

function selectWicket(n) {
  selectedWickets = n;
  document.getElementById('wickets_lost').value = n;
  wicketSelector.querySelectorAll('.wicket-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.value) === n);
  });
  clearFieldError('wickets_lost');
}

/* ══════════════════════════════════════════════
   Over / balls preview
══════════════════════════════════════════════ */
document.getElementById('balls_completed').addEventListener('input', function () {
  const val = parseInt(this.value);
  if (!isNaN(val) && val >= 0 && val <= 120) {
    const ov = Math.floor(val / 6);
    const bl = val % 6;
    overPreview.textContent = `→ Over ${ov}.${bl}  (${120 - val} ball${120 - val !== 1 ? 's' : ''} remaining)`;
  } else {
    overPreview.textContent = '';
  }
  clearFieldError('balls_completed');
});

/* ══════════════════════════════════════════════
   Client-side validation
══════════════════════════════════════════════ */
function clearFieldError(name) {
  const el = document.getElementById('err-' + name);
  if (el) el.textContent = '';
  const inp = document.getElementById(name);
  if (inp) inp.classList.remove('is-error');
}

function setFieldError(name, msg) {
  const el = document.getElementById('err-' + name);
  if (el) el.textContent = msg;
  const inp = document.getElementById(name);
  if (inp) inp.classList.add('is-error');
}

function validateForm() {
  let ok = true;
  const target   = parseInt(document.getElementById('target_score').value);
  const current  = parseInt(document.getElementById('current_score').value);
  const balls    = parseInt(document.getElementById('balls_completed').value);
  const wickets  = selectedWickets;

  clearFieldError('target_score');
  clearFieldError('current_score');
  clearFieldError('balls_completed');
  clearFieldError('wickets_lost');

  if (isNaN(target) || target < 1 || target > 500) {
    setFieldError('target_score', 'Enter a target between 1 and 500.');
    ok = false;
  }
  if (isNaN(current) || current < 0) {
    setFieldError('current_score', 'Current score cannot be negative.');
    ok = false;
  }
  if (!isNaN(target) && !isNaN(current) && current >= target) {
    setFieldError('current_score', 'Current score must be less than target (chase in progress).');
    ok = false;
  }
  if (isNaN(balls) || balls < 0 || balls > 119) {
    setFieldError('balls_completed', 'Balls completed must be 0–119.');
    ok = false;
  }
  return ok;
}

/* ══════════════════════════════════════════════
   Form submit
══════════════════════════════════════════════ */
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  globalError.hidden = true;

  if (!validateForm()) return;

  setLoading(true);

  const payload = {
    target_score:    parseInt(document.getElementById('target_score').value),
    current_score:   parseInt(document.getElementById('current_score').value),
    balls_completed: parseInt(document.getElementById('balls_completed').value),
    wickets_lost:    selectedWickets,
    toss_decision:   document.getElementById('toss_decision').value,
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      const msg = data.details
        ? Object.values(data.details).join(' ')
        : (data.error || 'An error occurred.');
      showGlobalError(msg);
      return;
    }

    renderResults(data);
  } catch (err) {
    showGlobalError('Network error — could not reach the server.');
  } finally {
    setLoading(false);
  }
});

/* ══════════════════════════════════════════════
   Loading state
══════════════════════════════════════════════ */
function setLoading(on) {
  predictBtn.disabled = on;
  btnText.hidden      = on;
  btnSpinner.hidden   = !on;
}

function showGlobalError(msg) {
  globalErrorMsg.textContent = msg;
  globalError.hidden = false;
}

/* ══════════════════════════════════════════════
   Render results
══════════════════════════════════════════════ */
function renderResults(d) {
  // Show results, hide empty state
  emptyState.hidden      = true;
  resultsContent.hidden  = false;
  resultsContent.classList.remove('fade-in');
  // Force reflow then re-add animation
  void resultsContent.offsetWidth;
  resultsContent.classList.add('fade-in');

  const prob = d.win_probability;  // 0–1

  // ── Gauge ──────────────────────────────────────────────────────
  const circumference = 2 * Math.PI * 90;   // r=90
  const filled        = circumference * prob;
  gaugeArc.setAttribute(
    'stroke-dasharray',
    `${filled.toFixed(2)} ${circumference.toFixed(2)}`
  );

  const gaugeColor = probColor(prob);
  gaugeArc.setAttribute('stroke', gaugeColor);
  gaugePct.textContent = `${(prob * 100).toFixed(1)}%`;
  gaugePct.setAttribute('fill', gaugeColor);

  // Win/loss bar
  const pct = (prob * 100).toFixed(1);
  const lpct = ((1 - prob) * 100).toFixed(1);
  vsBarWin.style.width = pct + '%';
  vsWinLbl.textContent  = pct + '%';
  vsLoseLbl.textContent = lpct + '%';

  // Explanation
  explanTxt.textContent = d.explanation;

  // ── Stats ──────────────────────────────────────────────────────
  statRuns.textContent    = d.runs_remaining;
  statOvers.textContent   = d.overs_remaining;
  statWickets.textContent = d.wickets_in_hand;
  statPhase.textContent   = d.phase;

  // ── Run rates ──────────────────────────────────────────────────
  const crr = parseFloat(d.current_run_rate);
  const rrr = parseFloat(d.required_run_rate);

  crrVal.textContent = crr.toFixed(2);
  rrrVal.textContent = rrr === 999 ? '∞' : rrr.toFixed(2);

  // Colour the CRR card
  const crrCard = crrVal.closest('.rate-card');
  const rrrCard = rrrVal.closest('.rate-card');
  crrCard.classList.remove('rate-ahead', 'rate-behind');
  rrrCard.classList.remove('rate-ahead', 'rate-behind');

  if (crr > 0 && rrr > 0) {
    if (crr >= rrr) {
      crrCard.classList.add('rate-ahead');
    } else {
      crrCard.classList.add('rate-behind');
    }
  }

  // Gap bar: fill = crr / max(crr, rrr)
  const maxRate = Math.max(crr, rrr, 1);
  const fillPct = Math.min((crr / maxRate) * 100, 100);
  rateGapFill.style.width = fillPct + '%';
  rateGapFill.style.background = crr >= rrr ? '#22C55E' : '#EF4444';

  if (crr === 0) {
    rateGapNote.textContent = 'No balls bowled yet.';
  } else if (crr >= rrr + 0.5) {
    const ahead = (crr - rrr).toFixed(1);
    rateGapNote.textContent = `Scoring ${ahead} RPO faster than required — comfortable position.`;
  } else if (Math.abs(crr - rrr) < 0.5) {
    rateGapNote.textContent = `Neck-and-neck with the required rate.`;
  } else {
    const behind = (rrr - crr).toFixed(1);
    rateGapNote.textContent = `${behind} RPO behind the required rate — need acceleration.`;
  }

  // ── Pressure ────────────────────────────────────────────────────
  const p       = parseFloat(d.pressure);
  const zoneName= d.pressure_zone;
  const zoneCl  = d.pressure_zone_color;
  const zoneBg  = d.pressure_zone_bg;

  pressureScore.textContent = p.toFixed(2);

  pressureZoneBdg.style.background = zoneBg;
  pressureZoneBdg.style.color      = zoneCl;
  pressureZoneBdg.style.border     = `1px solid ${zoneCl}40`;
  pressureZoneName.textContent     = zoneName;

  // Pressure fill: map 0–5+ to 0–100%
  const pPos = Math.min((p / 5) * 100, 100);
  pressureFill.style.left = pPos + '%';

  // Breakdown components
  compRateRatio.textContent  = parseFloat(d.rate_ratio).toFixed(2);
  compWicketFact.textContent = parseFloat(d.wicket_factor).toFixed(2);
  compTimeFact.textContent   = parseFloat(d.time_factor).toFixed(2);
  compPressure.textContent   = p.toFixed(2);
}

/* ══════════════════════════════════════════════
   Colour helpers
══════════════════════════════════════════════ */
function probColor(p) {
  if (p >= 0.65) return '#22C55E';   // green
  if (p >= 0.45) return '#F59E0B';   // amber
  return '#EF4444';                  // red
}

/* ══════════════════════════════════════════════
   Clear errors on input change
══════════════════════════════════════════════ */
['target_score', 'current_score'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => clearFieldError(id));
});
