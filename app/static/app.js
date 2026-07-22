/**
 * BEL Offline AI Interface — Frontend Logic
 * No external dependencies. No network calls outside 127.0.0.1.
 */

'use strict';

// ── Fault code reference data (verbatim from IRL_Fault_Codes.pdf) ────────────
// Used to populate the quick-reference table without an extra API call.
const FAULT_CODES = [
  { sl: 1,  code: '0x0001', desc: 'System Status',                         remarks: 'System OK' },
  { sl: 2,  code: '0x0002', desc: 'DEBAR Zone - fire restricted',          remarks: 'Fire command in Debar zone' },
  { sl: 3,  code: '0x0003', desc: 'Fire aborted',                          remarks: 'Dynamic accuracy failed (+/- 0.5 Deg)' },
  { sl: 4,  code: '0x0004', desc: 'Fire interlocks',                       remarks: 'Internal interlocks in IRL failed in time of firing' },
  { sl: 5,  code: '0x0005', desc: 'R1 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 6,  code: '0x0006', desc: 'R2 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 7,  code: '0x0007', desc: 'R3 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 8,  code: '0x0008', desc: 'R4 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 9,  code: '0x0009', desc: 'R5 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 10, code: '0x0010', desc: 'R6 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 11, code: '0x0011', desc: 'R7 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 12, code: '0x0012', desc: 'R8 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 13, code: '0x0013', desc: 'R9 Misfired',                           remarks: 'Fired but still rocket present' },
  { sl: 14, code: '0x0014', desc: 'R10 Misfired',                          remarks: 'Fired but still rocket present' },
  { sl: 15, code: '0x0015', desc: 'R11 Misfired',                          remarks: 'Fired but still rocket present' },
  { sl: 16, code: '0x0016', desc: 'R12 Misfired',                          remarks: 'Fired but still rocket present' },
  { sl: 17, code: '0x0017', desc: 'Depth setting failed',                  remarks: 'Depth setting is stopped on some error' },
  { sl: 18, code: '0x0018', desc: 'Throw range invalid for R1 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 19, code: '0x0019', desc: 'Throw range invalid for R2 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 20, code: '0x0020', desc: 'Throw range invalid for R3 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 21, code: '0x0021', desc: 'Throw range invalid for R4 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 22, code: '0x0022', desc: 'Throw range invalid for R5 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 23, code: '0x0023', desc: 'Throw range invalid for R6 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 24, code: '0x0024', desc: 'Throw range invalid for R7 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 25, code: '0x0025', desc: 'Throw range invalid for R8 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 26, code: '0x0026', desc: 'Throw range invalid for R9 (Note 6)',   remarks: 'not a valid throw range' },
  { sl: 27, code: '0x0027', desc: 'Throw range invalid for R10 (Note 6)',  remarks: 'not a valid throw range' },
  { sl: 28, code: '0x0028', desc: 'Throw range invalid for R11 (Note 6)',  remarks: 'not a valid throw range' },
];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const queryInput    = $('queryInput');
const submitBtn     = $('submitBtn');
const loadingCard   = $('loadingCard');
const answerCard    = $('answerCard');
const answerBadge   = $('answerBadge');
const answerBadgeText = $('answerBadgeText');
const answerBody    = $('answerBody');
const metaLatency   = $('metaLatency');
const metaScore     = $('metaScore');
const chunksSection = $('chunksSection');
const chunksList    = $('chunksList');
const guardrailNotice = $('guardrailNotice');
const notfoundCard  = $('notfoundCard');
const notfoundMsg   = $('notfoundMsg');
const errorCard     = $('errorCard');
const errorMsg      = $('errorMsg');
const statusDot     = $('statusDot');
const statusLabel   = $('statusLabel');
const refTableBody  = $('refTableBody');

// ── Populate quick-reference table ────────────────────────────────────────────
function populateTable() {
  refTableBody.innerHTML = FAULT_CODES.map(row => `
    <tr data-code="${row.code}" onclick="fillQuery('${row.desc}')">
      <td>${row.sl}</td>
      <td>${row.code}</td>
      <td>${escHtml(row.desc)}</td>
      <td>${escHtml(row.remarks)}</td>
    </tr>
  `).join('');
}

function fillQuery(text) {
  queryInput.value = `What does "${text}" mean?`;
  queryInput.focus();
}

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const resp = await fetch('/health');
    const data = await resp.json();

    if (data.ready) {
      statusDot.className   = 'status-dot ok';
      statusLabel.textContent = 'System ready';
    } else if (data.index_exists && data.ollama !== 'ok') {
      statusDot.className   = 'status-dot warn';
      statusLabel.textContent = 'Ollama not running';
    } else if (!data.index_exists) {
      statusDot.className   = 'status-dot warn';
      statusLabel.textContent = 'Index not built';
    } else {
      statusDot.className   = 'status-dot warn';
      statusLabel.textContent = 'Partial ready';
    }

    if (data.model) {
      $('footerModel').textContent = data.model;
    }
  } catch {
    statusDot.className   = 'status-dot error';
    statusLabel.textContent = 'Server offline';
  }
}

// ── State management ──────────────────────────────────────────────────────────
function hideAll() {
  loadingCard.hidden   = true;
  answerCard.hidden    = true;
  notfoundCard.hidden  = true;
  errorCard.hidden     = true;
}

function showLoading() {
  hideAll();
  loadingCard.hidden = false;
  submitBtn.disabled = true;
}

function showAnswer(data) {
  hideAll();

  // Badge
  answerBadge.className    = 'answer-badge';
  answerBadgeText.textContent = 'Grounded answer';

  // Meta
  metaLatency.textContent = data.latency_ms ? `${data.latency_ms}ms` : '';
  metaScore.textContent   = data.top_score  ? `score ${data.top_score.toFixed(3)}` : '';

  // Answer text — highlight citations like [IRL Fault Codes, 0x0003]
  answerBody.innerHTML = formatAnswerText(data.answer);

  // Retrieved chunks
  if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
    chunksSection.hidden = false;
    chunksList.innerHTML = data.retrieved_chunks.map(c => `
      <div class="chunk-item">
        <span class="chunk-code">${escHtml(c.error_code || 'N/A')}</span>
        <span class="chunk-desc">${escHtml(c.error_description || '')}</span>
        <span class="chunk-rem">${escHtml(c.error_remarks || '')}</span>
      </div>
    `).join('');
  } else {
    chunksSection.hidden = true;
  }

  // Guardrail notice
  guardrailNotice.hidden = !data.guardrail_triggered;

  answerCard.hidden = false;
}

function showNotFound(message) {
  hideAll();
  notfoundMsg.textContent = message || 'This information is not available in the current documentation.';
  notfoundCard.hidden = false;
}

function showError(message) {
  hideAll();
  errorMsg.textContent = message || 'An unexpected error occurred. Check that the server is running.';
  errorCard.hidden = false;
}

// ── Text formatting ────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatAnswerText(text) {
  if (!text) return '';
  // Escape HTML first
  let safe = escHtml(text);
  // Highlight [Citation, Code] patterns
  safe = safe.replace(
    /\[([^\]]+)\]/g,
    '<span class="citation">[$1]</span>'
  );
  // Convert newlines to <br>
  safe = safe.replace(/\n/g, '<br>');
  return safe;
}

// ── Query submission ──────────────────────────────────────────────────────────
async function submitQuery() {
  const question = queryInput.value.trim();
  if (!question) {
    queryInput.focus();
    return;
  }

  showLoading();

  try {
    const resp = await fetch('/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || `Server error (HTTP ${resp.status})`);
      return;
    }

    // Route to correct display state
    if (!data.found) {
      showNotFound(data.answer);
    } else if (data.error && !data.answer) {
      showError(data.answer || 'Service error — see server logs.');
    } else {
      showAnswer(data);
    }

  } catch (err) {
    showError(`Cannot reach the server. Is it running on port 8000? (${err.message})`);
  } finally {
    submitBtn.disabled = false;
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
submitBtn.addEventListener('click', submitQuery);

queryInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    submitQuery();
  }
});

// Example chips
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.q;
    queryInput.focus();
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────
populateTable();
checkHealth();

// Re-check health every 30 seconds
setInterval(checkHealth, 30_000);
