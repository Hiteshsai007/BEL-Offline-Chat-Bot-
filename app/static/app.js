/**
 * BEL Offline AI Chatbot — Frontend Logic
 * Conversation-style interface. No external dependencies.
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const messagesEl    = $('messages');
const welcomeScreen = $('welcomeScreen');
const queryInput    = $('queryInput');
const sendBtn       = $('sendBtn');
const statusDot     = $('statusDot');
const statusDotMob  = $('statusDotMobile');
const statusLabel   = $('statusLabel');
const footerModel   = $('footerModel');
const sidebar       = $('sidebar');
const menuBtn       = $('menuBtn');
const sidebarToggle = $('sidebarToggle');
const newChatBtn    = $('newChatBtn');

// ── Conversation state ────────────────────────────────────────────────
let isProcessing = false;

// ── Utilities ─────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatAnswer(text) {
  if (!text) return '';
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

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

// ── Auto-resize textarea ──────────────────────────────────────────────
function autoResize() {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
}

queryInput.addEventListener('input', () => {
  autoResize();
  sendBtn.disabled = !queryInput.value.trim();
});

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const resp = await fetch('/health');
    const data = await resp.json();

    const setStatus = (cls, text) => {
      statusDot.className   = 'status-dot ' + cls;
      if (statusDotMob) statusDotMob.className = 'status-dot ' + cls;
      statusLabel.textContent = text;
    };

    if (data.ready) {
      setStatus('ok', 'System ready');
    } else if (data.index_exists && data.ollama !== 'ok') {
      setStatus('warn', 'Ollama not running');
    } else if (!data.index_exists) {
      setStatus('warn', 'Index not built');
    } else {
      setStatus('warn', 'Partial ready');
    }

    if (data.model) {
      footerModel.textContent = data.model;
    }
  } catch {
    statusDot.className   = 'status-dot error';
    if (statusDotMob) statusDotMob.className = 'status-dot error';
    statusLabel.textContent = 'Server offline';
  }
}

// ── Message rendering ─────────────────────────────────────────────────
function addUserMessage(text) {
  // Hide welcome screen on first message
  if (welcomeScreen) welcomeScreen.style.display = 'none';

  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-bubble">${escHtml(text)}</div>
    <div class="msg-avatar">U</div>
  `;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addThinkingBubble() {
  const row = document.createElement('div');
  row.className = 'msg-row ai thinking';
  row.id = 'thinkingRow';
  row.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
      </svg>
    </div>
    <div class="msg-bubble">
      <div class="thinking-dots">
        <span></span><span></span><span></span>
      </div>
      <span class="thinking-text">Searching knowledge base...</span>
    </div>
  `;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function removeThinkingBubble() {
  const el = $('thinkingRow');
  if (el) el.remove();
}

function addAIMessage(data) {
  removeThinkingBubble();

  const row = document.createElement('div');
  row.className = 'msg-row ai';

  let bubbleClass = 'msg-bubble';
  let content = '';

  if (!data.found) {
    // Not found
    bubbleClass += ' notfound-bubble';
    content = `<strong>⚠ Not found in documentation</strong><br>${escHtml(data.answer)}`;
  } else if (data.error && !data.answer) {
    // Error
    bubbleClass += ' error-bubble';
    content = `<strong>❌ Service error</strong><br>${escHtml(data.error || 'Unexpected error — check server logs.')}`;
  } else {
    // Normal answer
    content = formatAnswer(data.answer);

    // Source chunks
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      content += '<div class="msg-sources">';
      content += '<div class="msg-sources-label">📄 Retrieved Sources</div>';
      data.retrieved_chunks.forEach(c => {
        content += `<div class="source-item">
          <span class="source-code">${escHtml(c.error_code || 'N/A')}</span>
          <span class="source-desc">${escHtml(c.error_description || '')}</span>
          <span class="source-remarks">${escHtml(c.error_remarks || '')}</span>
        </div>`;
      });
      content += '</div>';
    }

    // Guardrail notice
    if (data.guardrail_triggered) {
      content += `<div class="msg-guardrail">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        Citation guardrail triggered — answer was regenerated for accuracy.
      </div>`;
    }

    // Meta info
    content += `<div class="msg-meta">`;
    if (data.latency_ms) content += `<span>⏱ ${data.latency_ms}ms</span>`;
    if (data.top_score)  content += `<span>📊 score ${data.top_score.toFixed(3)}</span>`;
    content += '</div>';
  }

  row.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
      </svg>
    </div>
    <div class="${bubbleClass}">${content}</div>
  `;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addErrorMessage(errMsg) {
  removeThinkingBubble();
  addAIMessage({
    found: true,
    answer: null,
    error: errMsg,
    retrieved_chunks: [],
    guardrail_triggered: false,
    latency_ms: 0,
    top_score: 0
  });
}

// ── Query submission ──────────────────────────────────────────────────
async function submitQuery() {
  const question = queryInput.value.trim();
  if (!question || isProcessing) return;

  isProcessing = true;
  sendBtn.disabled = true;

  // Show user message
  addUserMessage(question);
  queryInput.value = '';
  autoResize();

  // Show thinking indicator
  addThinkingBubble();

  try {
    const resp = await fetch('/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      addErrorMessage(data.detail || `Server error (HTTP ${resp.status})`);
      return;
    }

    addAIMessage(data);

  } catch (err) {
    addErrorMessage(`Cannot reach the server. Is it running on port 8000? (${err.message})`);
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
  }
}

// ── New Chat ──────────────────────────────────────────────────────────
function resetChat() {
  // Remove all messages except the welcome screen
  messagesEl.innerHTML = '';
  // Rebuild the welcome screen
  messagesEl.innerHTML = `
    <div class="welcome" id="welcomeScreen">
      <div class="welcome-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
        </svg>
      </div>
      <h1 class="welcome-title">BEL Fault Code Assistant</h1>
      <p class="welcome-sub">Ask me about any IRL fault code. I'll look it up in the knowledge base and give you a grounded answer — completely offline.</p>
      <div class="welcome-chips">
        <button class="welcome-chip" data-q="What does error code 0x0003 mean?">
          <span class="wchip-icon">⚡</span>
          <span class="wchip-text">What does 0x0003 mean?</span>
        </button>
        <button class="welcome-chip" data-q="What is a misfire error?">
          <span class="wchip-icon">🔥</span>
          <span class="wchip-text">What is a misfire error?</span>
        </button>
        <button class="welcome-chip" data-q="Explain error 0x0017">
          <span class="wchip-icon">🔧</span>
          <span class="wchip-text">Explain error 0x0017</span>
        </button>
        <button class="welcome-chip" data-q="What does throw range invalid mean?">
          <span class="wchip-icon">📡</span>
          <span class="wchip-text">What is throw range invalid?</span>
        </button>
      </div>
    </div>
  `;
  bindWelcomeChips();
  closeSidebar();
}

// ── Sidebar toggle (mobile) ──────────────────────────────────────────
let overlay = null;

function openSidebar() {
  sidebar.classList.add('open');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay show';
    overlay.addEventListener('click', closeSidebar);
    document.body.appendChild(overlay);
  } else {
    overlay.classList.add('show');
  }
}

function closeSidebar() {
  sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('show');
}

// ── Bind chip clicks ──────────────────────────────────────────────────
function bindWelcomeChips() {
  document.querySelectorAll('.welcome-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      queryInput.value = chip.dataset.q;
      sendBtn.disabled = false;
      submitQuery();
    });
  });
}

function bindSidebarChips() {
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      queryInput.value = chip.dataset.q;
      sendBtn.disabled = false;
      closeSidebar();
      submitQuery();
    });
  });
}

// ── Event listeners ───────────────────────────────────────────────────
sendBtn.addEventListener('click', submitQuery);

queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
});

if (menuBtn)       menuBtn.addEventListener('click', openSidebar);
if (sidebarToggle) sidebarToggle.addEventListener('click', closeSidebar);
if (newChatBtn)    newChatBtn.addEventListener('click', resetChat);

// ── Init ──────────────────────────────────────────────────────────────
bindWelcomeChips();
bindSidebarChips();
checkHealth();
setInterval(checkHealth, 30_000);
