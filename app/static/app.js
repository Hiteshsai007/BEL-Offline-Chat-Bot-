/**
 * BEL Offline AI Chatbot Client Script
 * - Theme Switcher (Light / Dark)
 * - Animated Loading Symbol during LLM inference
 * - Elapsed Response Time Display
 * - Dynamic Ollama Model Selector
 */

'use strict';

const $ = id => document.getElementById(id);

// --- Element References ---
const sidebar = $('sidebar');
const sidebarCollapseBtn = $('sidebarCollapseBtn');
const sidebarExpandBtn = $('sidebarExpandBtn');

const newChatBtn = $('newChatBtn');
const historySearchToggle = $('historySearchToggle');
const searchWrap = $('searchWrap');
const historySearchInput = $('historySearchInput');
const historyList = $('historyList');
const viewAllBtn = $('viewAllBtn');
const viewAllText = $('viewAllText');

const ctxPopup = $('ctxPopup');
const ctxRename = $('ctxRename');
const ctxDelete = $('ctxDelete');

const heroSection = $('heroSection');
const messagesContainer = $('messagesContainer');
const mainInput = $('mainInput');
const sendBtn = $('sendBtn');
const fourCardsGrid = $('fourCardsGrid');

const statusDot = $('statusDot');
const statusLabel = $('statusLabel');
const footerModel = $('footerModel');
const modelSelect = $('modelSelect');
const themeToggleBtn = $('themeToggleBtn');
const themeIcon = $('themeIcon');

// --- Storage Store ---
const Storage = (() => {
  const KEY = 'bel_offline_chats_v5';

  function load() {
    try {
      const d = localStorage.getItem(KEY);
      return d ? JSON.parse(d) : [];
    } catch {
      return [];
    }
  }

  function save(list) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list));
    } catch {}
  }

  return {
    getAll() {
      return load().sort((a, b) => b.updatedAt - a.updatedAt);
    },
    get(id) {
      return load().find(s => s.id === id) || null;
    },
    create(id) {
      const list = load();
      const s = { id, title: 'New Chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
      list.push(s);
      save(list);
      return s;
    },
    addMsg(id, msg) {
      const list = load();
      const s = list.find(x => x.id === id);
      if (!s) return;
      s.messages.push(msg);
      s.updatedAt = Date.now();
      if (s.title === 'New Chat' && msg.role === 'user') {
        const text = String(msg.content).trim();
        s.title = text.length > 28 ? text.substring(0, 28) + '...' : text;
      }
      save(list);
    },
    delete(id) {
      save(load().filter(x => x.id !== id));
    },
    rename(id, title) {
      const list = load();
      const s = list.find(x => x.id === id);
      if (s) {
        s.title = title;
        save(list);
      }
    }
  };
})();

// --- Application State ---
let currentSessionId = null;
let activeCtxId = null;
let isViewAll = false;
let isQuerying = false;

function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const diffDays = Math.floor((now - d) / 86400000);
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (diffDays === 0) return 'Today, ' + time;
  if (diffDays === 1) return 'Yesterday, ' + time;
  if (diffDays < 7) {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    return days[d.getDay()] + ', ' + time;
  }
  return d.toLocaleString('default', { month: 'short' }) + ' ' + d.getDate() + ', ' + time;
}

// --- Theme Manager (Light / Dark Mode) ---
function initTheme() {
  const savedTheme = localStorage.getItem('bel_theme') || 'dark';
  if (savedTheme === 'light') {
    document.body.classList.add('light-mode');
    renderMoonIcon();
  } else {
    document.body.classList.remove('light-mode');
    renderSunIcon();
  }
}

function toggleTheme() {
  const isLight = document.body.classList.toggle('light-mode');
  localStorage.setItem('bel_theme', isLight ? 'light' : 'dark');
  if (isLight) renderMoonIcon();
  else renderSunIcon();
}

function renderSunIcon() {
  if (!themeIcon) return;
  themeIcon.innerHTML = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;
}

function renderMoonIcon() {
  if (!themeIcon) return;
  themeIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`;
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', toggleTheme);
}

// --- History List Rendering ---
function renderHistoryList() {
  const query = historySearchInput ? historySearchInput.value.trim().toLowerCase() : '';
  let all = Storage.getAll();

  if (query) {
    all = all.filter(s => s.title.toLowerCase().includes(query));
  } else if (!isViewAll) {
    all = all.slice(0, 10);
  }

  historyList.innerHTML = '';

  if (all.length === 0) {
    historyList.innerHTML = '<div style="padding:16px;text-align:center;font-size:12px;color:var(--text-dim);">No chats yet</div>';
    return;
  }

  all.forEach(s => {
    const item = document.createElement('div');
    const isActive = (s.id === currentSessionId);
    item.className = 'history-card-item' + (isActive ? ' active' : '');
    item.dataset.id = s.id;

    item.innerHTML = `
      <div class="item-content">
        <div class="item-title">${esc(s.title)}</div>
        <div class="item-time">${formatTime(s.updatedAt)}</div>
      </div>
      <button class="item-menu-btn" title="Options">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>
        </svg>
      </button>
    `;

    item.addEventListener('click', e => {
      if (e.target.closest('.item-menu-btn')) return;
      loadSession(s.id);
    });

    const trigger = item.querySelector('.item-menu-btn');
    trigger.addEventListener('click', e => {
      e.stopPropagation();
      openCtxMenu(s.id, trigger);
    });

    historyList.appendChild(item);
  });
}

// --- Context Menu Popup ---
function openCtxMenu(id, anchor) {
  activeCtxId = id;
  const rect = anchor.getBoundingClientRect();
  ctxPopup.style.top = Math.min(rect.bottom + 4, window.innerHeight - 80) + 'px';
  ctxPopup.style.left = Math.min(rect.left, window.innerWidth - 150) + 'px';
  ctxPopup.classList.add('show');
}

function closeCtxMenu() {
  ctxPopup.classList.remove('show');
  activeCtxId = null;
}

document.addEventListener('click', e => {
  if (!ctxPopup.contains(e.target) && !e.target.closest('.item-menu-btn')) {
    closeCtxMenu();
  }
});

ctxRename.addEventListener('click', () => {
  if (!activeCtxId) return;
  const targetId = activeCtxId;
  closeCtxMenu();

  const item = historyList.querySelector(`[data-id="${targetId}"]`);
  if (!item) return;

  const titleEl = item.querySelector('.item-title');
  const currentS = Storage.get(targetId);
  if (!titleEl || !currentS) return;

  const inp = document.createElement('input');
  inp.type = 'text';
  inp.value = currentS.title;
  inp.style.cssText = 'width:100%;background:var(--bg-page);border:1px solid var(--accent-blue);color:var(--text-white);font-size:12px;padding:2px 4px;border-radius:4px;outline:none;';

  titleEl.replaceWith(inp);
  inp.focus();
  inp.select();

  function commit() {
    const val = inp.value.trim() || currentS.title;
    Storage.rename(targetId, val);
    renderHistoryList();
  }

  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    if (e.key === 'Escape') renderHistoryList();
  });
});

ctxDelete.addEventListener('click', () => {
  if (!activeCtxId) return;
  const targetId = activeCtxId;
  closeCtxMenu();

  const wasActive = (targetId === currentSessionId);
  Storage.delete(targetId);

  if (wasActive) {
    const remaining = Storage.getAll();
    if (remaining.length > 0) {
      loadSession(remaining[0].id);
    } else {
      startNewChat();
    }
  } else {
    renderHistoryList();
  }
});

// --- Session Handling ---
function loadSession(id) {
  const s = Storage.get(id);
  if (!s) return;

  currentSessionId = id;
  messagesContainer.innerHTML = '';

  if (s.messages.length === 0) {
    heroSection.style.display = 'flex';
    messagesContainer.style.display = 'none';
    fourCardsGrid.style.display = 'grid';
  } else {
    heroSection.style.display = 'none';
    messagesContainer.style.display = 'flex';
    fourCardsGrid.style.display = 'none';

    s.messages.forEach(m => {
      if (m.role === 'user') renderUserMessage(m.content);
      else if (m.role === 'ai') renderAIMessage(m.data);
    });
  }

  renderHistoryList();
}

function startNewChat() {
  const id = generateUUID();
  Storage.create(id);
  loadSession(id);
  mainInput.value = '';
  sendBtn.disabled = true;
}

// --- Messages UI ---
function renderUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="bubble">${esc(text)}</div>
    <div class="msg-avatar-badge">U</div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
}

// Show Loading Symbol Bubble while waiting for LLM response
function showLoadingBubble() {
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.id = 'loadingBubbleRow';
  row.innerHTML = `
    <div class="msg-avatar-badge">AI</div>
    <div class="thinking-bubble">
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
      <span class="thinking-text">Searching knowledge base & generating answer...</span>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
}

function removeLoadingBubble() {
  const el = $('loadingBubbleRow');
  if (el) el.remove();
}

function renderAIMessage(data) {
  const row = document.createElement('div');
  row.className = 'msg-row ai';

  let body = '';
  if (data.error && !data.answer) {
    body = `<strong>Service error</strong><br/>${esc(data.error)}`;
  } else if (!data.found) {
    body = `<strong>Not found in documentation</strong><br/>${esc(data.answer)}`;
  } else {
    let formatted = esc(data.answer || '').replace(/\[([^\]]+)\]/g, '<span class="tag-citation">[$1]</span>').replace(/\n/g, '<br/>');
    body = formatted;

    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      body += `<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border-card);"><div style="font-size:11px;font-weight:700;color:var(--text-dim);margin-bottom:6px;text-transform:uppercase;">Retrieved Sources</div>`;
      data.retrieved_chunks.forEach(c => {
        if (c.error_code) {
          body += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;"><span style="font-family:monospace;color:var(--accent-blue);font-weight:600;">${esc(c.error_code)}</span> <span>${esc(c.error_description || '')}</span></div>`;
        }
      });
      body += `</div>`;
    }
  }

  // Response Time Badge at bottom of AI answer
  if (data.latency_ms) {
    const secStr = data.latency_ms >= 1000 ? (data.latency_ms / 1000).toFixed(2) + 's' : data.latency_ms + 'ms';
    body += `<div class="response-time-badge">⏱ Response time: ${secStr}</div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar-badge">AI</div>
    <div class="bubble">${body}</div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
}

function scrollToBottom() {
  const viewport = $('viewport');
  viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
}

// --- Query Submit Handler ---
async function submitQuery() {
  const q = mainInput.value.trim();
  if (!q || isQuerying) return;

  isQuerying = true;
  sendBtn.disabled = true;

  heroSection.style.display = 'none';
  messagesContainer.style.display = 'flex';
  fourCardsGrid.style.display = 'none';

  renderUserMessage(q);
  Storage.addMsg(currentSessionId, { role: 'user', content: q });
  renderHistoryList();

  mainInput.value = '';

  // Show loading symbol
  showLoadingBubble();

  const tStart = Date.now();

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 180000);
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: currentSessionId }),
      signal: controller.signal
    });
    clearTimeout(timer);

    const elapsed = Date.now() - tStart;
    let data;

    if (!res.ok) {
      let detail = `Server returned ${res.status}`;
      try {
        const payload = await res.json();
        detail = payload.detail || payload.error || detail;
      } catch {
        detail = `${detail} with a non-JSON body.`;
      }
      const errObj = { found: false, answer: null, error: detail, retrieved_chunks: [], latency_ms: elapsed };
      removeLoadingBubble();
      renderAIMessage(errObj);
      Storage.addMsg(currentSessionId, { role: 'ai', data: errObj });
      return;
    }

    try {
      data = await res.json();
    } catch {
      data = { found: false, answer: null, error: 'Server returned a non-JSON response.', retrieved_chunks: [], latency_ms: elapsed };
    }

    removeLoadingBubble();

    if (!data.latency_ms) {
      data.latency_ms = elapsed;
    }

    renderAIMessage(data);
    Storage.addMsg(currentSessionId, { role: 'ai', data: data });
  } catch (err) {
    const elapsed = Date.now() - tStart;
    removeLoadingBubble();
    const errObj = { found: false, answer: null, error: 'Connection failed: ' + err.message, retrieved_chunks: [], latency_ms: elapsed };
    renderAIMessage(errObj);
    Storage.addMsg(currentSessionId, { role: 'ai', data: errObj });
  } finally {
    isQuerying = false;
    sendBtn.disabled = false;
    renderHistoryList();
  }
}

// --- Ollama Model Selector Handling ---
async function loadAvailableModels() {
  if (!modelSelect) return;
  try {
    const res = await fetch('/models');
    const data = await res.json();
    const current = data.current || 'qwen2.5:3b';
    const available = data.available || [current];

    modelSelect.innerHTML = '';
    available.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      if (m === current) opt.selected = true;
      modelSelect.appendChild(opt);
    });

    if (footerModel) footerModel.textContent = current;
  } catch (e) {
    console.warn('Failed to load available models:', e);
  }
}

if (modelSelect) {
  modelSelect.addEventListener('change', async () => {
    const selectedModel = modelSelect.value;
    try {
      const res = await fetch('/model/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        if (footerModel) footerModel.textContent = data.current_model;
      }
    } catch (e) {
      alert('Failed to switch model: ' + e.message);
    }
  });
}

// --- Event Listeners ---
mainInput.addEventListener('input', () => {
  mainInput.style.height = 'auto';
  mainInput.style.height = Math.min(mainInput.scrollHeight, 120) + 'px';
  sendBtn.disabled = !mainInput.value.trim();
});

mainInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
});

sendBtn.addEventListener('click', submitQuery);
newChatBtn.addEventListener('click', startNewChat);

sidebarCollapseBtn.addEventListener('click', () => {
  sidebar.classList.add('collapsed');
});

sidebarExpandBtn.addEventListener('click', () => {
  sidebar.classList.remove('collapsed');
});

historySearchToggle.addEventListener('click', () => {
  searchWrap.classList.toggle('open');
  if (searchWrap.classList.contains('open')) {
    historySearchInput.focus();
  } else {
    historySearchInput.value = '';
    renderHistoryList();
  }
});

historySearchInput.addEventListener('input', renderHistoryList);

viewAllBtn.addEventListener('click', () => {
  isViewAll = !isViewAll;
  viewAllText.textContent = isViewAll ? 'Show recent history' : 'View all history';
  renderHistoryList();
});

document.querySelectorAll('.grid-card, .quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    mainInput.value = btn.dataset.q;
    sendBtn.disabled = false;
    submitQuery();
  });
});

// --- Health Check ---
async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    if (d.ready) {
      statusDot.className = 'status-dot-ok';
      statusLabel.textContent = 'System ready';
    } else {
      statusDot.className = 'status-dot-ok warn';
      statusLabel.textContent = 'Starting up...';
    }
    if (d.model) {
      if (footerModel) footerModel.textContent = d.model;
      if (modelSelect && modelSelect.value !== d.model) {
        modelSelect.value = d.model;
      }
    }
  } catch {
    statusDot.className = 'status-dot-ok error';
    statusLabel.textContent = 'Server offline';
  }
}

// --- Settings Modal Handler ---
const settingsBtn = $('settingsBtn');
const settingsModalBackdrop = $('settingsModalBackdrop');
const closeSettingsModal = $('closeSettingsModal');
const btnSettingsDone = $('btnSettingsDone');
const modalThemeSelect = $('modalThemeSelect');
const modalModelSelect = $('modalModelSelect');
const btnClearAllHistory = $('btnClearAllHistory');

function openSettingsModal() {
  if (!settingsModalBackdrop) return;
  
  // Sync theme select state
  const currentTheme = localStorage.getItem('bel_theme') || 'dark';
  if (modalThemeSelect) modalThemeSelect.value = currentTheme;

  // Sync model dropdown options
  if (modalModelSelect && modelSelect) {
    modalModelSelect.innerHTML = modelSelect.innerHTML;
    modalModelSelect.value = modelSelect.value;
  }

  settingsModalBackdrop.classList.add('open');
}

function closeSettingsModalFunc() {
  if (!settingsModalBackdrop) return;
  settingsModalBackdrop.classList.remove('open');
}

if (settingsBtn) {
  settingsBtn.addEventListener('click', openSettingsModal);
}

if (closeSettingsModal) {
  closeSettingsModal.addEventListener('click', closeSettingsModalFunc);
}

if (btnSettingsDone) {
  btnSettingsDone.addEventListener('click', closeSettingsModalFunc);
}

if (settingsModalBackdrop) {
  settingsModalBackdrop.addEventListener('click', e => {
    if (e.target === settingsModalBackdrop) closeSettingsModalFunc();
  });
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && settingsModalBackdrop && settingsModalBackdrop.classList.contains('open')) {
    closeSettingsModalFunc();
  }
});

if (modalThemeSelect) {
  modalThemeSelect.addEventListener('change', () => {
    const val = modalThemeSelect.value;
    if (val === 'light' && !document.body.classList.contains('light-mode')) {
      toggleTheme();
    } else if (val === 'dark' && document.body.classList.contains('light-mode')) {
      toggleTheme();
    }
  });
}

if (modalModelSelect) {
  modalModelSelect.addEventListener('change', () => {
    if (modelSelect) {
      modelSelect.value = modalModelSelect.value;
      modelSelect.dispatchEvent(new Event('change'));
    }
  });
}

if (btnClearAllHistory) {
  btnClearAllHistory.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear all local chat history?')) {
      localStorage.removeItem('bel_offline_chats_v5');
      startNewChat();
      renderHistoryList();
      closeSettingsModalFunc();
    }
  });
}

// --- Init ---
(function init() {
  initTheme();
  const chats = Storage.getAll();
  if (chats.length > 0) {
    loadSession(chats[0].id);
  } else {
    startNewChat();
  }
  loadAvailableModels();
  checkHealth();
  setInterval(checkHealth, 30000);
})();

