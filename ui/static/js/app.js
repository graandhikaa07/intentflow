/**
 * IntentFlow Interactive Client-Side Application
 * 
 * Features:
 * - Real DistilBERT (66M) + QueryRouter pipeline execution
 * - IntentFlow -> Home / New Chat navigation from both sidebar and topbar
 * - Unique local chat title generation without external API calls
 * - Strict conversation isolation with unique IDs, message histories, and switching
 * - Telemetry details accordion, live multi-stage indicators, and session persistence
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Main Layout
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const chatViewport = document.getElementById('chat-viewport');
  const messagesContainer = document.getElementById('messages-container');
  const emptyState = document.getElementById('empty-state');
  const processingState = document.getElementById('processing-state');
  const processingText = document.getElementById('processing-text');
  const newChatBtn = document.getElementById('new-chat-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const sessionList = document.getElementById('session-list');

  // Navigation: Dual IntentFlow Home buttons
  const sidebarBrand = document.getElementById('sidebar-brand');
  const topbarBrand = document.getElementById('topbar-brand');

  // Sidebar Elements
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
  const closeSidebarBtn = document.getElementById('close-sidebar-btn');

  // Modal Elements
  const aboutModal = document.getElementById('about-modal');
  const openAboutBtn = document.getElementById('open-about-btn');
  const topbarAboutBtn = document.getElementById('topbar-about-btn');
  const closeAboutBtn = document.getElementById('close-about-btn');
  const modalGotItBtn = document.getElementById('modal-got-it-btn');

  // Storage Key
  const STORAGE_KEY = 'intentflow_conversations_v3';

  // Application State
  let conversations = loadConversationsFromStorage();
  let activeConversationId = null;
  let isSubmitting = false;

  // ----------------------------------------------------
  // Home Navigation
  // ----------------------------------------------------
  function navigateToHome() {
    activeConversationId = null;
    messagesContainer.innerHTML = '';
    emptyState.style.display = 'flex';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    if (intentSelector) {
      intentSelector.value = 'auto';
      intentSelector.classList.remove('override-active');
    }
    chatInput.focus();

    // Remove active highlight from all sidebar sessions
    document.querySelectorAll('.session-item-wrapper').forEach(el => el.classList.remove('active'));

    if (sidebar && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
    scrollToBottom();
  }

  // 1. Sidebar Brand click & keypress -> Home
  if (sidebarBrand) {
    sidebarBrand.addEventListener('click', (e) => {
      e.preventDefault();
      navigateToHome();
    });
    sidebarBrand.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        navigateToHome();
      }
    });
  }

  // 2. Topbar Brand click & keypress -> Home
  if (topbarBrand) {
    topbarBrand.addEventListener('click', (e) => {
      e.preventDefault();
      navigateToHome();
    });
    topbarBrand.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        navigateToHome();
      }
    });
  }

  // 3. New Chat & Clear Chat buttons -> Home
  if (newChatBtn) newChatBtn.addEventListener('click', navigateToHome);
  if (clearChatBtn) clearChatBtn.addEventListener('click', navigateToHome);

  // ----------------------------------------------------
  // Smart Local Title Generation (No External API calls)
  // ----------------------------------------------------
  function generateChatTitle(query, chatIndex = 1) {
    if (!query || typeof query !== 'string') {
      return `New Chat ${chatIndex}`;
    }

    const raw = query.trim();
    const lower = raw.toLowerCase();

    // 1. Math / Calculation queries
    if (lower.includes('compound interest')) return 'Compound Interest Calculation';
    if (lower.includes('square root')) {
      const numMatch = raw.match(/\b\d+(?:\.\d+)?\b/);
      return numMatch ? `Square Root of ${numMatch[0]}` : 'Square Root Calculation';
    }
    if (lower.includes('population variance') || lower.includes('variance')) return 'Population Variance Calculation';
    if (lower.includes('arithmetic mean') || (lower.includes('mean') && lower.includes('numbers'))) return 'Arithmetic Mean Calculation';
    if (lower.includes('median')) return 'Median Calculation';
    if (lower.includes('factorial')) {
      const numMatch = raw.match(/\b\d+\b/);
      return numMatch ? `Factorial of ${numMatch[0]}` : 'Factorial Calculation';
    }
    if (lower.includes('remainder') || lower.includes('divided by')) return 'Remainder Calculation';
    if (lower.includes('discount')) return 'Discount Calculation';

    // 2. Factual / Search queries
    const capMatch = raw.match(/capital\s+(?:city\s+)?of\s+([A-Za-z\s]+?)(?:\?|\.|$)/i);
    if (capMatch) {
      const country = capMatch[1].trim();
      return `Capital of ${country.charAt(0).toUpperCase() + country.slice(1)}`;
    }
    if (lower.includes('capital of')) {
      const parts = raw.split(/capital of/i);
      const name = parts[1].replace(/[?.!]/g, '').trim().slice(0, 22);
      if (name) return `Capital of ${name.charAt(0).toUpperCase() + name.slice(1)}`;
    }

    const whoMatch = raw.match(/who\s+(?:is|was|were|discovered|invented|wrote)\s+([A-Za-z\s]+?)(?:\?|\.|$)/i);
    if (whoMatch) {
      const person = whoMatch[1].trim().slice(0, 24);
      return `About ${person.charAt(0).toUpperCase() + person.slice(1)}`;
    }

    // 3. Rewriting / Tone adaptation
    if (lower.includes('friendly') && (lower.includes('tone') || lower.includes('rewrite') || lower.includes('professional'))) {
      return 'Friendly Tone Rewrite';
    }
    if (lower.includes('professional tone') || lower.includes('professional rewrite')) {
      return 'Professional Tone Rewrite';
    }
    if (lower.includes('rewrite') || lower.includes('rephrase')) {
      const toneMatch = raw.match(/in\s+a\s+([a-zA-Z\s]+?)\s+tone/i);
      if (toneMatch) {
        const tone = toneMatch[1].trim();
        return `${tone.charAt(0).toUpperCase() + tone.slice(1)} Tone Rewrite`;
      }
      return 'Sentence Rewriting';
    }
    if (lower.includes('creative story') || lower.includes('short story')) {
      const storyMatch = raw.match(/story\s+about\s+([a-zA-Z\s]+?)(?:\.|$)/i);
      return storyMatch ? `Story of ${storyMatch[1].trim().slice(0, 20)}` : 'Creative Story';
    }

    // 4. Structured Output / Schemas
    if (lower.includes('json') || lower.includes('yaml') || lower.includes('xml') || lower.includes('schema')) {
      const fmt = lower.includes('json') ? 'JSON' : (lower.includes('yaml') ? 'YAML' : (lower.includes('xml') ? 'XML' : 'Schema'));
      if (lower.includes('name') && lower.includes('age')) return `${fmt} with Name and Age`;
      if (lower.includes('product') || lower.includes('sku')) return `${fmt} Product Schema`;
      if (lower.includes('book') || lower.includes('author')) return `${fmt} Book Record`;
      if (lower.includes('array') || lower.includes('list')) return `${fmt} Array Format`;
      return `${fmt} Structured Output`;
    }

    // 5. Clean Fallback: Extract 2-5 prominent words
    const stripped = raw
      .replace(/^(can\s+you\s+(please\s+)?|could\s+you\s+(please\s+)?|please\s+|tell\s+me\s+(about\s+)?|i\s+need\s+to\s+|i\s+want\s+to\s+|give\s+me\s+)/i, '')
      .replace(/[^\w\s-]/g, ' ')
      .trim();

    const words = stripped.split(/\s+/).filter(w => w.length > 0);
    if (words.length >= 2) {
      const maxWords = Math.min(words.length, 5);
      return words.slice(0, maxWords).map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
    } else if (words.length === 1) {
      return `${words[0].charAt(0).toUpperCase() + words[0].slice(1).toLowerCase()} Query`;
    }

    // Unique progressive fallback
    return `New Chat ${chatIndex}`;
  }

  // ----------------------------------------------------
  // LocalStorage Persistence & Chat Isolation
  // ----------------------------------------------------
  function loadConversationsFromStorage() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) return parsed;
      }
    } catch (e) {
      console.warn('Failed to load conversations from localStorage:', e);
    }
    return [];
  }

  function saveConversationsToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.warn('Failed to save conversations to localStorage:', e);
    }
  }

  function renderSessionList() {
    if (!sessionList) return;

    if (conversations.length === 0) {
      sessionList.innerHTML = '<div class="session-empty">No saved conversations yet</div>';
      return;
    }

    sessionList.innerHTML = '';
    conversations.forEach((conv) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'session-item-wrapper' + (conv.id === activeConversationId ? ' active' : '');
      wrapper.setAttribute('data-id', conv.id);

      const titleBtn = document.createElement('button');
      titleBtn.className = 'session-item-btn';
      titleBtn.textContent = conv.title || 'Untitled Chat';
      titleBtn.title = conv.title || 'Untitled Chat';
      titleBtn.addEventListener('click', () => {
        loadConversation(conv.id);
      });

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'session-delete-btn';
      deleteBtn.title = 'Delete chat';
      deleteBtn.setAttribute('aria-label', `Delete ${conv.title}`);
      deleteBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConversation(conv.id);
      });

      wrapper.appendChild(titleBtn);
      wrapper.appendChild(deleteBtn);
      sessionList.appendChild(wrapper);
    });
  }

  function loadConversation(convId) {
    const conv = conversations.find(c => c.id === convId);
    if (!conv) return;

    activeConversationId = conv.id;
    messagesContainer.innerHTML = '';
    emptyState.style.display = 'none';

    // Render only this conversation's messages
    conv.messages.forEach((msg) => {
      if (msg.role === 'user') {
        renderUserMessage(msg.content);
      } else if (msg.role === 'assistant' && msg.data) {
        renderAssistantMessage(msg.data);
      }
    });

    // Update active highlight in sidebar
    document.querySelectorAll('.session-item-wrapper').forEach((el) => {
      if (el.getAttribute('data-id') === conv.id) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });

    scrollToBottom();
    if (sidebar && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  }

  function deleteConversation(convId) {
    const idx = conversations.findIndex(c => c.id === convId);
    if (idx === -1) return;

    conversations.splice(idx, 1);
    saveConversationsToStorage();

    if (activeConversationId === convId) {
      navigateToHome();
    }
    renderSessionList();
  }

  // ----------------------------------------------------
  // Input & Form Submission
  // ----------------------------------------------------
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSend();
  });

  // Clickable Example Prompts
  document.querySelectorAll('.example-card').forEach((card) => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt && !isSubmitting) {
        chatInput.value = prompt;
        handleSend();
      }
    });
  });

  // Mobile Sidebar Toggle
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', () => sidebar.classList.add('open'));
  }
  if (closeSidebarBtn) {
    closeSidebarBtn.addEventListener('click', () => sidebar.classList.remove('open'));
  }

  // Modal Controls
  function openModal() {
    aboutModal.style.display = 'flex';
  }
  function closeModal() {
    aboutModal.style.display = 'none';
  }

  if (openAboutBtn) openAboutBtn.addEventListener('click', openModal);
  if (topbarAboutBtn) topbarAboutBtn.addEventListener('click', openModal);
  if (closeAboutBtn) closeAboutBtn.addEventListener('click', closeModal);
  if (modalGotItBtn) modalGotItBtn.addEventListener('click', closeModal);

  aboutModal.addEventListener('click', (e) => {
    if (e.target === aboutModal) closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && aboutModal.style.display === 'flex') {
      closeModal();
    }
  });

  // ----------------------------------------------------
  const intentSelector = document.getElementById('intent-selector');
  if (intentSelector) {
    const updateIntentSelectorStyle = () => {
      if (intentSelector.value !== 'auto') {
        intentSelector.classList.add('override-active');
      } else {
        intentSelector.classList.remove('override-active');
      }
    };
    intentSelector.addEventListener('change', updateIntentSelectorStyle);
    updateIntentSelectorStyle();
  }

  // ----------------------------------------------------
  // Real Pipeline Execution
  // ----------------------------------------------------
  async function handleSend() {
    const query = chatInput.value.trim();
    if (!query || isSubmitting) return;

    // Capture selected intent before resetting selector
    const selectedIntent = intentSelector ? intentSelector.value : 'auto';
    const isExplicitIntent = selectedIntent && selectedIntent !== 'auto';
    const intentPayload = isExplicitIntent ? selectedIntent : null;

    isSubmitting = true;
    sendBtn.disabled = true;
    chatInput.disabled = true;

    // Hide empty state
    emptyState.style.display = 'none';

    // Check if this is a brand new conversation
    let currentConv = null;

    if (!activeConversationId) {
      const newId = 'chat_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
      const uniqueTitle = generateChatTitle(query, conversations.length + 1);

      currentConv = {
        id: newId,
        title: uniqueTitle,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: []
      };

      activeConversationId = newId;
      conversations.unshift(currentConv);
    } else {
      currentConv = conversations.find(c => c.id === activeConversationId);
      if (!currentConv) {
        const newId = 'chat_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
        const uniqueTitle = generateChatTitle(query, conversations.length + 1);
        currentConv = {
          id: newId,
          title: uniqueTitle,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: []
        };
        activeConversationId = newId;
        conversations.unshift(currentConv);
      }
    }

    // Append user message to active chat
    currentConv.messages.push({ role: 'user', content: query });
    currentConv.updatedAt = Date.now();
    saveConversationsToStorage();
    renderSessionList();

    // Render user message bubble in DOM
    renderUserMessage(query);

    // Clear input field
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show clean processing indicator
    processingState.style.display = 'flex';
    processingText.textContent = 'Generating response...';
    scrollToBottom();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, intent: intentPayload })
      });

      const result = await response.json();
      processingState.style.display = 'none';

      if (response.ok && result.status === 'success') {
        renderAssistantMessage(result);
        const routing = result.routing || {};
        currentConv.messages.push({
          role: 'assistant',
          query: query,
          response: result.response,
          intent: routing.intent,
          decision_source: routing.decision_source || (isExplicitIntent ? 'user_provided' : 'automatic'),
          confidence: routing.decision_source === 'automatic' ? routing.confidence : null,
          data: result
        });
        currentConv.updatedAt = Date.now();
        saveConversationsToStorage();
      } else {
        const errorMsg = result.error || 'Server returned an error.';
        renderErrorMessage(errorMsg);
        currentConv.messages.push({
          role: 'assistant',
          data: { response: errorMsg, routing: { intent: 'ERROR', handler_name: 'error' } }
        });
        saveConversationsToStorage();
      }
    } catch (err) {
      if (stageTimer1) clearTimeout(stageTimer1);
      if (stageTimer2) clearTimeout(stageTimer2);
      processingState.style.display = 'none';
      renderErrorMessage('Connection error. Please ensure the backend is running.');
      console.error(err);
    } finally {
      isSubmitting = false;
      sendBtn.disabled = false;
      chatInput.disabled = false;
      chatInput.focus();
      scrollToBottom();
    }
  }

  // ----------------------------------------------------
  // DOM Renderers
  // ----------------------------------------------------
  function renderUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.innerHTML = `
      <div class="user-bubble">${escapeHtml(text)}</div>
    `;
    messagesContainer.appendChild(msgDiv);
  }

  function renderAssistantMessage(data) {
    const routing = data.routing || {};
    // Log routing details strictly in browser console for developers/researchers
    console.log('[IntentFlow Pipeline Details]', routing);

    const intent = routing.intent || 'UNKNOWN';
    const intentLower = intent.toLowerCase();
    const confPct = routing.confidence_pct || '';
    const isManual = routing.decision_source === 'user_provided';
    const badgeText = confPct ? `${intent} · ${confPct}` : (isManual ? `${intent} · Manual` : intent);

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message';

    const formattedBody = formatMessageText(data.response);

    msgDiv.innerHTML = `
      <div class="assistant-wrapper">
        <div class="assistant-header">
          <div class="assistant-brand">
            <div class="assistant-avatar-badge">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
            </div>
            <span class="assistant-name">IntentFlow</span>
            <span class="intent-badge ${intentLower}">
              <span class="pill-dot"></span>
              ${badgeText}
            </span>
          </div>
          <button class="copy-response-btn" type="button" title="Copy response" aria-label="Copy response">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          </button>
        </div>

        <div class="assistant-body">${formattedBody}</div>
      </div>
    `;

    // Hook up copy response button
    const copyBtn = msgDiv.querySelector('.copy-response-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(data.response);
          const originalText = copyBtn.innerHTML;
          copyBtn.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span style="color: #10b981; font-weight: 500;">Copied!</span>
          `;
          setTimeout(() => {
            copyBtn.innerHTML = originalText;
          }, 2000);
        } catch (e) {
          console.warn('Copy failed:', e);
        }
      });
    }

    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  function renderErrorMessage(errorText) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message';
    msgDiv.innerHTML = `
      <div class="assistant-wrapper" style="border-color: rgba(239, 68, 68, 0.35);">
        <div class="assistant-header">
          <div class="assistant-brand">
            <div class="assistant-avatar-badge" style="background: rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.4); color: #f87171;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
            </div>
            <span class="assistant-name" style="color: #f87171;">Notice</span>
          </div>
        </div>
        <div class="assistant-body" style="color: #fca5a5;">
          ${escapeHtml(errorText)}
        </div>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  // Response text formatting
  function formatMessageText(text) {
    if (!text) return '';

    const trimmed = text.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        const parsed = JSON.parse(trimmed);
        return `<pre><code>${escapeHtml(JSON.stringify(parsed, null, 2))}</code></pre>`;
      } catch (e) {
        // Continue formatting
      }
    }

    let escaped = escapeHtml(text);

    // Code blocks ```code```
    escaped = escaped.replace(/```([a-zA-Z]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code `code`
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italics *text*
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Bullet points
    escaped = escaped.replace(/^[\s]*[-*]\s+(.*)$/gm, '<li>$1</li>');
    escaped = escaped.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Numbered lists
    escaped = escaped.replace(/^[\s]*(\d+)\.\s+(.*)$/gm, '<li>$2</li>');

    // Newlines to <br> unless inside <pre>
    const parts = escaped.split(/(<pre>[\s\S]*?<\/pre>)/);
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i].startsWith('<pre>')) {
        parts[i] = parts[i].replace(/\n/g, '<br>');
      }
    }

    return parts.join('');
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  // Initial load
  renderSessionList();
});
