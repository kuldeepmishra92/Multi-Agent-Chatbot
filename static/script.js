// --- State Management ---
let sessionId = Math.random().toString(36).substring(2, 10);
let messages = [];
let isWelcomeMode = true;
let isDarkTheme = localStorage.getItem('theme') !== 'light';

// --- Selectors ---
const body = document.body;
const themeToggle = document.getElementById('theme-toggle');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const welcomeScreen = document.getElementById('welcome-screen');
const inputWrapper = document.getElementById('input-container-wrapper');
const newChatBtn = document.getElementById('new-chat-btn');
const statMessages = document.getElementById('stat-messages');
const statChunks = document.getElementById('stat-chunks');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const fileList = document.getElementById('file-list');
const downloadBtn = document.getElementById('download-chat-btn');

// --- Initialization ---
function init() {
    // Set initial theme
    if (!isDarkTheme) {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.querySelector('i').setAttribute('data-lucide', 'sun');
    }
    lucide.createIcons();
    updateStats();
    
    // Configure Marked options
    if (window.marked) {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
    }
}

// --- Theme Management ---
themeToggle.addEventListener('click', () => {
    isDarkTheme = !isDarkTheme;
    body.classList.toggle('dark-theme');
    body.classList.toggle('light-theme');
    
    const icon = isDarkTheme ? 'moon' : 'sun';
    themeToggle.querySelector('i').setAttribute('data-lucide', icon);
    localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
    lucide.createIcons();
});

// --- UI Logic ---
function handleWelcomeTransition() {
    if (isWelcomeMode) {
        isWelcomeMode = false;
        welcomeScreen.style.opacity = '0';
        welcomeScreen.style.visibility = 'hidden';
        inputWrapper.classList.remove('welcome-mode');
    }
}

userInput.addEventListener('input', () => {
    // Auto-resize textarea
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
    
    // Enable/disable send button
    sendBtn.disabled = userInput.value.trim() === '';
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

newChatBtn.addEventListener('click', () => {
    sessionId = Math.random().toString(36).substring(2, 10);
    messages = [];
    chatMessages.innerHTML = '';
    isWelcomeMode = true;
    welcomeScreen.style.opacity = '1';
    welcomeScreen.style.visibility = 'visible';
    inputWrapper.classList.add('welcome-mode');
    statMessages.textContent = '0';
    downloadBtn.disabled = true;
});

// --- Chat Logic ---
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    handleWelcomeTransition();
    
    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Add User Message
    addMessage('user', text);
    
    // Add Assistant Typing Indicator
    const typingId = 'typing-' + Date.now();
    addTypingIndicator(typingId);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        // Remove typing indicator and prepare for content
        removeTypingIndicator(typingId);
        const assistantMsgId = 'msg-' + Date.now();
        const contentArea = initAssistantMessage(assistantMsgId);
        
        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullMarkdown = "";
        let metadata = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            
            // Look for metadata line (METADATA:{"agent":"...", "latency":...})
            if (chunk.startsWith('METADATA:')) {
                const parts = chunk.split('\n');
                metadata = JSON.parse(parts[0].replace('METADATA:', ''));
                if (parts[1]) {
                    fullMarkdown += parts[1];
                }
                // Add agent badge if metadata exists
                if (metadata) {
                    addAgentBadge(assistantMsgId, metadata.agent, metadata.latency);
                }
            } else {
                fullMarkdown += chunk;
            }
            
            // Pre-process raw stars:
            // 1. Fix "** word" (space after opening **) → "**word" — LLM sometimes generates this
            // 2. Fix **Hello!**nice → **Hello!** nice (no space after closing **)
            const processedMarkdown = fullMarkdown
                .replace(/\*\* /g, '**')
                .replace(/\*\*([^*]+)\*\*(?!\s|$)/g, '**$1** ');
            
            // Render markdown incrementally
            const html = window.marked ? marked.parse(processedMarkdown) : processedMarkdown;
            contentArea.innerHTML = html;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        messages.push({ role: 'assistant', content: fullMarkdown, metadata });
        statMessages.textContent = messages.length / 2; // Rough estimate
        downloadBtn.disabled = false;

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addMessage('assistant', 'Sorry, I encountered an error. Please check the logs.');
    }
}

function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const avatarIcon = role === 'user' ? 'user' : 'bot';
    const avatar = role === 'user' ? 'You' : 'Kuldeep AI';

    const processedContent = content
        .replace(/\*\* /g, '**')
        .replace(/\*\*([^*]+)\*\*(?!\s|$)/g, '**$1** ');
    msgDiv.innerHTML = `
        <div class="avatar">${role === 'user' ? 'You' : '<img src="/static/ai-avatar.png" alt="Kuldeep AI">'}</div>
        <div class="message-content">
            <div class="bubble">${role === 'user' ? processedContent : (window.marked ? marked.parse(processedContent) : processedContent)}</div>
            <div class="message-meta">
                ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
        </div>
    `;
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (role === 'user') messages.push({ role, content });
}

function addTypingIndicator(id) {
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="avatar"><img src="/static/ai-avatar.png" alt="Kuldeep AI"></div>
        <div class="message-content">
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function initAssistantMessage(id) {
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = 'message assistant';
    msgDiv.innerHTML = `
        <div class="avatar"><img src="/static/ai-avatar.png" alt="Kuldeep AI"></div>
        <div class="message-content">
            <div id="${id}-badge-area"></div>
            <div class="bubble assistant-bubble"></div>
            <div class="message-meta">
                ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    return msgDiv.querySelector('.assistant-bubble');
}

function addAgentBadge(parentId, agentName, latency) {
    const area = document.getElementById(`${parentId}-badge-area`);
    if (!area) return;
    
    const colors = {
        'Math Agent': 'badge-math',
        'RAG Agent': 'badge-rag',
        'Memory Agent': 'badge-memory',
        'General Agent': 'badge-general',
        'Web Search Agent': 'badge-search'
    };
    
    const colorClass = colors[agentName] || '';
    area.innerHTML = `
        <div class="agent-badge-inline ${colorClass}">
            ${agentName} · ${latency}s
        </div>
    `;
}

// --- Stats Logic ---
async function updateStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        statChunks.textContent = data.total_chunks;
    } catch (e) { console.error('Stats error:', e); }
}

// --- File Upload Logic ---
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => handleFiles(fileInput.files));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

async function handleFiles(files) {
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let f of files) {
        if (f.type !== 'application/pdf') continue;
        formData.append('files', f);
        
        // Add to UI list
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `<i data-lucide="file-text"></i> ${f.name}`;
        fileList.appendChild(item);
    }
    lucide.createIcons();
    
    // Show spinner in dropzone
    const originalText = dropZone.querySelector('span').textContent;
    dropZone.querySelector('span').textContent = 'Indexing...';
    
    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        
        updateStats();
        dropZone.querySelector('span').textContent = '✅ Success';
        setTimeout(() => dropZone.querySelector('span').textContent = originalText, 3000);
    } catch (e) {
        console.error('Upload error:', e);
        dropZone.querySelector('span').textContent = '❌ Failed';
        setTimeout(() => dropZone.querySelector('span').textContent = originalText, 3000);
    }
}

// --- Download Logic ---
downloadBtn.addEventListener('click', () => {
    let md = `# Chat Session ${sessionId}\n\n`;
    messages.forEach(m => {
        const role = m.role === 'user' ? 'You' : (m.metadata?.agent || 'AI');
        md += `### ${role}\n${m.content}\n\n---\n\n`;
    });
    
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_${sessionId}.md`;
    a.click();
});

// --- Run ---
init();
