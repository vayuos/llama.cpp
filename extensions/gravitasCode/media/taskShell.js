// @ts-nocheck
/* eslint-disable no-undef */

/**
 * Gravitas Chat Infrastructure - Frontend Controller
 * Manages high-fidelity message rendering and lifecycle sync.
 */
class ChatController {
    constructor() {
        console.log('Gravitas Chat: Initializing ChatController...');
        this.debugBuffer = [];
        
        try {
            this.currentStatusEl = null;
            this.vscode = acquireVsCodeApi();
            this.taskFeed = document.getElementById('taskFeed');
            this.commandInput = document.getElementById('commandInput');
            this.welcomeScreen = document.getElementById('welcomeScreen');
            this.submitBtn = document.getElementById('submitBtn');
            this.debugOverlay = document.getElementById('debugOverlay');
            this.coderTelemetry = document.getElementById('coderTelemetry');
            this.reviewerTelemetry = document.getElementById('reviewerTelemetry');
            
            if (!this.taskFeed || !this.commandInput) {
                console.error('Gravitas Chat: DOM elements missing!', { feed: !!this.taskFeed, input: !!this.commandInput });
                throw new Error('Required DOM elements (taskFeed or commandInput) not found.');
            }

            this.activeTaskId = null;
            this.userHasScrolledUp = false;

            this.initEventListeners();
            this.reportReady();
            this.debugLog('System: Internal Boot Sequence Complete.');
        } catch (error) {
            this.reportCrash(error);
        }
    }

    debugLog(msg) {
        const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        this.debugBuffer.push(`[${timestamp}] ${msg}`);
        if (this.debugBuffer.length > 5) this.debugBuffer.shift();
        
        if (this.debugOverlay) {
            this.debugOverlay.innerHTML = this.debugBuffer.join('<br>');
            // Uncomment next line to show overlay for debugging
            // this.debugOverlay.style.display = 'block';
        }
    }

    reportReady() {
        console.log('Gravitas Chat: Signaling ready to Extension Host...');
        this.debugLog('TX: ready');
        this.vscode.postMessage({ type: 'ready' });
    }

    reportCrash(error) {
        console.error('Gravitas Chat: Critical Initialization Error:', error);
        this.debugLog(`ERR: ${error.message}`);
        if (this.vscode) {
            this.vscode.postMessage({ 
                type: 'error', 
                message: error.message, 
                stack: error.stack 
            });
        }
    }

    initEventListeners() {
        console.log('Gravitas Chat: Binding event listeners...');
        
        window.onerror = (message, source, lineno, colno, error) => {
            this.reportCrash(error || new Error(message));
        };

        window.addEventListener('message', event => {
            const message = event.data;
            this.debugLog(`RX: ${message.type}`);
            
            switch (message.type) {
                case 'loadSnapshot':
                    this.renderSnapshot(message.task);
                    break;
                case 'event':
                    this.handleLiveEvent(message.taskId, message.event);
                    break;
                case 'updateTask':
                    this.updateTaskUI(message.task);
                    break;
                case 'reset':
                    this.resetUI();
                    break;
                case 'focus':
                    this.focusInput();
                    break;
                case 'telemetry':
                    this.updateGlobalTelemetry(message.coder, message.reviewer, message.rag);
                    break;
            }
        });

        this.taskFeed.addEventListener('scroll', () => {
            const scrollPos = this.taskFeed.scrollTop + this.taskFeed.clientHeight;
            this.userHasScrolledUp = scrollPos < this.taskFeed.scrollHeight - 50;
        });

        this.commandInput.addEventListener('input', () => {
            this.commandInput.style.height = 'auto';
            this.commandInput.style.height = (this.commandInput.scrollHeight) + 'px';
        });

        this.commandInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.submitPrompt();
            }
        });

        if (this.submitBtn) {
            this.submitBtn.addEventListener('click', () => this.submitPrompt());
        }
    }

    submitPrompt() {
        const text = this.commandInput.value.trim();
        if (text) {
            console.log('Gravitas Chat: Submitting prompt...');
            this.debugLog(`TX: submitPrompt (${text.substring(0, 10)}...)`);
            this.commandInput.disabled = true;
            this.vscode.postMessage({ type: 'submitPrompt', text: text });
            this.commandInput.value = '';
            this.commandInput.style.height = 'auto'; // Reset height
            setTimeout(() => {
                this.commandInput.disabled = false;
                this.commandInput.focus();
            }, 500);
        }
    }

    autoScroll() {
        if (!this.userHasScrolledUp) {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
    }

    hideWelcome() {
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
    }

    focusInput() {
        this.commandInput.focus();
    }

    resetUI() {
        this.activeTaskId = null;
        this.taskFeed.innerHTML = '';
    }

    focusInput() {
        this.commandInput.focus();
    }

    renderSnapshot(task) {
        if (!task) return;
        this.hideWelcome();
        this.activeTaskId = task.id;

        // Clear existing shell if any
        let shell = this.getTaskShell(task.id);
        if (shell) shell.remove();

        shell = this.createTaskShell(task);
        this.taskFeed.appendChild(shell);
        
        // Cache current controls
        this.currentStatusEl = shell.querySelector('#task-status');
        this.currentStopBtn = shell.querySelector('#stop-task-btn');
        
        if (this.currentStopBtn) {
            this.currentStopBtn.addEventListener('click', () => {
                this.vscode.postMessage({ type: 'abortTask', taskId: task.id });
                this.currentStopBtn.classList.add('hidden');
            });
        }

        // Render all existing events
        task.attempts.forEach(attempt => {
            attempt.phases.forEach(phase => {
                phase.events.forEach(event => {
                    this.renderEvent(task.id, event);
                });
            });
        });

        this.updateTaskUI(task); // Apply final status
        this.autoScroll();
    }

    resetUI() {
        this.taskFeed.innerHTML = '';
        if (this.welcomeScreen) {
            this.taskFeed.appendChild(this.welcomeScreen);
            this.welcomeScreen.style.display = 'flex';
        }
        this.activeTaskId = null;
    }

    hideWelcome() {
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
    }

    handleLiveEvent(taskId, event) {
        if (taskId !== this.activeTaskId) return;
        this.hideWelcome();
        this.renderEvent(taskId, event);
        this.autoScroll();
    }

    createTaskShell(task) {
        const shell = document.createElement('div');
        shell.className = 'agent-shell';
        shell.id = `task-${task.id}`;
        shell.innerHTML = `
            <div class="agent-header">
                <div class="agent-avatar">G</div>
                <div class="agent-info">
                    <span class="agent-name">GRAVITAS chat</span>
                    <div class="header-right">
                        <vscode-button appearance="icon" id="stop-task-btn" title="Stop Task" class="hidden">
                            <span class="codicon codicon-stop"></span>
                        </vscode-button>
                        <div class="task-status-chip" id="task-status">Ready</div>
                    </div>
                </div>
            </div>
            <div class="user-bubble">${task.command}</div>
           <div class="task-body" id="body-${task.id}">
                <div class="operational-status" id="status-${task.id}">Initializing...</div>
            </div>
        `;
        return shell;
    }

    getTaskShell(taskId) {
        return document.getElementById(`task-${taskId}`);
    }

    updateTaskUI(task) {
        const badge = document.getElementById(`badge-${task.id}`);
        const status = document.getElementById(`status-${task.id}`);
        if (badge) badge.innerText = task.status;
        if (status && task.operationalStatus) {
            status.innerText = task.operationalStatus;
            status.classList.toggle('thinking', task.operationalStatus.includes('...'));
        }
    }

    renderEvent(taskId, event) {
        const body = document.getElementById(`body-${taskId}`);
        if (!body) return;

        // Cleanup: If a "Final result" arrives, clear the streaming buffer
        const finalResults = ['ThoughtCompleted', 'CoderResultEmitted', 'ReviewerResultEmitted'];
        if (finalResults.includes(event.type)) {
            const stream = body.querySelector('.active-stream-block');
            if (stream) stream.remove();
        }

        switch (event.type) {
            case 'TaskStatusEmitted':
                const statusLine = document.getElementById(`status-${taskId}`);
                if (statusLine) {
                    if (this.currentStatusEl) {
                        this.currentStatusEl.textContent = event.status;
                    }
        
                    // Show/Hide Stop button based on activity
                    const activeStates = ['Thinking...', 'Reviewing code...', 'Implementing...'];
                    if (this.currentStopBtn) {
                        if (activeStates.includes(event.status)) {
                            this.currentStopBtn.classList.remove('hidden');
                        } else {
                            this.currentStopBtn.classList.add('hidden');
                        }
                    }

                    statusLine.innerText = event.status;
                    statusLine.classList.toggle('thinking', event.status.includes('...'));
                }
                break;
            case 'ThoughtStarted':
                this.addEventLog(body, `💭 Thinking...`, 'thought-log');
                break;
            case 'ThoughtCompleted':
                this.addEventLog(body, `✅ ${event.content}`, 'success-log');
                break;
            case 'ToolCallEmitted':
                const toolColor = '#61AFEF'; // Soft blue for tools
                this.addEventLog(body, `🔧 Tool: ${event.tool}(${event.args})`, 'tool-log', toolColor);
                break;
            case 'CoderResultEmitted':
                this.addCodeLog(body, event.content, event.file);
                break;
            case 'ReviewerResultEmitted':
                const revColor = event.verdict === 'PASS' ? 'var(--accent-emerald)' : 'var(--vscode-errorForeground)';
                const summary = event.summary ? `: ${event.summary}` : '';
                this.addEventLog(body, `🔎 Review ${event.verdict}${summary}`, 'review-log', revColor);
                break;
            case 'StreamingChunkEmitted':
                this.handleStreamingChunk(taskId, event.chunk, event.stage);
                break;
        }
    }

    updateGlobalTelemetry(coder, reviewer, rag) {
        this.updateAgentDash('coder', coder);
        this.updateAgentDash('reviewer', reviewer);
        if (rag) {
            const ragDot = document.getElementById('ragStatus');
            if (ragDot) {
                ragDot.className = `status-dot ${rag.status.toLowerCase()}`;
                ragDot.title = `RAG Server: ${rag.status}`;
            }
        }
    }

    updateAgentDash(prefix, data) {
        const dash = document.getElementById(`${prefix}Dash`);
        const vram = document.getElementById(`${prefix}Vram`);
        const tps = document.getElementById(`${prefix}Tps`);
        const kv = document.getElementById(`${prefix}Kv`);
        const load = document.getElementById(`${prefix}Load`);

        if (dash) {
            dash.classList.remove('online', 'offline');
            dash.classList.add(data.status);
        }
        if (vram) {
            vram.textContent = data.vram;
            if (data.driver) vram.title = `${data.driver} Hardware (${data.mode})`;
        }
        if (tps) tps.textContent = data.tps;
        if (kv) kv.textContent = data.slots;
        if (load) {
            load.textContent = `${data.load.toUpperCase()} (${data.mode || 'LOCAL'})`;
            load.style.background = data.load === 'Idle' ? 'rgba(255,255,255,0.05)' : 'rgba(0, 229, 255, 0.2)';
            load.style.color = data.load === 'Idle' ? 'rgba(255,255,255,0.3)' : 'var(--accent-primary)';
        }
    }

    updateHardwareUI(taskId, metrics) {
        // Obsolete: Hardware UI now handled by global telemetry in header
        return;
    }

    handleStreamingChunk(taskId, chunk, stage) {
        const body = document.getElementById(`body-${taskId}`);
        if (!body) return;

        // Try to find if we already have an "Active Stream Block"
        let streamBlock = body.querySelector('.active-stream-block');
        
        if (!streamBlock) {
            // No active stream, create one
            streamBlock = document.createElement('div');
            streamBlock.className = 'active-stream-block';
            
            if (stage === 'implementation') {
                const wrapper = document.createElement('div');
                wrapper.className = 'code-result-container streaming';
                wrapper.innerHTML = `
                    <div class="code-header"><span>impl.ts (streaming)</span></div>
                    <pre class="code-block"><code></code></pre>
                `;
                streamBlock.appendChild(wrapper);
            } else {
                streamBlock.className += ' prose-stream';
                streamBlock.innerHTML = `<div class="markdown-log prose"></div>`;
            }
            body.appendChild(streamBlock);
        }

        // Append chunk
        if (stage === 'implementation') {
            const code = streamBlock.querySelector('code');
            if (code) code.innerText += chunk;
        } else {
            const md = streamBlock.querySelector('.markdown-log');
            if (md) md.innerText += chunk;
        }

        this.autoScroll();
    }

    addEventLog(container, text, className, color) {
        const log = document.createElement('div');
        log.className = `event-log ${className}`;
        log.innerText = text;
        if (color) log.style.color = color;
        container.appendChild(log);
    }

    addCodeLog(container, content, fileName) {
        // Intelligence: Is this a "Code Artifact" or "Message prose"?
        // 🧪 Refined detection: Ignore filename having a '.' as it's often a default.
        // Look for actual code signatures.
        const codeSigs = ['function', 'import ', 'export ', 'class ', 'const ', 'let ', 'var ', 'private ', 'public ', 'def ', 'async '];
        const looksLikeCode = codeSigs.some(sig => content.includes(sig)) || content.includes('=>') || (content.includes('{') && content.includes('}'));
        const looksLikeMarkdown = content.includes('###') || content.includes('| --- |') || content.includes('- **') || content.includes('## ');

        if (looksLikeMarkdown && !looksLikeCode) {
            const md = document.createElement('div');
            md.className = 'markdown-log prose';
            md.innerHTML = this.simpleMarkdown(content);
            container.appendChild(md);
        } else {
            const wrapper = document.createElement('div');
            wrapper.className = 'code-result-container';
            
            const header = document.createElement('div');
            header.className = 'code-header';
            header.innerHTML = `<span>${fileName || 'impl.ts'}</span>`;
            
            const pre = document.createElement('pre');
            pre.className = 'code-block';
            const codeElem = document.createElement('code');
            codeElem.innerText = content;
            pre.appendChild(codeElem);
            
            wrapper.appendChild(header);
            wrapper.appendChild(pre);
            container.appendChild(wrapper);
        }
    }

    simpleMarkdown(text) {
        return text
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/gm, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gm, '<em>$1</em>')
            .replace(/^\- (.*$)/gim, '<li>$1</li>')
            .replace(/\| (.*) \|/g, (match, p1) => {
                if (p1.includes('---')) return '<hr class="table-sep">';
                const cells = p1.split('|').map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${cells}</tr>`;
            })
            .replace(/\n\n/g, '<br>')
            .replace(/\n/g, ' '); // Basic normalization
    }
}

// 🚀 Boot Selection
(function() {
    try {
        console.log('Gravitas Chat: DOM Content Starting Boot...');
        const chatControl = new ChatController();
    } catch (e) {
        console.error('Gravitas Chat: Boot Failure!', e);
    }
})();
