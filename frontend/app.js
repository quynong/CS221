// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Example texts for demo
const EXAMPLE_TEXTS = [
    "Xin chào, tôi tên là Nguyễn Văn A. Email của tôi là nguyen.van.a@example.com và số điện thoại là 0912345678. Tôi sống tại số 123 đường Lê Lợi, quận 1, thành phố Hồ Chí Minh.",
    "Thông tin khách hàng: Họ tên: Trần Thị B, SĐT: 0987654321, Email: tran.thi.b@gmail.com, Địa chỉ: 456 phố Hàng Bông, quận Hoàn Kiếm, Hà Nội. Số thẻ tín dụng: 4532-1234-5678-9010.",
    "Liên hệ với chúng tôi qua email support@company.com hoặc gọi số 1900-1234. Văn phòng tại tầng 5, tòa nhà ABC, 789 đường Nguyễn Huệ, quận 3, TP.HCM."
];

// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const exampleBtn = document.getElementById('exampleBtn');
const charCount = document.getElementById('charCount');
const status = document.getElementById('status');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const highlightedText = document.getElementById('highlightedText');
const entitiesList = document.getElementById('entitiesList');
const tokensList = document.getElementById('tokensList');
const resultsStats = document.getElementById('resultsStats');

// Color mapping for entity types
const ENTITY_COLORS = {
    'FIRSTNAME': { bg: 'rgba(99, 102, 241, 0.2)', border: '#6366f1', text: '#4f46e5' },
    'MIDDLENAME': { bg: 'rgba(99, 102, 241, 0.2)', border: '#6366f1', text: '#4f46e5' },
    'LASTNAME': { bg: 'rgba(139, 92, 246, 0.2)', border: '#8b5cf6', text: '#8b5cf6' },
    'EMAIL': { bg: 'rgba(16, 185, 129, 0.2)', border: '#10b981', text: '#059669' },
    'PHONENUMBER': { bg: 'rgba(245, 158, 11, 0.2)', border: '#f59e0b', text: '#d97706' },
    'STREET': { bg: 'rgba(239, 68, 68, 0.2)', border: '#ef4444', text: '#dc2626' },
    'CITY': { bg: 'rgba(239, 68, 68, 0.2)', border: '#ef4444', text: '#dc2626' },
    'ADDRESS': { bg: 'rgba(239, 68, 68, 0.2)', border: '#ef4444', text: '#dc2626' },
    'CREDITCARDNUMBER': { bg: 'rgba(168, 85, 247, 0.2)', border: '#a855f7', text: '#7c3aed' },
    'ACCOUNTNUMBER': { bg: 'rgba(168, 85, 247, 0.2)', border: '#a855f7', text: '#7c3aed' },
    'DOB': { bg: 'rgba(236, 72, 153, 0.2)', border: '#ec4899', text: '#db2777' },
    'AGE': { bg: 'rgba(236, 72, 153, 0.2)', border: '#ec4899', text: '#db2777' },
    'IPV4': { bg: 'rgba(34, 197, 94, 0.2)', border: '#22c55e', text: '#16a34a' },
    'URL': { bg: 'rgba(34, 197, 94, 0.2)', border: '#22c55e', text: '#16a34a' },
};

// Get color for entity type
function getEntityColor(entityType) {
    return ENTITY_COLORS[entityType] || {
        bg: 'rgba(99, 102, 241, 0.15)',
        border: '#6366f1',
        text: '#4f46e5'
    };
}

// Update character count
function updateCharCount() {
    const count = textInput.value.length;
    charCount.textContent = `${count.toLocaleString()} ký tự`;
}

// Show status message
function showStatus(message, type = '') {
    status.textContent = message;
    status.className = `status ${type}`;
    if (message) {
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 3000);
    }
}

// Set loading state
function setLoading(loading) {
    analyzeBtn.disabled = loading;
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline-block';
        btnLoader.style.display = 'none';
    }
}

// Make API request
async function analyzeText() {
    const text = textInput.value.trim();
    
    if (!text) {
        showStatus('Vui lòng nhập văn bản cần phân tích', 'error');
        return;
    }

    setLoading(true);
    hideResults();
    showStatus('Đang phân tích...', 'success');

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Lỗi khi gọi API');
        }

        const data = await response.json();
        displayResults(data);
        showStatus('Phân tích thành công!', 'success');
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Không thể kết nối đến server. Vui lòng kiểm tra backend đã chạy chưa.');
        showStatus('Lỗi khi phân tích', 'error');
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    resultsSection.style.display = 'block';
    errorSection.style.display = 'none';

    // Display statistics
    displayStats(data.stats, data.has_pii);

    // Display highlighted text
    displayHighlightedText(data.tokens);

    // Display entities
    displayEntities(data.entities);

    // Display token details
    displayTokenDetails(data.tokens);
}

// Display statistics
function displayStats(stats, hasPII) {
    resultsStats.innerHTML = `
        <div class="stat-badge ${hasPII ? 'warning' : 'success'}">
            <span>${hasPII ? '⚠️' : '✅'}</span>
            <span>${hasPII ? 'Có PII' : 'Không có PII'}</span>
        </div>
        <div class="stat-badge primary">
            <span>📊</span>
            <span>${stats.total_entities || 0} thực thể</span>
        </div>
        <div class="stat-badge">
            <span>🔤</span>
            <span>${stats.total_tokens || 0} tokens</span>
        </div>
        <div class="stat-badge">
            <span>🏷️</span>
            <span>${stats.pii_tokens || 0} PII tokens</span>
        </div>
    `;
}

// Display highlighted text
function displayHighlightedText(tokens) {
    let html = '';
    let currentEntity = null;
    let entityTokens = [];

    tokens.forEach((tokenData, index) => {
        const { token, label } = tokenData;
        const isSubword = token.startsWith('##');
        const displayToken = isSubword ? token.replace('##', '') : token;
        const space = isSubword ? '' : ' ';

        if (label === 'O') {
            // End current entity if exists
            if (currentEntity) {
                html += createEntitySpan(entityTokens.join(''), currentEntity);
                currentEntity = null;
                entityTokens = [];
            }
            html += `<span class="token O">${escapeHtml(displayToken)}</span>${space}`;
        } else if (label.startsWith('B-')) {
            // End previous entity if exists
            if (currentEntity) {
                html += createEntitySpan(entityTokens.join(''), currentEntity);
            }
            // Start new entity
            const entityType = label.substring(2);
            currentEntity = entityType;
            entityTokens = [displayToken];
        } else if (label.startsWith('I-')) {
            const entityType = label.substring(2);
            if (entityType === currentEntity) {
                entityTokens.push(displayToken);
            } else {
                // Entity type changed
                if (currentEntity) {
                    html += createEntitySpan(entityTokens.join(''), currentEntity);
                }
                currentEntity = entityType;
                entityTokens = [displayToken];
            }
        }
    });

    // Handle last entity
    if (currentEntity) {
        html += createEntitySpan(entityTokens.join(''), currentEntity);
    }

    highlightedText.innerHTML = html;
}

// Create entity span
function createEntitySpan(text, entityType) {
    const color = getEntityColor(entityType);
    return `<span class="token B-${entityType}" style="background: ${color.bg}; color: ${color.text}; border-bottom: 2px solid ${color.border};" title="${entityType}">${escapeHtml(text)}</span> `;
}

// Display entities
function displayEntities(entities) {
    if (!entities || Object.keys(entities).length === 0) {
        entitiesList.innerHTML = '<p style="color: var(--text-secondary);">Không phát hiện thực thể nào.</p>';
        return;
    }

    let html = '';
    Object.entries(entities).forEach(([entityType, values]) => {
        const color = getEntityColor(entityType);
        html += `
            <div class="entity-group">
                <div class="entity-group-header" style="color: ${color.text};">
                    ${entityType} (${values.length})
                </div>
                <div class="entity-items">
                    ${values.map(value => `
                        <div class="entity-item">${escapeHtml(value)}</div>
                    `).join('')}
                </div>
            </div>
        `;
    });

    entitiesList.innerHTML = html;
}

// Display token details
function displayTokenDetails(tokens) {
    if (!tokens || tokens.length === 0) {
        tokensList.innerHTML = '<p style="color: var(--text-secondary);">Không có token nào.</p>';
        return;
    }

    // Limit display to first 100 tokens for performance
    const displayTokens = tokens.slice(0, 100);
    const remaining = tokens.length - 100;

    let html = displayTokens.map(({ token, label }) => {
        const isPII = label !== 'O';
        return `
            <div class="token-item">
                <span class="token-text">${escapeHtml(token)}</span>
                <span class="token-label ${label}">${escapeHtml(label)}</span>
            </div>
        `;
    }).join('');

    if (remaining > 0) {
        html += `<div class="token-item" style="width: 100%; justify-content: center; background: var(--bg-secondary);">
            <span>... và ${remaining} tokens khác</span>
        </div>`;
    }

    tokensList.innerHTML = html;
}

// Show error
function showError(message) {
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorMessage.textContent = message;
}

// Hide results
function hideResults() {
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Clear input
function clearInput() {
    textInput.value = '';
    updateCharCount();
    hideResults();
    showStatus('Đã xóa', 'success');
}

// Load example
function loadExample() {
    const randomExample = EXAMPLE_TEXTS[Math.floor(Math.random() * EXAMPLE_TEXTS.length)];
    textInput.value = randomExample;
    updateCharCount();
    hideResults();
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                showStatus('API đã sẵn sàng', 'success');
            } else {
                showStatus('API chưa load model', 'error');
            }
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
}

// Event Listeners
textInput.addEventListener('input', updateCharCount);
analyzeBtn.addEventListener('click', analyzeText);
clearBtn.addEventListener('click', clearInput);
exampleBtn.addEventListener('click', loadExample);

// Enter key to analyze (Ctrl+Enter or Cmd+Enter)
textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        analyzeText();
    }
});

// Initialize
updateCharCount();
checkAPIHealth();

// Check API health every 30 seconds
setInterval(checkAPIHealth, 30000);


