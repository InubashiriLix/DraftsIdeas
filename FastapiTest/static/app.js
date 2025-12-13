const API_URL = 'http://localhost:8000';

let currentFilter = 'all';
let activeSession = null;
let timerInterval = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadTodos();
    loadStats();
    checkActivePomodoro();
    
    // Set up auto-refresh
    setInterval(checkActivePomodoro, 1000);
    setInterval(loadStats, 5000);
});

// Notification system
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Todo Functions
async function createTodo() {
    const title = document.getElementById('todoTitle').value.trim();
    const description = document.getElementById('todoDescription').value.trim();
    
    if (!title) {
        showNotification('Please enter a todo title', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/todos`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, description: description || null })
        });
        
        if (response.ok) {
            document.getElementById('todoTitle').value = '';
            document.getElementById('todoDescription').value = '';
            showNotification('Todo created successfully!');
            loadTodos();
        }
    } catch (error) {
        showNotification('Error creating todo', 'error');
        console.error(error);
    }
}

async function loadTodos() {
    try {
        const url = currentFilter === 'all' 
            ? `${API_URL}/todos`
            : `${API_URL}/todos?status=${currentFilter}`;
            
        const response = await fetch(url);
        const todos = await response.json();
        displayTodos(todos);
    } catch (error) {
        console.error('Error loading todos:', error);
    }
}

function displayTodos(todos) {
    const todoList = document.getElementById('todoList');
    
    if (todos.length === 0) {
        todoList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>No todos yet. Create one to get started!</p>
            </div>
        `;
        return;
    }
    
    todoList.innerHTML = todos.map(todo => `
        <div class="todo-item ${todo.status}" data-id="${todo.id}">
            <div class="todo-header">
                <div class="todo-title">${escapeHtml(todo.title)}</div>
            </div>
            ${todo.description ? `<div class="todo-description">${escapeHtml(todo.description)}</div>` : ''}
            <div class="todo-footer">
                <span class="todo-status ${todo.status}">
                    ${todo.status.replace('_', ' ')}
                </span>
                <div class="todo-actions">
                    ${todo.status !== 'in_progress' ? 
                        `<button class="btn btn-primary btn-small" onclick="updateTodoStatus('${todo.id}', 'in_progress')">
                            ▶️ Start
                        </button>` : ''}
                    ${todo.status !== 'completed' ? 
                        `<button class="btn btn-success btn-small" onclick="updateTodoStatus('${todo.id}', 'completed')">
                            ✓ Complete
                        </button>` : ''}
                    <button class="btn btn-danger btn-small" onclick="deleteTodo('${todo.id}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

async function updateTodoStatus(todoId, status) {
    try {
        const response = await fetch(`${API_URL}/todos/${todoId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status })
        });
        
        if (response.ok) {
            showNotification('Todo updated!');
            loadTodos();
        }
    } catch (error) {
        showNotification('Error updating todo', 'error');
        console.error(error);
    }
}

async function deleteTodo(todoId) {
    if (!confirm('Are you sure you want to delete this todo?')) return;
    
    try {
        const response = await fetch(`${API_URL}/todos/${todoId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('Todo deleted!');
            loadTodos();
        }
    } catch (error) {
        showNotification('Error deleting todo', 'error');
        console.error(error);
    }
}

function filterTodos(filter) {
    currentFilter = filter;
    
    // Update active tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    loadTodos();
}

// Pomodoro Functions
async function startPomodoro(sessionType) {
    try {
        const response = await fetch(`${API_URL}/pomodoro/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_type: sessionType })
        });
        
        if (response.ok) {
            const session = await response.json();
            showNotification(`${sessionType.replace('_', ' ')} session started!`);
            activeSession = session;
            updateTimerDisplay();
            showStopButton();
        } else {
            const error = await response.json();
            showNotification(error.detail, 'error');
        }
    } catch (error) {
        showNotification('Error starting pomodoro', 'error');
        console.error(error);
    }
}

async function stopPomodoro() {
    if (!activeSession) return;
    
    try {
        const response = await fetch(`${API_URL}/pomodoro/${activeSession.id}/stop`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showNotification('Pomodoro session stopped!');
            activeSession = null;
            hideStopButton();
            resetTimerDisplay();
            loadStats();
        }
    } catch (error) {
        showNotification('Error stopping pomodoro', 'error');
        console.error(error);
    }
}

async function checkActivePomodoro() {
    try {
        const response = await fetch(`${API_URL}/pomodoro/active`);
        const session = await response.json();
        
        if (session && session.is_active) {
            activeSession = session;
            updateTimerDisplay();
            showStopButton();
        } else if (activeSession && activeSession.is_active) {
            // Session just completed
            showNotification('Pomodoro session completed! 🎉');
            activeSession = null;
            hideStopButton();
            resetTimerDisplay();
            loadStats();
        }
    } catch (error) {
        console.error('Error checking active pomodoro:', error);
    }
}

function updateTimerDisplay() {
    if (!activeSession) return;
    
    const now = new Date();
    const endTime = new Date(activeSession.end_time);
    const remainingMs = endTime - now;
    
    if (remainingMs <= 0) {
        document.querySelector('.time').textContent = '00:00';
        document.querySelector('.timer-status').textContent = 'Completed!';
        document.getElementById('timerDisplay').classList.remove('active');
        return;
    }
    
    const minutes = Math.floor(remainingMs / 60000);
    const seconds = Math.floor((remainingMs % 60000) / 1000);
    
    document.querySelector('.time').textContent = 
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    document.querySelector('.timer-status').textContent = 
        `${activeSession.session_type.replace('_', ' ')} in progress`;
    document.getElementById('timerDisplay').classList.add('active');
}

function resetTimerDisplay() {
    document.querySelector('.time').textContent = '25:00';
    document.querySelector('.timer-status').textContent = 'Ready to start';
    document.getElementById('timerDisplay').classList.remove('active');
}

function showStopButton() {
    document.getElementById('startWorkBtn').style.display = 'none';
    document.getElementById('startBreakBtn').style.display = 'none';
    document.getElementById('startLongBreakBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('stopBtn').style.gridColumn = '1 / -1';
}

function hideStopButton() {
    document.getElementById('startWorkBtn').style.display = 'block';
    document.getElementById('startBreakBtn').style.display = 'block';
    document.getElementById('startLongBreakBtn').style.display = 'block';
    document.getElementById('stopBtn').style.display = 'none';
}

async function loadStats() {
    try {
        const response = await fetch(`${API_URL}/pomodoro/stats`);
        const stats = await response.json();
        
        document.getElementById('totalSessions').textContent = stats.total_sessions;
        document.getElementById('workSessions').textContent = stats.completed_work_sessions;
        document.getElementById('totalHours').textContent = `${stats.total_work_hours}h`;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Utility function
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key to submit todo
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('todoTitle')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') createTodo();
    });
    document.getElementById('todoDescription')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') createTodo();
    });
});
