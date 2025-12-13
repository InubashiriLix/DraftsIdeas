# 🍅 Todo & Pomodoro Timer

A beautiful and modern web application combining a Todo list manager with a Pomodoro timer, built with FastAPI and vanilla JavaScript.

## Features

### Todo Management
- Create, read, update, and delete todo items
- Track todo status (pending, in_progress, completed)
- Filter todos by status
- Timestamp tracking for creation and updates

### Pomodoro Timer
- Start work sessions (default: 25 minutes)
- Short breaks (default: 5 minutes)
- Long breaks (default: 15 minutes)
- Track active sessions
- View session history
- Get productivity statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload
```

3. Open your browser and visit:
   - **Web App**: `http://localhost:8000` (Beautiful frontend UI)
   - **API Docs**: `http://localhost:8000/docs` (Swagger UI)
   - **ReDoc**: `http://localhost:8000/redoc` (Alternative API docs)

## Frontend Features

The beautiful dark-themed UI includes:
- 🎨 Modern gradient design with smooth animations
- ⏰ Real-time countdown timer with pulsing effects
- 📊 Live productivity statistics
- 🎯 Intuitive todo management with status tracking
- 📱 Responsive design for desktop and mobile
- 🔔 Toast notifications for actions
- ⌨️ Keyboard shortcuts (Enter to add todos)

## API Endpoints

### Todo Endpoints

- `POST /todos` - Create a new todo
- `GET /todos` - Get all todos (optional: filter by status)
- `GET /todos/{todo_id}` - Get a specific todo
- `PUT /todos/{todo_id}` - Update a todo
- `DELETE /todos/{todo_id}` - Delete a todo

### Pomodoro Endpoints

- `POST /pomodoro/start` - Start a new Pomodoro session
- `GET /pomodoro/active` - Get the currently active session
- `POST /pomodoro/{session_id}/stop` - Stop an active session
- `GET /pomodoro/history` - Get session history
- `GET /pomodoro/stats` - Get productivity statistics

## Example Usage

### Create a Todo
```bash
curl -X POST "http://localhost:8000/todos" \
  -H "Content-Type: application/json" \
  -d '{"title": "Complete project", "description": "Finish the FastAPI app"}'
```

### Start a Pomodoro Work Session
```bash
curl -X POST "http://localhost:8000/pomodoro/start" \
  -H "Content-Type: application/json" \
  -d '{"session_type": "work"}'
```

### Get Active Pomodoro Session
```bash
curl "http://localhost:8000/pomodoro/active"
```

### Update Todo Status
```bash
curl -X PUT "http://localhost:8000/todos/{todo_id}" \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

## Data Storage

This application uses in-memory storage, so all data will be lost when the server restarts. For production use, consider integrating a database like PostgreSQL or SQLite.
