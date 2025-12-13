from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path

app = FastAPI(title="Todo & Pomodoro Timer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")


# Models
class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None


class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TodoStatus] = None


class Todo(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: TodoStatus = TodoStatus.pending
    created_at: datetime
    updated_at: datetime


class PomodoroSessionType(str, Enum):
    work = "work"
    short_break = "short_break"
    long_break = "long_break"


class PomodoroStart(BaseModel):
    session_type: PomodoroSessionType = PomodoroSessionType.work
    duration_minutes: Optional[int] = None


class PomodoroSession(BaseModel):
    id: str
    session_type: PomodoroSessionType
    duration_minutes: int
    start_time: datetime
    end_time: datetime
    is_active: bool
    is_completed: bool


# In-memory storage
todos_db: dict[str, Todo] = {}
pomodoro_sessions: dict[str, PomodoroSession] = {}

# Default durations in minutes
DEFAULT_DURATIONS = {
    PomodoroSessionType.work: 25,
    PomodoroSessionType.short_break: 5,
    PomodoroSessionType.long_break: 15,
}


# Root endpoint - serve frontend
@app.get("/")
def root():
    return FileResponse(static_path / "index.html")


@app.get("/api")
def api_info():
    return {
        "message": "Todo & Pomodoro Timer API",
        "endpoints": {"todos": "/todos", "pomodoro": "/pomodoro", "docs": "/docs"},
    }


# Todo endpoints
@app.post("/todos", response_model=Todo, status_code=201)
def create_todo(todo: TodoCreate):
    """Create a new todo item"""
    todo_id = str(uuid.uuid4())
    now = datetime.now()
    new_todo = Todo(
        id=todo_id,
        title=todo.title,
        description=todo.description,
        status=TodoStatus.pending,
        created_at=now,
        updated_at=now,
    )
    todos_db[todo_id] = new_todo
    return new_todo


@app.get("/todos", response_model=List[Todo])
def get_todos(status: Optional[TodoStatus] = None):
    """Get all todos, optionally filtered by status"""
    todos = list(todos_db.values())
    if status:
        todos = [todo for todo in todos if todo.status == status]
    return sorted(todos, key=lambda x: x.created_at, reverse=True)


@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: str):
    """Get a specific todo by ID"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos_db[todo_id]


@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: str, todo_update: TodoUpdate):
    """Update a todo item"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")

    todo = todos_db[todo_id]
    update_data = todo_update.model_dump(exclude_unset=True)

    if "title" in update_data:
        todo.title = update_data["title"]
    if "description" in update_data:
        todo.description = update_data["description"]
    if "status" in update_data:
        todo.status = update_data["status"]

    todo.updated_at = datetime.now()
    return todo


@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: str):
    """Delete a todo item"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos_db[todo_id]
    return None


# Pomodoro endpoints
@app.post("/pomodoro/start", response_model=PomodoroSession, status_code=201)
def start_pomodoro(session: PomodoroStart):
    """Start a new Pomodoro session"""
    # Check if there's already an active session
    active_sessions = [s for s in pomodoro_sessions.values() if s.is_active]
    if active_sessions:
        raise HTTPException(
            status_code=400,
            detail="There is already an active Pomodoro session. Stop it first.",
        )

    session_id = str(uuid.uuid4())
    duration = session.duration_minutes or DEFAULT_DURATIONS[session.session_type]
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration)

    new_session = PomodoroSession(
        id=session_id,
        session_type=session.session_type,
        duration_minutes=duration,
        start_time=start_time,
        end_time=end_time,
        is_active=True,
        is_completed=False,
    )
    pomodoro_sessions[session_id] = new_session
    return new_session


@app.get("/pomodoro/active", response_model=Optional[PomodoroSession])
def get_active_pomodoro():
    """Get the currently active Pomodoro session"""
    active_sessions = [s for s in pomodoro_sessions.values() if s.is_active]
    if not active_sessions:
        return None

    session = active_sessions[0]
    # Check if session should be auto-completed
    if datetime.now() >= session.end_time:
        session.is_active = False
        session.is_completed = True

    return session


@app.post("/pomodoro/{session_id}/stop")
def stop_pomodoro(session_id: str):
    """Stop an active Pomodoro session"""
    if session_id not in pomodoro_sessions:
        raise HTTPException(status_code=404, detail="Pomodoro session not found")

    session = pomodoro_sessions[session_id]
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")

    session.is_active = False
    session.is_completed = datetime.now() >= session.end_time

    return {"message": "Pomodoro session stopped", "session": session}


@app.get("/pomodoro/history", response_model=List[PomodoroSession])
def get_pomodoro_history(limit: int = 10):
    """Get Pomodoro session history"""
    sessions = sorted(
        pomodoro_sessions.values(), key=lambda x: x.start_time, reverse=True
    )
    return sessions[:limit]


@app.get("/pomodoro/stats")
def get_pomodoro_stats():
    """Get Pomodoro statistics"""
    completed_sessions = [s for s in pomodoro_sessions.values() if s.is_completed]
    work_sessions = [
        s for s in completed_sessions if s.session_type == PomodoroSessionType.work
    ]

    total_work_minutes = sum(s.duration_minutes for s in work_sessions)

    return {
        "total_sessions": len(completed_sessions),
        "completed_work_sessions": len(work_sessions),
        "total_work_minutes": total_work_minutes,
        "total_work_hours": round(total_work_minutes / 60, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
