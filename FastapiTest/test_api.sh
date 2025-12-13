#!/bin/bash
# Simple test script for the Todo & Pomodoro API

BASE_URL="http://localhost:8000"

echo "=== Testing Todo & Pomodoro API ==="
echo ""

echo "1. Testing root endpoint:"
curl -s $BASE_URL | python -m json.tool
echo ""

echo "2. Creating a todo:"
TODO_RESPONSE=$(curl -s -X POST $BASE_URL/todos \
  -H "Content-Type: application/json" \
  -d '{"title":"Complete FastAPI project","description":"Build todo and pomodoro app"}')
echo $TODO_RESPONSE | python -m json.tool
TODO_ID=$(echo $TODO_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo ""

echo "3. Getting all todos:"
curl -s $BASE_URL/todos | python -m json.tool
echo ""

echo "4. Starting a Pomodoro work session:"
POMODORO_RESPONSE=$(curl -s -X POST $BASE_URL/pomodoro/start \
  -H "Content-Type: application/json" \
  -d '{"session_type":"work"}')
echo $POMODORO_RESPONSE | python -m json.tool
POMODORO_ID=$(echo $POMODORO_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo ""

echo "5. Getting active Pomodoro session:"
curl -s $BASE_URL/pomodoro/active | python -m json.tool
echo ""

echo "6. Updating todo status:"
curl -s -X PUT $BASE_URL/todos/$TODO_ID \
  -H "Content-Type: application/json" \
  -d '{"status":"in_progress"}' | python -m json.tool
echo ""

echo "7. Getting Pomodoro stats:"
curl -s $BASE_URL/pomodoro/stats | python -m json.tool
echo ""

echo "=== Tests completed! ==="
