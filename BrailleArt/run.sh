#!/bin/bash

# Braille Art Converter - Startup Script

echo "🎨 Starting Braille Art Converter..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR" || exit 1

# Check if port 8001 is already in use
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8001 is already in use!"
    echo "Stopping existing server..."
    pkill -f "python3 -m http.server 8001" 2>/dev/null
    sleep 1
fi

# Start the web server
echo "🚀 Starting web server on http://localhost:8001"
echo ""
echo "📋 Open your browser and navigate to:"
echo "   http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop the server"
echo "─────────────────────────────────────"

# Start Python HTTP server
python3 -m http.server 8001
