#!/usr/bin/env bash
set -euo pipefail

# Simple static file server for the SuperSubScriptGenerator.
# Usage: ./serve.sh [host] [port]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to serve the static files." >&2
  exit 1
fi

cd "$ROOT_DIR"
echo "Serving SuperSubScriptGenerator on http://${HOST}:${PORT} (Ctrl+C to stop)"
python3 -m http.server "$PORT" --bind "$HOST"
