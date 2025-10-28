#!/bin/bash
set -euo pipefail

# ===========================
# Config
# ===========================
DATA_ZIP="data.zip"
DATA_DIR="data"

FRONTEND_SERVICE="${FRONTEND_SERVICE:-frontend}"
BACKEND_SERVICE="${BACKEND_SERVICE:-backend}"
QDRANT_SERVICE="${QDRANT_SERVICE:-qdrant}"

FRONTEND_INTERNAL_PORT="${FRONTEND_INTERNAL_PORT:-8501}"
BACKEND_INTERNAL_PORT="${BACKEND_INTERNAL_PORT:-8888}"
QDRANT_INTERNAL_PORT="${QDRANT_INTERNAL_PORT:-6333}"

OPEN_BROWSER="${OPEN_BROWSER:-1}"

# ===========================
# Helper functions
# ===========================
echo_hr() { echo "=========================================="; }

DC() {
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    docker compose "$@"
  fi
}

dc_port() {
  local svc="$1" internal="$2"
  local port
  port="$(DC port "$svc" "$internal" 2>/dev/null | awk -F: 'NF{print $NF; exit}')"
  echo "${port:-$internal}"
}

wait_http_ok() {
  local url="$1"
  local tries="${2:-10}"
  for ((i=1;i<=tries;i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

open_url() {
  local url="$1"
  if [[ "$OPEN_BROWSER" != "1" ]]; then
    echo "ℹ️  Please open manually: $url"
    return
  fi
  case "$OSTYPE" in
    darwin*) open "$url" ;;
    linux-gnu*)
      command -v xdg-open >/dev/null && xdg-open "$url" \
        || command -v gnome-open >/dev/null && gnome-open "$url" \
        || echo "ℹ️  Please open manually: $url"
      ;;
    msys*|cygwin*) start "$url" ;;
    *) echo "ℹ️  Please open manually: $url" ;;
  esac
}

# ===========================
# Step 0: Ensure unzip + extract data.zip into ./data/
# ===========================
echo_hr
echo "📦 Checking and extracting ${DATA_ZIP}..."
echo_hr

if [[ ! -f "$DATA_ZIP" ]]; then
  echo "❌ Required file '$DATA_ZIP' not found!"
  echo "   Please place data.zip in the same directory as this script."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "❌ unzip not found! Please install first:"
  echo "   macOS: brew install unzip  |  Ubuntu: sudo apt install unzip"
  exit 1
fi

# Always clean previous data folder
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# Extract to a temporary folder to detect nesting
TMP_DIR=$(mktemp -d)
unzip -o -q "$DATA_ZIP" -d "$TMP_DIR" || {
  echo "❌ Failed to extract $DATA_ZIP"
  exit 1
}

# Check if the extracted folder contains a single directory wrapper (data/data/xxx)
first_subdir="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
subdir_count="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"

if [[ -n "$first_subdir" && "$subdir_count" -eq 1 ]]; then
  echo "   • Detected nested folder inside zip: flattening structure..."
  mv "$first_subdir"/* "$DATA_DIR"/
else
  echo "   • Extracting files directly into $DATA_DIR"
  mv "$TMP_DIR"/* "$DATA_DIR"/ 2>/dev/null || true
fi

rm -rf "$TMP_DIR"
echo "   ✓ Extraction complete: $DATA_DIR"
echo

# ===========================
# Step 1: Clean env
# ===========================
unset QDRANT_HOST
unset QDRANT_SEED_PATH

echo_hr
echo "🚀 Starting AI Assessment Platform"
echo_hr
echo "🧹 Environment cleaned:"
echo "   ✓ QDRANT_HOST unset"
echo "   ✓ QDRANT_SEED_PATH unset"
echo

# ===========================
# Step 2: Start containers
# ===========================
echo "🐳 Starting Docker containers..."
DC up -d "$@" || {
  echo "❌ Docker startup failed! Check logs with: docker-compose logs"
  exit 1
}
echo "✅ Containers started successfully!"
echo

# ===========================
# Step 3: Detect mapped ports
# ===========================
FRONTEND_PORT="$(dc_port "$FRONTEND_SERVICE" "$FRONTEND_INTERNAL_PORT")"
BACKEND_PORT="$(dc_port "$BACKEND_SERVICE" "$BACKEND_INTERNAL_PORT")"
QDRANT_PORT="$(dc_port "$QDRANT_SERVICE" "$QDRANT_INTERNAL_PORT")"

# ===========================
# Step 4: Health checks
# ===========================
echo "⏳ Waiting for backend and Qdrant..."
sleep 3

echo "🔍 Checking backend API..."
if wait_http_ok "http://localhost:${BACKEND_PORT}/health"; then
  echo "   ✓ Backend API is ready"
else
  echo "   ⚠️  Backend not responding"
fi

echo "🔍 Checking Qdrant..."
if wait_http_ok "http://localhost:${QDRANT_PORT}"; then
  echo "   ✓ Qdrant is ready"
else
  echo "   ⚠️  Qdrant not responding"
fi

# ===========================
# Step 5: Summary
# ===========================
echo
echo_hr
echo "✅ AI Assessment Platform Started!"
echo_hr
echo
echo "📊 Services:"
echo "   • Frontend:  http://localhost:${FRONTEND_PORT}"
echo "   • Backend:   http://localhost:${BACKEND_PORT}"
echo "   • API Docs:  http://localhost:${BACKEND_PORT}/docs"
echo "   • Qdrant:    http://localhost:${QDRANT_PORT}/dashboard"
echo
echo "📝 Commands:"
echo "   • Logs:      docker-compose logs -f"
echo "   • Stop:      docker-compose down"
echo "   • Restart:   docker-compose restart"
echo "   • Status:    docker-compose ps"
echo

# ===========================
# Step 6: Open UI
# ===========================
URL="http://localhost:${FRONTEND_PORT}"
echo "🌐 Opening frontend: $URL"
sleep 1
open_url "$URL"

echo
echo "🎉 Ready! If the UI didn’t open automatically, visit:"
echo "   $URL"
echo_hr
