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
# Helpers
# ===========================
echo_hr() { echo "=========================================="; }

DC() {
  if command -v docker-compose >/dev/null 2>&1; then docker-compose "$@"; else docker compose "$@"; fi
}

dc_port() {
  local svc="$1" internal="$2"
  local port
  port="$(DC port "$svc" "$internal" 2>/dev/null | awk -F: 'NF{print $NF; exit}')"
  echo "${port:-$internal}"
}

wait_http_ok() {
  local url="$1" tries="${2:-10}"
  for ((i=1;i<=tries;i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

open_url() {
  local url="$1"
  [[ "$OPEN_BROWSER" == "1" ]] || { echo "ℹ️  Open manually: $url"; return; }
  case "$OSTYPE" in
    darwin*) open "$url" ;;
    linux-gnu*) command -v xdg-open >/dev/null && xdg-open "$url" || command -v gnome-open >/dev/null && gnome-open "$url" || echo "ℹ️  Open manually: $url" ;;
    msys*|cygwin*) start "$url" ;;
    *) echo "ℹ️  Open manually: $url" ;;
  esac
}
# ===========================
# Step 0: OPTIONAL extract data.zip → ./data (flatten)
# ===========================
echo_hr
echo "📦 Preparing optional ${DATA_ZIP} → ${DATA_DIR} (flatten 1-level)"
echo_hr

if [[ -f "$DATA_ZIP" ]]; then
  if ! command -v unzip >/dev/null 2>&1; then
    echo "❌ 'unzip' not found. Install it first (macOS: brew install unzip, Ubuntu: sudo apt install unzip)."
    exit 1
  fi

  # 清理并解压到临时目录
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  TMP_DIR="$(mktemp -d)"
  if ! unzip -o -q "$DATA_ZIP" -d "$TMP_DIR"; then
    echo "❌ Failed to extract $DATA_ZIP"
    exit 1
  fi

  # 移除 macOS 垃圾文件
  find "$TMP_DIR" -name "__MACOSX" -type d -prune -exec rm -rf {} + || true
  find "$TMP_DIR" -name ".DS_Store" -type f -delete || true

  # 扁平化：把顶层目录的“内容”合并到 data/，顶层文件直接移入
  shopt -s dotglob nullglob
  for entry in "$TMP_DIR"/*; do
    if [[ -d "$entry" ]]; then
      rsync -a "$entry"/ "$DATA_DIR"/
    elif [[ -f "$entry" ]]; then
      mv "$entry" "$DATA_DIR"/
    fi
  done
  shopt -u dotglob nullglob

  rm -rf "$TMP_DIR"
  echo "   ✓ Flattened into: $DATA_DIR"
  echo
else
  echo "ℹ️  '$DATA_ZIP' not found — skipping extraction. Existing '$DATA_DIR' will be used if present."
  mkdir -p "$DATA_DIR"
fi

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
if ! DC up -d "$@"; then
  echo "❌ Docker startup failed! Check logs with: docker-compose logs (or docker compose logs)"
  exit 1
fi
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
if command -v docker-compose >/dev/null 2>&1; then
  echo "   • Logs:      docker-compose logs -f"
  echo "   • Stop:      docker-compose down"
  echo "   • Restart:   docker-compose restart"
  echo "   • Status:    docker-compose ps"
else
  echo "   • Logs:      docker compose logs -f"
  echo "   • Stop:      docker compose down"
  echo "   • Restart:   docker compose restart"
  echo "   • Status:    docker compose ps"
fi
echo

# ===========================
# Step 6: Open UI
# ===========================
URL="http://localhost:${FRONTEND_PORT}"
echo "🌐 Opening frontend: $URL"
sleep 1
open_url "$URL"

echo
echo "🎉 Ready! If the browser didn’t open, visit:"
echo "   $URL"
echo_hr
