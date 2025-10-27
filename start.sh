#!/bin/bash

# Unset host environment variables that should use container names/paths in Docker
# These variables may be set in your shell for local development,
# but Docker containers need different values (container names, absolute paths)
unset QDRANT_HOST
unset QDRANT_SEED_PATH

echo "=========================================="
echo "🚀 Starting AI Assessment Platform"
echo "=========================================="
echo ""
echo "🧹 Cleaning environment variables..."
echo "   ✓ QDRANT_HOST unset (will use 'qdrant' container name)"
echo "   ✓ QDRANT_SEED_PATH unset (will use container path)"
echo ""

# Start Docker Compose
echo "🐳 Starting Docker containers..."
docker-compose up -d "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to start containers"
    echo "   Check: docker-compose logs"
    exit 1
fi

echo ""
echo "✅ Containers started successfully!"
echo ""

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 3

# Check backend health
echo "🔍 Checking backend API..."
for i in {1..10}; do
    if curl -s http://localhost:8888/health > /dev/null 2>&1; then
        echo "   ✓ Backend API is ready"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "   ⚠️  Backend API not responding (check logs with: docker-compose logs backend)"
    else
        sleep 2
    fi
done

# Check Qdrant
echo "🔍 Checking Qdrant..."
if curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo "   ✓ Qdrant is ready"
else
    echo "   ⚠️  Qdrant not responding (check logs with: docker-compose logs qdrant)"
fi

echo ""
echo "=========================================="
echo "✅ AI Assessment Platform Started!"
echo "=========================================="
echo ""
echo "📊 Available Services:"
echo "   • Frontend:        http://localhost:8501"
echo "   • Backend API:     http://localhost:8888"
echo "   • API Docs:        http://localhost:8888/docs"
echo "   • Qdrant:          http://localhost:6333/dashboard"
echo ""
echo "📝 Useful Commands:"
echo "   • View logs:       docker-compose logs -f"
echo "   • Stop services:   docker-compose down"
echo "   • Restart:         docker-compose restart"
echo "   • Check status:    docker-compose ps"
echo ""

# Open browser to frontend UI
echo "🌐 Opening frontend in browser..."
sleep 2

# Detect OS and open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:8501
    elif command -v gnome-open > /dev/null; then
        gnome-open http://localhost:8501
    else
        echo "   ℹ️  Please manually open: http://localhost:8501"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash or Cygwin)
    start http://localhost:8501
else
    echo "   ℹ️  Please manually open: http://localhost:8501"
fi

echo ""
echo "🎉 Ready to use! The UI should open automatically."
echo "   If not, visit: http://localhost:8501"
echo ""
echo "=========================================="
