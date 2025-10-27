#!/bin/bash

# Unset host environment variables that should use container names/paths in Docker
# These variables may be set in your shell for local development,
# but Docker containers need different values (container names, absolute paths)
unset QDRANT_HOST
unset QDRANT_SEED_PATH

echo "🐳 Starting Docker Compose with clean environment..."
echo "   QDRANT_HOST will use default: qdrant (container name)"
echo "   QDRANT_SEED_PATH will use default: /app/data/qdrant_seed/assessment_docs_minilm.jsonl"

# Start Docker Compose
docker-compose up -d "$@"

echo ""
echo "✅ Containers started. Check status with:"
echo "   docker-compose ps"
echo ""
echo "📊 Monitor Qdrant seeding:"
echo "   curl -s http://localhost:8888/api/rag/seed-status | python3 -m json.tool"
