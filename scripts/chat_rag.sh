#!/bin/bash
# Interactive RAG Chat System
# Usage: bash scripts/chat_rag.sh

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

echo "======================================================================================================"
echo "> RAG J)ûß (Interactive RAG Chat)"
echo "======================================================================================================"
echo ""
echo "This chat system allows you to ask questions about the document collection."
echo "Using INT8-quantized BGE-M3 embeddings + BGE-Reranker for fast semantic search."
echo ""
echo "======================================================================================================"
echo ""

# Run the chat
python chat_rag.py
