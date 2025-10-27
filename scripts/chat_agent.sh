#!/bin/bash
# Interactive Chat with Planning Agent
# Usage: bash scripts/chat_agent.sh

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# Set environment
export PYTHONPATH=.

# Run the chat interface
python chat_agent.py
