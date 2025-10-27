#!/bin/bash
# Interactive Code Generation Chat
# Usage: bash scripts/chat_code.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================================================================================"
echo "Code Generation Assistant (Task 3.4)"
echo "======================================================================================================"
echo ""
echo "Self-healing code generation with automated testing:"
echo "  ✓ Generate code from natural language"
echo "  ✓ Automatic testing (pytest, cargo test, npm test)"
echo "  ✓ Self-healing on failures (up to 3 retries)"
echo "  ✓ Supports: Python, Rust, JavaScript, Go"
echo ""
echo "======================================================================================================"
echo ""

python chat_code.py
