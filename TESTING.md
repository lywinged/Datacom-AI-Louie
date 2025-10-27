# Testing Guide

## Problem: Double Backend Directory Structure

This project has a nested `backend/backend/` structure:

```
backend/                    # Docker build context
├── main.py                # Entry point for Docker
├── requirements.txt
├── Dockerfile
└── backend/              # Actual Python package
    ├── __init__.py
    ├── services/
    ├── routers/
    └── ...
```

This causes import path conflicts:
- **Docker**: Uses `from backend.xxx import` (working directory is `/app`)
- **pytest**: Would need `from backend.backend.xxx import` if run from project root

## Solution: conftest.py Path Manipulation

The `tests/conftest.py` file **already solves this issue** using Python's module system:

```python
# Lines 19-23 in tests/conftest.py
INNER_BACKEND_PATH = PROJECT_ROOT / "backend" / "backend"
if "backend" not in sys.modules:
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(INNER_BACKEND_PATH)]
    sys.modules["backend"] = backend_pkg
```

This creates a virtual `backend` module that points to the inner directory, allowing tests to use the same `from backend.xxx` imports as Docker.

## Running Tests Locally

### 1. Install Dependencies

```bash
# From project root
pip install -r backend/requirements.txt
```

This installs all dependencies including:
- fastapi, uvicorn (core framework)
- pytest, pytest-asyncio, pytest-cov, pytest-mock (testing)
- All other backend dependencies

### 2. Run pytest

```bash
# From project root (ai-assessment-deploy/)
python -m pytest tests/ -v
```

The `-v` flag shows verbose output. Other useful options:

```bash
# Quick mode (minimal output)
pytest tests/ -q

# Run specific test file
pytest tests/test_essentials.py -v

# Show coverage report
pytest tests/ --cov=backend --cov-report=html

# Stop on first failure
pytest tests/ -x

# Run with detailed output on failures
pytest tests/ -vv --tb=long
```

### 3. Expected Output

```
============================= test session starts ==============================
platform darwin -- Python 3.10.x, pytest-7.4.3
collected 15 items

tests/test_essentials.py::test_health_endpoint PASSED                     [  6%]
tests/test_essentials.py::test_rag_config PASSED                          [ 13%]
tests/test_essentials.py::test_rag_ask PASSED                             [ 20%]
tests/test_essentials.py::test_chat_basic PASSED                          [ 26%]
tests/test_essentials.py::test_agent_plan PASSED                          [ 33%]
tests/test_essentials.py::test_code_generate PASSED                       [ 40%]
...

======================== 15 passed in 2.35s ================================
```

## Running Tests in CI/CD

### GitHub Actions

Use the provided `.github/workflows/ci.yml` configuration:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r backend/requirements.txt

- name: Run pytest
  run: python -m pytest tests/ -v
  env:
    PYTHONPATH: .
    TESTING: true
```

The `conftest.py` handles all path manipulation automatically.

### GitLab CI

```yaml
test:
  stage: test
  image: python:3.10-slim
  script:
    - pip install -r backend/requirements.txt
    - pytest tests/ -v
  variables:
    PYTHONPATH: "."
    TESTING: "true"
```

### Other CI Systems

Key requirements:
1. Install Python 3.10+
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Run from project root: `pytest tests/ -v`
4. Set `PYTHONPATH=.` and `TESTING=true` environment variables

## How conftest.py Works

### Module System Manipulation

```python
# Create a virtual "backend" module
backend_pkg = types.ModuleType("backend")

# Point it to the inner backend/ directory
backend_pkg.__path__ = [str(INNER_BACKEND_PATH)]

# Register it globally
sys.modules["backend"] = backend_pkg
```

Now when tests do `from backend.services import X`, Python:
1. Checks `sys.modules["backend"]` → finds our virtual module
2. Uses its `__path__` → points to `backend/backend/`
3. Imports successfully without needing `from backend.backend.xxx`

### Heavy Dependency Stubbing

The `conftest.py` also stubs out heavy dependencies to speed up tests:

- **onnxruntime**: Native ONNX inference (100+ MB, slow to load)
- **transformers**: HuggingFace tokenizers (100+ MB models)
- **qdrant-client**: Vector database client (native libraries)

This allows tests to run quickly without downloading models or loading native libraries.

## Why Not Change the Structure?

### Option 1: Flatten to single backend/
**Problem**: Would break Docker build context and require restructuring

### Option 2: Change all imports to backend.backend.xxx
**Problem**: Would break Docker (imports would be wrong inside container)

### Option 3: Use conftest.py (CURRENT SOLUTION ✅)
**Advantage**:
- No code changes needed
- Works in both Docker and pytest
- Standard pytest pattern
- Already implemented

## Verifying the Solution

### Check Import Path

```bash
python -c "import sys; sys.path.insert(0, '.'); from tests.conftest import *; import backend; print(backend.__path__)"
# Output: ['/path/to/backend/backend']
```

### Test Imports

```python
# This works in tests thanks to conftest.py:
from backend.services.qdrant_seed import ensure_seed_collection
from backend.models.chat_schemas import ChatMessage
from backend.routers.rag_routes import router
```

## Common Issues

### Issue: ModuleNotFoundError: No module named 'fastapi'
**Solution**: Install dependencies first: `pip install -r backend/requirements.txt`

### Issue: ModuleNotFoundError: No module named 'backend'
**Solution**: Run pytest from project root, not from tests/ directory

### Issue: Import errors in Docker
**Solution**: Docker uses different working directory - imports work correctly there

### Issue: Circular imports or sys.modules conflicts
**Solution**: conftest.py only creates the module if it doesn't exist (line 20: `if "backend" not in sys.modules`)

## Summary

✅ **No code changes needed** - conftest.py handles everything
✅ **Works in CI/CD** - Just install dependencies and run pytest
✅ **Fast tests** - Heavy dependencies are stubbed
✅ **Same imports everywhere** - `from backend.xxx` works in Docker and tests
✅ **Already implemented** - Solution is in tests/conftest.py lines 19-23

Your project is **ready for CI/CD** right now. Just ensure the CI environment:
1. Installs `backend/requirements.txt`
2. Runs `pytest tests/` from project root
3. Sets `TESTING=true` environment variable
