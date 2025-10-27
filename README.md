# AI Assessment Project - Production Deployment

A comprehensive AI system featuring **RAG Q&A**, **intelligent trip planning**, and **self-healing code generation**, optimized for production deployment with ONNX quantization, in-memory vector search, and Docker containerization.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [System Components](#system-components)
4. [Setup Guide](#setup-guide)
5. [Running Automated Tests](#running-automated-tests)
6. [Performance Benchmarks](#performance-benchmarks)
7. [API Documentation](#api-documentation)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (Port 18501)              │
│         Intent Classification & Multi-Service Dashboard         │
└───────────────────────────┬────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend (Port 8888)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  RAG Routes  │  │ Agent Routes │  │   Code Routes      │   │
│  │  (Task 3.2)  │  │  (Task 3.3)  │  │    (Task 3.4)      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────────────┘   │
│         │                 │                  │                  │
│  ┌──────▼─────────────────▼──────────────────▼─────────────┐   │
│  │           ONNX Inference Engine (INT8)                   │   │
│  │  BGE-M3 Embeddings + BGE/MiniLM Reranker                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│             Qdrant Vector DB (Port 6333)                        │
│       In-Memory Mode + Persistent Disk Backup                   │
│         138,146 vectors (Project Gutenberg Corpus)              │
└────────────────────────────────────────────────────────────────┘
```

### Key Features

- **ONNX INT8 Quantization**: 2-3x faster inference, 4x smaller models
- **In-Memory Vector Search**: 40-60% latency reduction (20-60ms vs 50-120ms)
- **Hybrid Information Extraction**: Regex-first (80% cases) + LLM fallback
- **AutoPlan Learning System**: Experience replay with session persistence
- **Self-Healing Code Generation**: Iterative refinement with automated testing
- **Production-Ready**: Docker Compose orchestration with health checks

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose** installed
- **8GB+ RAM** (16GB recommended for optimal performance)
- **OpenAI API Key** (or Azure OpenAI credentials)

### 1-Minute Setup

```bash
# Clone repository
cd ai-assessment-deploy

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all services
docker-compose up -d

# Wait for services to initialize (~2 minutes)
# Monitor startup progress:
docker-compose logs -f backend

# Access the application
open http://localhost:18501
```

### Verify Installation

```bash
# Check all services are healthy
docker-compose ps

# Test backend health
curl http://localhost:8888/health

# Test Qdrant collection
curl http://localhost:6333/collections/assessment_docs

# Run quick API test
python3 scripts/eval_rag_api_random30.py --seed 42
```

Expected output: Top-5 accuracy ~85%, median latency ~160ms

---

## System Components

### 1. RAG Q&A System (Task 3.2)

**Features:**
- **Embedding Model**: BGE-M3 (INT8 quantized, 330MB)
- **Vector Database**: Qdrant with 138k document chunks
- **Reranking**: BGE Reranker (primary) + MiniLM (fallback)
- **Answer Generation**: OpenAI GPT-3.5-turbo / GPT-4

**Performance:**
- Embed Time: 30-60ms (warm)
- Vector Search: 20-60ms (in-memory)
- Rerank Time: 60-100ms
- Total Retrieval: 150-250ms

**API Endpoint:**
```bash
POST /api/rag/ask
{
  "question": "Who wrote The Great Gatsby?",
  "top_k": 5,
  "include_timings": true
}
```

### 2. Trip Planning Agent (Task 3.3)

**Features:**
- **AutoPlan Learning**: Experience replay with signature matching
- **Session Persistence**: SQLite-backed conversation history
- **Hybrid Extraction**: Regex (80%) + LLM fallback (20%)
- **Tool Integration**: Mock APIs for flights, hotels, attractions

**API Endpoint:**
```bash
POST /api/agent/plan
{
  "query": "Plan a 3-day trip to Paris",
  "session_id": "optional-session-id"
}
```

### 3. Code Generation Assistant (Task 3.4)

**Features:**
- **Automated Testing**: Generates and executes unit tests
- **Iterative Refinement**: Up to 3 retry attempts
- **Multi-Language**: Python, JavaScript, Java support
- **Assertion Injection**: Validates output correctness

**API Endpoint:**
```bash
POST /api/code/generate
{
  "task_description": "Write a function to check if a number is prime",
  "language": "python"
}
```

### 4. Streamlit Dashboard

**Features:**
- Intent classification (RAG / Agent / Code)
- Real-time service status indicators
- Session management with history
- Responsive multi-tab interface

---

## Setup Guide

### Detailed Installation

#### 1. Environment Configuration

Create `.env` file with required settings:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo-0125

# Or use Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo

# Qdrant Settings
QDRANT_COLLECTION=assessment_docs

# Performance Tuning
OMP_NUM_THREADS=16  # Backend CPU threads
USE_INT8_QUANTIZATION=true
```

#### 2. Model Preparation

Models are included in the `models/` directory:

```
models/
├── bge-m3-embed-int8/       # 2.6GB (542MB ONNX + tokenizer)
├── bge-reranker-int8/       # 1.3GB (266MB ONNX + tokenizer)
└── minilm-reranker-onnx/    # 110MB (22MB ONNX + tokenizer)
```

If models are missing, download from:
```bash
# Option 1: Use provided models (recommended)
# Already included in deployment package

# Option 2: Convert from HuggingFace (advanced)
python3 scripts/convert_to_onnx.py --model BAAI/bge-m3 --output models/bge-m3-embed-int8
```

#### 3. Data Seeding

On first startup, backend automatically:
1. Creates Qdrant collection `assessment_docs`
2. Uploads 138,146 vectors from `data/qdrant_seed/assessment_docs.jsonl`
3. Builds HNSW index for fast similarity search

**Initial seed time**: ~10 minutes
**Subsequent restarts**: 1-2 minutes (loads from disk to memory)

#### 4. Service Startup Order

Docker Compose ensures correct startup sequence:

```yaml
1. Qdrant       (6333)    # Vector database
2. Inference    (8001)    # ONNX model server (optional)
3. Backend      (8888)    # FastAPI + seed upload
4. Frontend     (18501)   # Streamlit UI
```

Monitor logs during startup:
```bash
docker-compose logs -f backend | grep -E "(seed|Ready|Startup)"
```

---

## Running Automated Tests

### Unit & Integration Tests

**Run all pytest tests:**
```bash
# Inside backend container
docker exec backend-api pytest tests/ -v

# Or from host (requires local dependencies)
cd backend
pytest tests/ -v --cov=backend --cov-report=html
```

**Test coverage:**
- `test_rag_routes.py`: RAG API endpoints
- `test_agent_routes.py`: Trip planning agent
- `test_code_routes.py`: Code generation
- `test_chat_routes.py`: Chat session management
- `test_rag_pipeline_utils.py`: Core retrieval logic

**Expected results:**
```
tests/test_rag_routes.py::test_rag_health .......................... PASSED
tests/test_rag_routes.py::test_rag_ask_basic ....................... PASSED
tests/test_agent_routes.py::test_agent_plan ........................ PASSED
tests/test_code_routes.py::test_code_generate ...................... PASSED

========================== 12 passed in 8.5s ===========================
```

### RAG Accuracy Evaluation

**Full evaluation (300 questions):**
```bash
# From backend directory (slow: ~15 minutes)
python3 scripts/eval_rag_accuracy_warmup.py --reranker minilm

# API-based evaluation (requires Docker backend running)
python3 scripts/eval_rag_api.py
```

**Quick evaluation (60 questions, 20 per category):**
```bash
# Recommended for CI/CD pipelines
python3 scripts/eval_rag_api_random30.py --seed 42 --output results.json

# Custom random seed
python3 scripts/eval_rag_api_random30.py --seed 123
```

**Expected metrics:**
```json
{
  "top5_accuracy_percent": 85.0,
  "top1_accuracy_percent": 65.0,
  "latency": {
    "mean_ms": 162.3,
    "median_ms": 155.5,
    "p95_ms": 225.8
  },
  "category_breakdown": {
    "metadata": { "top5_accuracy_percent": 90.0 },
    "keyword":  { "top5_accuracy_percent": 82.5 },
    "semantic": { "top5_accuracy_percent": 82.5 }
  }
}
```

### Agent Learning System Test

```bash
# Test learning persistence and retrieval
bash scripts/test_learning.sh

# Expected: Session history preserved across requests
```

### Performance Benchmarks

**RAG retrieval latency test:**
```bash
python3 scripts/test_rag_retrieval_only.py

# Measures: embed_ms, vector_ms, rerank_ms, total_ms
```

**Embedding speed with different batch sizes:**
```bash
python3 scripts/debug_embed_speed.py

# Output:
# batch_size=32 -> 118.3ms total, 3.7ms per text
```

---

## Performance Benchmarks

### RAG System Performance

| Metric | Cold Start | After Warm-up | Target |
|--------|-----------|---------------|---------|
| Embed Time | 700-1500ms | 30-60ms | <100ms |
| Vector Search (disk) | 80-150ms | 50-120ms | <100ms |
| Vector Search (memory) | 40-80ms | 20-60ms | <80ms ✅ |
| Rerank Time | 300-500ms | 60-100ms | <150ms |
| **Total Retrieval** | 1500-2500ms | 150-250ms | <300ms ✅ |

### Accuracy Metrics (300-question benchmark)

| Dataset | Top-1 | Top-5 | Notes |
|---------|-------|-------|-------|
| Metadata | 70% | 90% | Author/title queries |
| Keyword | 60% | 82% | Exact phrase matching |
| Semantic | 65% | 85% | Conceptual similarity |
| **Overall** | **65%** | **85%** | MiniLM reranker |

### Resource Usage

| Component | CPU | Memory | Disk | Notes |
|-----------|-----|--------|------|-------|
| Backend | 30-50% | 2.5GB | 500MB | 16 threads |
| Qdrant (in-memory) | 10-20% | 2.0GB | 3.5GB | 138k vectors |
| Frontend | 5-10% | 300MB | 100MB | Streamlit |
| **Total** | **50-80%** | **5GB** | **4GB** | Single host |

---

## API Documentation

### RAG Q&A Endpoints

**Ask Question**
```http
POST /api/rag/ask
Content-Type: application/json

{
  "question": "What is machine learning?",
  "top_k": 5,
  "include_timings": true,
  "reranker": "fallback",          // "fallback" or "bge"
  "vector_limit": 10,               // Pre-rerank candidates
  "content_char_limit": 500         // Max chunk size
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "chunks": [
    {
      "content": "Machine learning...",
      "source": "Introduction_to_ML.txt",
      "score": 0.89,
      "metadata": {
        "title": "Introduction to ML",
        "authors": "John Doe"
      }
    }
  ],
  "timings": {
    "embed_ms": 42.3,
    "vector_ms": 35.1,
    "rerank_ms": 68.5,
    "llm_ms": 1200.0,
    "total_ms": 1346.2
  }
}
```

**Health Check**
```http
GET /api/rag/health

Response: { "status": "healthy", "qdrant_connected": true }
```

### Agent Endpoints

**Plan Trip**
```http
POST /api/agent/plan
Content-Type: application/json

{
  "query": "Plan a romantic 3-day trip to Paris for 2 people",
  "session_id": "user123"
}
```

**Response:**
```json
{
  "plan": {
    "destination": "Paris",
    "duration_days": 3,
    "budget_usd": 2500,
    "itinerary": [...]
  },
  "learning_used": true,
  "session_id": "user123_20251027"
}
```

### Code Generation Endpoints

**Generate Code**
```http
POST /api/code/generate
Content-Type: application/json

{
  "task_description": "Write a function to check if a number is prime",
  "language": "python",
  "max_retries": 3
}
```

**Response:**
```json
{
  "code": "def is_prime(n):\n    ...",
  "tests_passed": true,
  "iterations": 1,
  "test_output": "All tests passed ✓"
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | **Required**: OpenAI API key |
| `OPENAI_MODEL` | gpt-3.5-turbo-0125 | LLM model name |
| `QDRANT_HOST` | qdrant | Qdrant hostname (Docker) |
| `QDRANT_PORT` | 6333 | Qdrant HTTP port |
| `QDRANT_COLLECTION` | assessment_docs | Collection name |
| `USE_ONNX_INFERENCE` | true | Enable ONNX models |
| `USE_INT8_QUANTIZATION` | true | INT8 quantization |
| `OMP_NUM_THREADS` | 16 | CPU parallelism |
| `ENABLE_REMOTE_INFERENCE` | false | Use separate inference service |

### Docker Compose Configuration

**Qdrant In-Memory Mode** (recommended for performance):
```yaml
qdrant:
  environment:
    - QDRANT__STORAGE__IN_MEMORY=true
    - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
  volumes:
    - qdrant_storage:/qdrant/storage  # Disk backup
```

**Backend CPU Optimization:**
```yaml
backend:
  environment:
    - OMP_NUM_THREADS=16  # Adjust based on CPU cores
```

---

## Troubleshooting

### Common Issues

**1. Services not starting**
```bash
# Check logs for errors
docker-compose logs backend
docker-compose logs qdrant

# Restart services
docker-compose restart
```

**2. Qdrant collection empty after restart**
```bash
# Check if seed file exists
ls -lh data/qdrant_seed/assessment_docs.jsonl  # Should be ~2GB

# Manually trigger seed upload
docker-compose restart backend

# Monitor upload progress
docker logs backend-api -f | grep seed
```

**3. Slow embedding performance**
```bash
# Check if models are being reloaded
docker logs backend-api | grep "CUDA not available"

# Verify batch_size=32 in onnx_inference.py:108
docker exec backend-api grep -A 2 "def encode" /app/backend/services/onnx_inference.py
```

**4. High reranker latency (>500ms)**
```bash
# Switch to fallback reranker
curl -X POST http://localhost:8888/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "reranker": "fallback"}'
```

**5. Out of memory errors**
```bash
# Check Docker memory limit
docker stats

# Increase Docker memory allocation (Docker Desktop settings)
# Recommended: 8GB minimum, 16GB optimal

# Or disable in-memory Qdrant mode
# Edit docker-compose.yml: Remove QDRANT__STORAGE__IN_MEMORY=true
```

### Performance Tuning

**Reduce latency:**
1. Ensure Qdrant in-memory mode is enabled
2. Use `reranker: "fallback"` for faster reranking
3. Reduce `vector_limit` to 5-10 for fewer candidates
4. Increase `OMP_NUM_THREADS` to match CPU cores

**Improve accuracy:**
1. Use `reranker: "bge"` for better quality
2. Increase `vector_limit` to 20-30 for more candidates
3. Increase `top_k` to retrieve more chunks

---

## Project Structure

```
ai-assessment-deploy/
├── backend/
│   ├── Dockerfile
│   ├── backend/
│   │   ├── main.py                    # FastAPI application
│   │   ├── routers/
│   │   │   ├── rag_routes.py          # RAG endpoints
│   │   │   ├── agent_routes.py        # Trip planning
│   │   │   └── code_routes.py         # Code generation
│   │   ├── services/
│   │   │   ├── rag_pipeline.py        # Core RAG logic
│   │   │   ├── onnx_inference.py      # ONNX model wrapper
│   │   │   ├── qdrant_client.py       # Vector DB client
│   │   │   └── qdrant_seed.py         # Auto-seeding
│   │   └── config/
│   │       └── settings.py            # Configuration
├── frontend/
│   ├── Dockerfile
│   └── app.py                         # Streamlit dashboard
├── data/
│   ├── qdrant_seed/
│   │   └── assessment_docs.jsonl      # 138k vectors (2GB)
│   ├── rag_eval_metadata.json         # 100 metadata questions
│   ├── rag_eval_keyword.json          # 100 keyword questions
│   └── rag_eval_semantic.json         # 100 semantic questions
├── models/
│   ├── bge-m3-embed-int8/             # Embedding model
│   ├── bge-reranker-int8/             # Primary reranker
│   └── minilm-reranker-onnx/          # Fallback reranker
├── scripts/
│   ├── eval_rag_api_random30.py       # Quick 60-question eval
│   ├── eval_rag_accuracy_warmup.py    # Full 300-question eval
│   ├── test_agent.sh                  # Agent system test
│   └── debug_embed_speed.py           # Performance debugging
├── tests/
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_rag_routes.py             # RAG API tests
│   ├── test_agent_routes.py           # Agent API tests
│   └── test_code_routes.py            # Code API tests
├── docker-compose.yml                 # Service orchestration
├── .env                               # Environment config
└── README.md                          # This file
```

---

## License & Credits

**Project**: AI Assessment System
**Models**: BAAI/bge-m3, BAAI/bge-reranker-base, sentence-transformers/all-MiniLM-L6-v2
**Dataset**: Project Gutenberg (138,146 book excerpts)
**Technologies**: FastAPI, Streamlit, Qdrant, ONNX Runtime, Docker

---

## Additional Resources

- [QUICK_START.md](QUICK_START.md) - Minimal setup guide
- [REPORT.md](REPORT.md) - Design decisions and trade-offs
- [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) - Production deployment notes
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

**For questions or issues, please check the troubleshooting section or review Docker logs.**
