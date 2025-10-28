# AI Assessment Project

Enterprise-grade AI platform with Chat, RAG, Agent, and Code Assistant capabilities.

## Features

- **Task 3.1:** Conversational Chat with streaming 
- **Task 3.2:** High-Performance RAG QA with Qdrant vector database
- **Task 3.3:** Autonomous Planning Agent for trip planning
- **Task 3.4:** Self-Healing Code Assistant with automated testing

## Architecture

```
┌─────────────────┐
│   Frontend      │  Streamlit UI (Port 8501)
│   (Streamlit)   │  - Chat interface
└────────┬────────┘  - RAG Q&A
         │           - Agent planning
         ▼           - Code generation
┌─────────────────┐
│   Backend API   │  FastAPI (Port 8888)
│   (FastAPI)     │  - RESTful endpoints
└────────┬────────┘  - Request validation
         │           - Business logic
    ┌────┴────┬──────────┬─────────┐
    ▼         ▼          ▼         ▼
┌────────┐ ┌─────┐  ┌────────┐ ┌─────────┐
│ Qdrant │ │ LLM │  │ ONNX   │ │ SQLite  │
│ Vector │ │Azure│  │Inference│ │ Session │
│   DB   │ │OpenAI  │Models  │ │   DB    │
└────────┘ └─────┘  └────────┘ └─────────┘
 152K pts  GPT-3.5    MiniLM     Chat Hist
```

### Technology Stack

**Backend:**
- FastAPI 0.104 - High-performance async API framework
- Qdrant 1.15 - Vector database for semantic search
- ONNX Runtime 1.16 - Optimized embedding/reranking inference
- Azure OpenAI - GPT-4 for chat and generation
- SQLAlchemy 2.0 - Database ORM

**Frontend:**
- Streamlit 1.29 - Interactive web interface
- Plotly 5.18 - Data visualization

**Infrastructure:**
- Docker Compose - Container orchestration
- GitHub Actions - CI/CD automation
- pytest 7.4 - Testing framework

## Setup

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- Azure OpenAI API key

### Quick Start (Docker - Recommended)

1. **Configure environment variables:**

```bash
cp .env
# I left my own API key here, because yours doesn't work and there is no response from you at all 😂
# Edit .env with your Azure OpenAI credentials:
# AZURE_OPENAI_ENDPOINT=your-endpoint
# AZURE_OPENAI_KEY=your-key
# OPENAI_MODEL=gpt-4
```

2. **Start all services:**

```bash
# 🔥 All in one, easy to play 🔥
🌟🌟🌟🌟🌟🌟🌟🌟 

bash start.sh

🌟🌟🌟🌟🌟🌟🌟🌟 



# Or use docker-compose directly
docker-compose up -d
```

3. **Verify services are running:**

```bash
# Check API health
curl http://localhost:8888/health

# Check Qdrant
curl http://localhost:6333
```

4. **Access the application:**

- Frontend: http://localhost:8501
- Backend API: http://localhost:8888
- API Docs: http://localhost:8888/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### Local Development Setup

1. **Install dependencies:**

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

2. **Start services manually:**

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# Terminal 2: Start frontend
cd frontend
streamlit run app.py --server.port 8501

# Terminal 3: Start Qdrant (via Docker)
docker run -p 6333:6333 qdrant/qdrant
```

## Running Automated Tests

### Full Test Suite

```bash
# Install test dependencies
pip install -r backend/requirements.txt

# Run all tests
pytest tests/ -v

# Quick mode
pytest tests/ -q

# With coverage report
pytest tests/ --cov=backend --cov-report=html
open htmlcov/index.html
```

### Expected Output

```
============================= test session starts ==============================
collected 29 items

tests/test_agent_routes.py::test_agent_plan_returns_itinerary PASSED     [  3%]
tests/test_agent_routes.py::test_agent_health_reports_stub_model PASSED  [  6%]
tests/test_agent_routes.py::test_agent_metrics_returns_defaults PASSED   [ 10%]
tests/test_app_api.py::test_root_endpoint PASSED                         [ 13%]
tests/test_app_api.py::test_health_endpoint PASSED                       [ 17%]
tests/test_chat_routes.py::test_chat_message_returns_echo_response PASSED [ 20%]
tests/test_chat_routes.py::test_chat_history_and_clear_workflow PASSED   [ 24%]
tests/test_chat_routes.py::test_chat_stream_endpoint PASSED              [ 27%]
tests/test_chat_routes.py::test_chat_metrics_endpoint PASSED             [ 31%]
tests/test_chat_routes.py::test_chat_message_with_stream_false PASSED    [ 34%]
tests/test_code_routes.py::test_code_generate_returns_successful_payload PASSED [ 37%]
tests/test_code_routes.py::test_code_generate_with_test_framework PASSED [ 41%]
tests/test_code_routes.py::test_code_generate_includes_test_result PASSED [ 44%]
tests/test_code_routes.py::test_code_generate_includes_token_usage PASSED [ 48%]
tests/test_code_routes.py::test_code_generate_with_different_language PASSED [ 51%]
tests/test_code_routes.py::test_code_health_reports_stub_model PASSED    [ 55%]
tests/test_code_routes.py::test_code_metrics_returns_defaults PASSED     [ 58%]
tests/test_code_routes.py::test_code_generate_retry_attempts_field PASSED [ 62%]
tests/test_rag_pipeline_utils.py::test_should_not_switch_when_remote_inference_enabled PASSED [ 65%]
tests/test_rag_pipeline_utils.py::test_should_switch_when_latency_exceeds_threshold PASSED [ 68%]
tests/test_rag_pipeline_utils.py::test_should_not_switch_when_gpu_available PASSED [ 72%]
tests/test_rag_routes.py::test_rag_ask_returns_stubbed_response PASSED   [ 75%]
tests/test_rag_routes.py::test_rag_health_success PASSED                 [ 79%]
tests/test_rag_routes.py::test_rag_config_returns_model_info PASSED      [ 82%]
tests/test_rag_routes.py::test_rag_seed_status_returns_status PASSED     [ 86%]
tests/test_rag_routes.py::test_rag_upload_document PASSED                [ 89%]
tests/test_rag_routes.py::test_rag_switch_mode_to_fallback PASSED        [ 93%]
tests/test_rag_routes.py::test_rag_switch_mode_to_primary PASSED         [ 96%]
tests/test_rag_routes.py::test_rag_stats_returns_collection_info PASSED  [100%]

============================== 29 passed in 0.54s ==============================
```

### Test Coverage

- **Total Tests:** 29
- **API Coverage:** 95% (20/21 endpoints)
- **Execution Time:** 0.54s
- **Pass Rate:** 100%

### Test Organization

```
tests/
├── conftest.py              # Fixtures and test configuration
├── test_app_api.py         # Root and health endpoints
├── test_chat_routes.py     # Chat API tests (5 tests)
├── test_agent_routes.py    # Agent API tests (6 tests)
├── test_code_routes.py     # Code Assistant tests (8 tests)
├── test_rag_routes.py      # RAG API tests (9 tests)
└── test_rag_pipeline_utils.py  # RAG utilities (3 tests)
```

### Running Specific Tests

```bash
# Run specific test file
pytest tests/test_rag_routes.py -v

# Run specific test
pytest tests/test_rag_routes.py::test_rag_config_returns_model_info -v

# Run tests matching pattern
pytest tests/ -k "rag" -v

# Stop on first failure
pytest tests/ -x

# Show detailed output on failure
pytest tests/ -vv --tb=long
```

## API Endpoints

### Chat API (`/api/chat`)

- `POST /message` - Send chat message (non-streaming)
- `POST /stream` - Send chat message (streaming SSE)
- `GET /history` - Get conversation history
- `DELETE /history` - Clear conversation history
- `GET /metrics` - Get chat metrics

### RAG API (`/api/rag`)

- `POST /ask` - Ask questions with semantic search and citations
- `POST /upload` - Upload and index documents
- `POST /ingest/sample` - Ingest sample corpus
- `GET /stats` - Get Qdrant collection statistics
- `GET /health` - RAG service health check
- `GET /config` - Get RAG configuration (models, limits)
- `GET /seed-status` - Get Qdrant seed status
- `POST /switch-mode` - Switch between primary/fallback models

### Agent API (`/api/agent`)

- `POST /plan` - Create trip plans with autonomous agent
- `GET /health` - Agent service health check
- `GET /metrics` - Get agent planning metrics

### Code API (`/api/code`)

- `POST /generate` - Generate code with self-healing
- `GET /health` - Code service health check
- `GET /metrics` - Get code generation metrics

## Performance Benchmarks

| Operation | Average Latency | Notes |
|-----------|-----------------|-------|
| RAG Query | ~300ms | With reranking |
| Chat Response | ~500ms | GPT-3.5 streaming |
| Code Generation | 2-5s | With testing & retries |
| Agent Planning | 3-8s | With tool calls |
| Vector Search | ~50ms | 152K points |
| Reranking | ~200ms | 20 candidates |

## Project Structure

```
ai-assessment-deploy/
├── backend/
│   ├── backend/              # Python package
│   │   ├── routers/         # API route handlers
│   │   ├── services/        # Business logic
│   │   ├── models/          # Pydantic schemas
│   │   ├── config/          # Configuration
│   │   └── utils/           # Utilities
│   ├── main.py              # Docker entry point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit application
│   ├── Dockerfile
│   └── requirements.txt
├── tests/                  # Test suite (29 tests)
│   ├── conftest.py         # Test fixtures
│   ├── test_rag_routes.py
│   ├── test_chat_routes.py
│   ├── test_agent_routes.py
│   └── test_code_routes.py
├── data/
│   └── assessment_docs/    # 150 Gutenberg documents
├── models/                 # ONNX models (MiniLM)
├── eval/                   # Evaluation results
├── scripts/               # Utility scripts
├── .github/workflows/     # CI/CD configuration
├── docker-compose.yml     # Container orchestration
├── start.sh              # Clean startup script
└── README.md
```

## Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-key-here
OPENAI_MODEL=gpt-4

# Qdrant Configuration
QDRANT_HOST=qdrant          # Container name (Docker)
# QDRANT_HOST=localhost     # Uncomment for local dev
QDRANT_PORT=6333
QDRANT_COLLECTION=assessment_docs_minilm

# ONNX Inference
USE_ONNX_INFERENCE=true
USE_INT8_QUANTIZATION=true
OMP_NUM_THREADS=16

# Application
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### Model Configuration

- **Embedding Model:** all-MiniLM-L6-v2 (384D, INT8 quantized)
- **Reranker Model:** MiniLM-reranker
- **LLM:** OpenAI GPT-3.5
- **Vector Size:** 384 dimensions
- **Distance Metric:** Cosine similarity

## CI/CD

### GitHub Actions

Automated testing on push/PR:

```yaml
# .github/workflows/ci.yml
- Install dependencies
- Run pytest (29 tests)
- Upload coverage report
```

View workflow: `.github/workflows/ci.yml`

### Running CI Locally

```bash
# Simulate CI environment
PYTHONPATH=. TESTING=true pytest tests/ -v
```

## Troubleshooting

### Common Issues

**1. Qdrant connection refused**

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**2. Tests failing with import errors**

```bash
# Install all dependencies
pip install -r backend/requirements.txt

# Run from project root
cd /path/to/ai-assessment-deploy
pytest tests/ -v
```

**3. Environment variable conflicts**

```bash
# Use the clean startup script
bash start.sh

# This unsets QDRANT_HOST and QDRANT_SEED_PATH
# which may conflict with Docker container names
```

**4. Model loading issues**

```bash
# Check models directory
ls -lh models/minilm-*/

# Ensure ONNX models are present
# If missing, they will be downloaded on first run
```

## Documentation

- **[REPORT.md](REPORT.md)** - Design decisions and trade-offs
- **[TESTING.md](TESTING.md)** - Comprehensive testing guide
- **[TESTS_UPDATED.md](TESTS_UPDATED.md)** - Test update summary
- **[CI_CD_READY.md](CI_CD_READY.md)** - CI/CD configuration details

## Contributing and Credit

1. Great assessment from Datacom
2. My old projects
3. Claude CODE & CODEX


## License

MIT License - See LICENSE file for details

---

**Built with:** FastAPI, Streamlit, Qdrant, ONNX Runtime, Azure OpenAI
**Test Coverage:** 95% (29/29 tests passing)
**Performance:** RAG queries ~450ms | Chat ~500ms
**CI/CD:** GitHub Actions ready
