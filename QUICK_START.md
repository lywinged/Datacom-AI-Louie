# 🚀 Quick Start Guide - AI Assessment Project

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- OpenAI or Azure OpenAI API key

## ⚡ 5-Minute Setup

### Step 1: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# Required: OPENAI_API_KEY=sk-your-key-here
# OR: Azure OpenAI credentials
nano .env
```

### Step 2: Copy Models (Important!)

```bash
# The ONNX models are NOT included in this repository due to size
# Copy from the original project:
cp -r /path/to/original/models ./models/

# You need these two models:
# - models/bge-m3-embed-int8/
# - models/bge-reranker-int8/
```

### Step 3: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Watch logs (optional)
docker-compose logs -f
```

### Step 4: Wait for Services to be Ready

```bash
# Check service status (all should show "healthy")
docker-compose ps

# Expected output:
# NAME              STATUS
# qdrant            Up (healthy)
# backend-api       Up (healthy)
# streamlit-ui      Up (healthy)
```

### Step 5: Access the Application

Open your browser and navigate to:

- **Streamlit UI**: http://localhost:8501
- **Backend API Docs**: http://localhost:8888/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 🧪 Run Tests

```bash
# Run all tests
docker-compose exec backend pytest -q tests/

# Expected output:
# test_rag_api.py .......         [30%]
# test_agent_api.py ........      [65%]
# test_code_api.py ........       [100%]
# =============== 23 passed ===============
```

## 🎯 Quick Test Examples

### Test RAG Q&A API
```bash
curl -X POST http://localhost:8888/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Gandalf?", "top_k": 3}'
```

### Test Trip Planning Agent
```bash
curl -X POST http://localhost:8888/api/agent/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo",
    "origin": "Auckland",
    "days": 5,
    "budget": 2000,
    "currency": "USD"
  }'
```

### Test Code Generation
```bash
curl -X POST http://localhost:8888/api/code/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a function to check if a number is prime",
    "language": "python"
  }'
```

## 📊 Initial Data Setup (Optional)

If you want to ingest documents for RAG:

```bash
# Copy sample data
cp -r /path/to/original/data/gutenberg_corpus_5mb ./data/

# Run ingestion inside container
docker-compose exec backend python scripts/ingest_serial_optimized.py \
  --docs-dir /app/data/gutenberg_corpus_5mb \
  --collection gutenberg_corpus
```

## 🛠️ Troubleshooting

### Services won't start
```bash
# Check logs for specific service
docker-compose logs backend
docker-compose logs qdrant

# Restart all services
docker-compose restart
```

### "Connection refused" errors
```bash
# Ensure services are healthy
docker-compose ps

# Wait a bit longer (services may still be initializing)
sleep 30 && docker-compose ps
```

### Tests failing
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Check backend health
curl http://localhost:8888/health

# Run tests with verbose output
docker-compose exec backend pytest tests/ -v -s
```

### Out of memory
```bash
# Check Docker memory limit (Docker Desktop > Settings > Resources)
# Recommended: 8GB+ for Docker

# Reduce OMP_NUM_THREADS in docker-compose.yml:
# OMP_NUM_THREADS=6  # instead of 16
```

## 🧹 Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📁 Project Structure

```
ai-assessment-deploy/
├── docker-compose.yml      # Service orchestration
├── .env                    # Your secrets (DO NOT COMMIT)
├── .env.example            # Environment template
├── README.md               # Full documentation
├── REPORT.md               # Design decisions
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py             # Entry point
│   └── backend/            # Application code
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py              # Streamlit dashboard
├── tests/
│   ├── test_rag_api.py
│   ├── test_agent_api.py
│   └── test_code_api.py
├── models/                 # ONNX models (YOU NEED TO COPY THESE)
├── data/                   # Training data (optional)
└── scripts/                # Utility scripts
```

## ✅ Success Checklist

- [ ] Environment variables configured in `.env`
- [ ] ONNX models copied to `models/` directory
- [ ] All services started: `docker-compose up -d`
- [ ] All services healthy: `docker-compose ps`
- [ ] Streamlit UI accessible at http://localhost:8501
- [ ] Backend API docs at http://localhost:8888/docs
- [ ] All tests passing: `docker-compose exec backend pytest -q tests/`

## 🆘 Get Help

1. Check [README.md](README.md) for detailed documentation
2. Check [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) for complete setup guide
3. Review [REPORT.md](REPORT.md) for design decisions
4. Check Docker logs: `docker-compose logs <service-name>`

## 🎉 You're Ready!

Once all services are healthy and tests pass, you can:

1. **Use the Streamlit UI** at http://localhost:8501 for interactive access
2. **Call the APIs** directly for programmatic access
3. **Run the test clients** (chat_agent.py, chat_code.py, chat_rag.py) from host

Enjoy your AI-powered application! 🚀
