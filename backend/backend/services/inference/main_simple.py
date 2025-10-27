"""
Inference service (simple variant without heavy model dependencies).

Purpose:
- Exercise the API contract without downloading large models
- Provide deterministic mock embeddings for development and testing
- Run in constrained environments without GPU support

Use `main.py` in production environments.
"""
import time
import random
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# === Pydantic Models ===
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(None, description="Model name (ignored in simple mode)")
    normalize: bool = Field(True, description="Whether to normalize vectors")
    batch_size: Optional[int] = Field(None, description="Batch size override")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    processing_time_ms: float
    batch_info: dict


class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="Documents to rerank")
    model: Optional[str] = Field(None, description="Model name (ignored)")
    top_k: Optional[int] = Field(None, description="Return the top-k results")
    return_documents: bool = Field(False, description="Include document text in response")


class RerankResponse(BaseModel):
    scores: List[float]
    indices: Optional[List[int]] = None
    documents: Optional[List[str]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    mode: str
    message: str


# === FastAPI App ===
app = FastAPI(
    title="Inference Worker Service (Simple Mode)",
    description="Simplified inference service for testing purposes",
    version="0.1.0-simple"
)

startup_time = time.time()


def generate_random_embedding(text: str, dimension: int = 1024) -> List[float]:
    """Generate a deterministic pseudo-random embedding for testing."""
    # Use the text hash as the seed so identical texts map to the same vector
    random.seed(hash(text) % (2**32))
    vector = [random.gauss(0, 1) for _ in range(dimension)]

    # Normalize to unit length
    norm = sum(x**2 for x in vector) ** 0.5
    return [x / norm for x in vector]


def compute_similarity(query: str, doc: str) -> float:
    """Compute a toy similarity score based on character overlap."""
    # Character overlap heuristic for demonstration only
    query_chars = set(query.lower())
    doc_chars = set(doc.lower())

    if not query_chars or not doc_chars:
        return 0.0

    intersection = len(query_chars & doc_chars)
    union = len(query_chars | doc_chars)

    return intersection / union if union > 0 else 0.0


# === API Endpoints ===
@app.get("/", tags=["Meta"])
async def root():
    """Service root endpoint."""
    return {
        "service": "Inference Worker (Simple Mode)",
        "version": "0.1.0-simple",
        "status": "running",
        "mode": "mock",
        "message": "This is a simplified version for testing. Returns random embeddings.",
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "rerank": "/rerank"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        mode="simple",
        message="Running in simple mode (mock embeddings)"
    )


@app.get("/ready", tags=["Meta"])
async def readiness_check():
    """Readiness probe."""
    return {"status": "ready", "mode": "simple"}


@app.post("/embed", response_model=EmbedResponse, tags=["Inference"])
async def embed_texts(request: EmbedRequest):
    """Return deterministic random embeddings for supplied texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    start_time = time.perf_counter()

    # Produce mock embeddings
    embeddings = [generate_random_embedding(text) for text in request.texts]

    duration = (time.perf_counter() - start_time) * 1000

    return EmbedResponse(
        embeddings=embeddings,
        model="mock-bge-m3",
        dimension=1024,
        processing_time_ms=round(duration, 2),
        batch_info={
            "total_texts": len(request.texts),
            "batch_size": request.batch_size or 32,
            "num_batches": 1
        }
    )


@app.post("/rerank", response_model=RerankResponse, tags=["Inference"])
async def rerank_documents(request: RerankRequest):
    """Rerank documents using the toy character overlap heuristic."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents cannot be empty")

    start_time = time.perf_counter()

    # Compute toy similarity scores
    scores = [compute_similarity(request.query, doc) for doc in request.documents]

    # Sort descending by score
    scored_docs = list(enumerate(scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Truncate to top_k results
    top_k = request.top_k or len(request.documents)
    scored_docs = scored_docs[:top_k]

    indices = [idx for idx, score in scored_docs]
    sorted_scores = [score for idx, score in scored_docs]

    duration = (time.perf_counter() - start_time) * 1000

    result = RerankResponse(
        scores=sorted_scores,
        indices=indices,
        processing_time_ms=round(duration, 2)
    )

    if request.return_documents:
        result.documents = [request.documents[i] for i in indices]

    return result


# === Main Entry ===
if __name__ == "__main__":
    uvicorn.run(
        "inference_service.main_simple:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
