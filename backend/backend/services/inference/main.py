"""
Inference service main application.

Lightweight API exposing embedding and rerank endpoints.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from inference_service.config import config

# Global model instances loaded at startup
embedding_model = None
rerank_model = None


# === Pydantic Models ===
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(None, description="Model name (defaults to configured model)")
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
    model: Optional[str] = Field(None, description="Model name override")
    top_k: Optional[int] = Field(None, description="Return top-k results")
    return_documents: bool = Field(False, description="Include document text in response")


class RerankResponse(BaseModel):
    scores: List[float]
    indices: Optional[List[int]] = None
    documents: Optional[List[str]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: dict
    config: dict


class ModelsResponse(BaseModel):
    embedding_models: dict
    rerank_models: dict


# === Application Lifespan ===
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global embedding_model, rerank_model

    print("=" * 60)
    print("🚀 Starting Inference Service")
    print("=" * 60)
    print(f"Config: {config.get_config_summary()}")
    print("-" * 60)

    # Ensure dependencies are available
    try:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker
        print("✅ FlagEmbedding library available")
    except ImportError:
        print("⚠️  FlagEmbedding not installed. Install with:")
        print("   pip install FlagEmbedding")
        print("\n   Service will start but models won't be loaded.")
        print("   You can still test the API endpoints.")
        yield
        return

    # Load configured models
    try:
        # 🚀 Optimisation: tune PyTorch thread pools
        import torch
        import os as _os
        num_threads = int(_os.getenv("OMP_NUM_THREADS", "8"))
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"\n⚙️  PyTorch optimization:")
        print(f"   Threads: {num_threads}")
        print(f"   MKLDNN: {torch.backends.mkldnn.is_available()}")

        print(f"\n📦 Loading embedding model: {config.EMBED_MODEL_NAME}")
        print(f"   Device: {config.DEVICE}")

        embedding_model = BGEM3FlagModel(
            config.EMBED_MODEL_NAME,
            use_fp16=config.USE_FP16 and config.DEVICE != "cpu",
            device=config.DEVICE
        )
        print("✅ Embedding model loaded")

        print(f"\n📦 Loading rerank model: {config.RERANK_MODEL_NAME}")
        rerank_model = FlagReranker(
            config.RERANK_MODEL_NAME,
            use_fp16=config.USE_FP16 and config.DEVICE != "cpu",
            device=config.DEVICE
        )
        print("✅ Rerank model loaded")

        # 🚀 Optimisation: switch models to eval/inference mode
        if hasattr(rerank_model, 'model'):
            rerank_model.model.eval()
            # Disable gradients for inference-only workloads
            for param in rerank_model.model.parameters():
                param.requires_grad = False

        # Warm up both models once
        print("\n🔥 Warming up models...")
        _ = embedding_model.encode(["warm up text"])
        _ = rerank_model.compute_score([["warm up query", "warm up doc"]])
        print("✅ Models warmed up")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("   Service will start but models won't be available.")
        embedding_model = None
        rerank_model = None

    print("\n" + "=" * 60)
    print("✅ Inference Service Ready!")
    print("=" * 60)

    yield

    # Teardown
    print("\n🧹 Shutting down...")
    if embedding_model is not None:
        del embedding_model
    if rerank_model is not None:
        del rerank_model

    # Release GPU cache if applicable
    if config.DEVICE.startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


# === FastAPI App ===
app = FastAPI(
    title="Inference Worker Service",
    description="Standalone embedding and rerank inference service",
    version="0.1.0",
    lifespan=lifespan
)


# === API Endpoints ===
@app.get("/", tags=["Meta"])
async def root():
    """Service root endpoint."""
    return {
        "service": "Inference Worker",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "embed": "/embed",
            "rerank": "/rerank"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time

    models_loaded = {
        "embedding": embedding_model is not None,
        "rerank": rerank_model is not None
    }

    status = "healthy" if all(models_loaded.values()) else "degraded"

    return HealthResponse(
        status=status,
        uptime_seconds=uptime,
        models_loaded=models_loaded,
        config=config.get_config_summary()
    )


@app.get("/ready", tags=["Meta"])
async def readiness_check():
    """Readiness probe indicating models are loaded."""
    if embedding_model is None or rerank_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


@app.get("/models", response_model=ModelsResponse, tags=["Meta"])
async def list_models():
    """List available embedding/rerank models."""
    return ModelsResponse(
        embedding_models={
            config.EMBED_MODEL_NAME: {
                "loaded": embedding_model is not None,
                "dimension": 1024,  # BGE-M3
                "device": config.DEVICE
            }
        },
        rerank_models={
            config.RERANK_MODEL_NAME: {
                "loaded": rerank_model is not None,
                "device": config.DEVICE
            }
        }
    )


@app.post("/embed", response_model=EmbedResponse, tags=["Inference"])
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for supplied texts."""
    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded. Please check service logs."
        )

    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    start_time = time.perf_counter()

    try:
        # 🚀 Use thread pool to offload blocking work
        from starlette.concurrency import run_in_threadpool

        def _encode():
            return embedding_model.encode(
                request.texts,
                batch_size=request.batch_size or config.MAX_BATCH_SIZE,
                max_length=8192
            )

        embeddings = await run_in_threadpool(_encode)

        # BGE-M3 returns dictionaries; extract dense embeddings
        if isinstance(embeddings, dict):
            embeddings = embeddings['dense_vecs']

        # Convert numpy arrays into plain lists
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, list) and hasattr(embeddings[0], 'tolist'):
            embeddings = [e.tolist() for e in embeddings]

        # L2-normalise when requested
        if request.normalize:
            import numpy as np
            embeddings = np.array(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = (embeddings / norms).tolist()

        duration = (time.perf_counter() - start_time) * 1000

        return EmbedResponse(
            embeddings=embeddings,
            model=request.model or config.EMBED_MODEL_NAME,
            dimension=len(embeddings[0]),
            processing_time_ms=round(duration, 2),
            batch_info={
                "total_texts": len(request.texts),
                "batch_size": request.batch_size or config.MAX_BATCH_SIZE,
                "num_batches": (len(request.texts) + (request.batch_size or config.MAX_BATCH_SIZE) - 1) // (request.batch_size or config.MAX_BATCH_SIZE)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/rerank", response_model=RerankResponse, tags=["Inference"])
async def rerank_documents(request: RerankRequest):
    """Rerank documents using the configured cross-encoder."""
    if rerank_model is None:
        raise HTTPException(
            status_code=503,
            detail="Rerank model not loaded. Please check service logs."
        )

    if not request.documents:
        raise HTTPException(status_code=400, detail="documents cannot be empty")

    start_time = time.perf_counter()

    try:
        # Build query-document pairs
        pairs = [[request.query, doc] for doc in request.documents]

        # 🚀 Batch rerank calls and push to thread pool
        from starlette.concurrency import run_in_threadpool

        def _compute_scores():
            return rerank_model.compute_score(pairs, batch_size=128)

        scores = await run_in_threadpool(_compute_scores)

        # Ensure we operate on a list copy
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [float(scores)]

        # Sort by score and capture indices
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Keep only the requested top_k items
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {str(e)}")


# === Main Entry ===
if __name__ == "__main__":
    uvicorn.run(
        "inference_service.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )
