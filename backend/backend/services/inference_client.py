"""
Inference Service Client

Client wrapper for the inference services with retry, fallback, and circuit-breaker logic.

Important: this client is disabled by default. Set `ENABLE_REMOTE_INFERENCE=true` to enable it.
"""
import asyncio
import httpx
import logging
from typing import List
from datetime import datetime, timedelta

from backend.config.knowledge_config.inference_config import inference_config

logger = logging.getLogger(__name__)


class SimpleCircuitBreaker:
    """Minimal circuit breaker implementation."""

    def __init__(self, threshold: int = 10, timeout_sec: int = 60):
        self.threshold = threshold
        self.timeout_sec = timeout_sec
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open

    def record_success(self):
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.threshold:
            self.state = "open"
            logger.warning(f"🔴 Circuit breaker opened after {self.failures} failures")

    def can_attempt(self) -> bool:
        if self.state == "closed":
            return True

        # Allow retries once the cooldown period has passed
        if self.last_failure_time and \
           datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_sec):
            self.state = "closed"
            self.failures = 0
            logger.info("🟢 Circuit breaker reset")
            return True

        return False


class EmbeddingClient:
    """Lightweight client for the embedding inference service."""

    def __init__(
        self,
        base_url: str = None,
        timeout: float = None,
        max_parallel: int = None
    ):
        self.base_url = base_url or inference_config.EMBEDDING_SERVICE_URL
        self.timeout = timeout or inference_config.EMBED_TIMEOUT_SEC
        max_parallel = max_parallel or inference_config.MAX_PARALLEL_INFERENCE

        self._semaphore = asyncio.Semaphore(max_parallel)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=5.0)
        )
        self._circuit_breaker = SimpleCircuitBreaker(
            threshold=inference_config.CIRCUIT_BREAKER_THRESHOLD,
            timeout_sec=inference_config.CIRCUIT_BREAKER_TIMEOUT_SEC
        )
        self._consecutive_failures = 0

    async def embed(
        self,
        texts: List[str],
        model: str = "bge-m3",
        normalize: bool = True,
        retry_count: int = 0
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""

        # Abort early if the circuit breaker is open
        if not self._circuit_breaker.can_attempt():
            raise Exception("Circuit breaker is open")

        async with self._semaphore:
            try:
                resp = await self._client.post(
                    "/embed",
                    json={
                        "texts": texts,
                        "model": model,
                        "normalize": normalize,
                        "batch_size": inference_config.EMBED_BATCH_SIZE
                    }
                )
                resp.raise_for_status()

                data = resp.json()
                embeddings = data["embeddings"]

                # Record success and reset counters
                self._circuit_breaker.record_success()
                self._consecutive_failures = 0

                return embeddings

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                # Timeout or connection failure
                self._circuit_breaker.record_failure()
                self._consecutive_failures += 1

                logger.warning(f"Inference request failed: {type(e).__name__}")

                # Retry with simple backoff
                if retry_count < inference_config.MAX_RETRIES:
                    await asyncio.sleep(inference_config.RETRY_DELAY_SEC * (retry_count + 1))
                    return await self.embed(texts, model, normalize, retry_count + 1)

                raise

            except httpx.HTTPStatusError as e:
                # Non-success status codes are escalated immediately
                logger.error(f"HTTP error {e.response.status_code}: {e}")
                raise

    async def health_check(self) -> bool:
        """Return True when the embedding service responds to /health."""
        try:
            resp = await self._client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except:
            return False

    async def close(self):
        """Release underlying networking resources."""
        await self._client.aclose()


class RerankClient:
    """Lightweight client for the rerank inference service."""

    def __init__(
        self,
        base_url: str = None,
        timeout: float = None,
        max_parallel: int = None
    ):
        self.base_url = base_url or inference_config.RERANK_SERVICE_URL
        self.timeout = timeout or inference_config.RERANK_TIMEOUT_SEC
        max_parallel = max_parallel or inference_config.MAX_PARALLEL_INFERENCE

        self._semaphore = asyncio.Semaphore(max_parallel)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=5.0)
        )
        self._circuit_breaker = SimpleCircuitBreaker(
            threshold=inference_config.CIRCUIT_BREAKER_THRESHOLD,
            timeout_sec=inference_config.CIRCUIT_BREAKER_TIMEOUT_SEC
        )

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = "bge-reranker-base",
        top_k: int = None,
        retry_count: int = 0
    ) -> List[float]:
        """Return rerank scores for the provided documents."""

        if not self._circuit_breaker.can_attempt():
            # Provide deterministic fallback scores when circuits are open
            logger.warning("Circuit breaker open for rerank, using fallback")
            return [1.0 / (i + 1) for i in range(len(documents))]

        async with self._semaphore:
            try:
                resp = await self._client.post(
                    "/rerank",
                    json={
                        "query": query,
                        "documents": documents,
                        "model": model,
                        "top_k": top_k or len(documents),
                        "return_documents": False
                    }
                )
                resp.raise_for_status()

                data = resp.json()
                scores = data["scores"]

                self._circuit_breaker.record_success()

                return scores

            except Exception as e:
                self._circuit_breaker.record_failure()

                logger.warning(f"Rerank failed: {e}, using fallback")

                if retry_count < inference_config.MAX_RETRIES:
                    await asyncio.sleep(inference_config.RETRY_DELAY_SEC * (retry_count + 1))
                    return await self.rerank(query, documents, model, top_k, retry_count + 1)

                # Final fallback: monotonically decreasing scores
                return [1.0 / (i + 1) for i in range(len(documents))]

    async def health_check(self) -> bool:
        """Return True when the rerank service responds to /health."""
        try:
            resp = await self._client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except:
            return False

    async def close(self):
        await self._client.aclose()

# Lazy singleton instances
_embedding_client: EmbeddingClient = None
_rerank_client: RerankClient = None


def get_embedding_client() -> EmbeddingClient:
    """Return the shared embedding client."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def get_rerank_client() -> RerankClient:
    """Return the shared rerank client."""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClient()
    return _rerank_client
