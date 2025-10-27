import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from models import KnowledgeChunk, KnowledgeDocument  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    KnowledgeChunk = None  # type: ignore
    KnowledgeDocument = None  # type: ignore
try:
    from utils.metrics_helpers import (  # type: ignore
        track_pgvector_query,
        track_embedding,
        track_rerank,
        register_model_version,
        track_embedding_tokens,
    )
except ImportError:  # pragma: no cover - fallback to local stubs
    from backend.utils.metrics_helpers import (  # type: ignore
        track_pgvector_query,
        track_embedding,
        track_rerank,
        register_model_version,
        track_embedding_tokens,
    )

try:
    from pgvector.sqlalchemy import Vector as PgVectorType  # type: ignore
except ImportError:  # pragma: no cover - pgvector optional
    PgVectorType = None  # type: ignore

if os.getenv("ALLOW_TORCH_LOAD_UNSAFE", "true").lower() in {"true", "1", "yes"}:
    try:
        from transformers.utils import import_utils as _import_utils
        from transformers import modeling_utils as _modeling_utils

        def _noop(*args, **kwargs):
            return None

        if hasattr(_import_utils, "check_torch_load_is_safe"):
            _import_utils.check_torch_load_is_safe = _noop  # type: ignore
        if hasattr(_modeling_utils, "check_torch_load_is_safe"):
            _modeling_utils.check_torch_load_is_safe = _noop  # type: ignore
    except Exception:
        pass

_EMBED_MODEL: Optional[SentenceTransformer] = None
_RERANKER_MODEL: Optional[AutoModelForSequenceClassification] = None
_RERANKER_TOKENIZER: Optional[AutoTokenizer] = None
_VECTOR_SEARCH_AVAILABLE: bool = True
KNOWLEDGE_SCHEMA = os.getenv("KNOWLEDGE_DB_SCHEMA", "knowledge")


def _get_embed_device() -> str:
    return os.getenv("EMBEDDING_DEVICE", "cpu")


def _get_rerank_device() -> str:
    return os.getenv("RERANKER_DEVICE", _get_embed_device())


def _reranker_enabled() -> bool:
    return os.getenv("ENABLE_RERANKER", "true").lower() not in {"0", "false", "no"}


def _get_cache_dir() -> Optional[str]:
    cache_dir = os.getenv("MODEL_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        cache_dir = _get_cache_dir()
        backend = os.getenv("EMBEDDING_BACKEND", "auto")
        backend_arg = {}
        if backend in {"torch", "onnx", "openvino"}:
            backend_arg["backend"] = backend
        elif backend not in {"auto", "", None}:
            os.environ["EMBEDDING_BACKEND"] = "torch"
            backend_arg["backend"] = "torch"
        version_label = "local"
        try:
            _EMBED_MODEL = SentenceTransformer(
                model_name,
                device=_get_embed_device(),
                cache_folder=cache_dir,
                **backend_arg,
            )
        except TypeError:
            # Older sentence-transformers versions don't accept backend kwarg
            _EMBED_MODEL = SentenceTransformer(
                model_name,
                device=_get_embed_device(),
                cache_folder=cache_dir,
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            if os.getenv("EMBEDDING_FALLBACK_ONNX", "true").lower() in {"true", "1", "yes"}:
                _EMBED_MODEL = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder=cache_dir,
                    backend="onnx",
                )
                version_label = "local-onnx"
            else:
                raise exc
        register_model_version("embedding", model_name, version=version_label)
    return _EMBED_MODEL


def _get_reranker() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    global _RERANKER_MODEL, _RERANKER_TOKENIZER
    if _RERANKER_MODEL is None or _RERANKER_TOKENIZER is None:
        model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
        cache_dir = _get_cache_dir()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        model.to(_get_rerank_device())
        model.eval()
        _RERANKER_MODEL = model
        _RERANKER_TOKENIZER = tokenizer
        # Register reranker model version with Prometheus metrics
        register_model_version("reranker", model_name, version="local")
    return _RERANKER_MODEL, _RERANKER_TOKENIZER


async def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Generate text embeddings.
    - Prefer the remote inference service when `ENABLE_REMOTE_INFERENCE=true`
    - Fall back to the local model if the service fails or is disabled
    - Return L2-normalized vectors suitable for cosine similarity
    """
    import logging
    logger = logging.getLogger(__name__)

    # Try remote inference service if enabled
    try:
        from knowledge_config.inference_config import inference_config

        if inference_config.ENABLE_REMOTE_INFERENCE:
            try:
                from services.inference_client import get_embedding_client
                from services.metrics import rag_operation_counter, inference_latency_histogram
                import time

                logger.info(
                    "📡 Using remote inference for embeddings (texts=%s, url=%s)",
                    len(texts),
                    inference_config.EMBEDDING_SERVICE_URL,
                )

                start_time = time.perf_counter()
                client = get_embedding_client()
                embeddings = await client.embed(list(texts), normalize=True)

                # Record metrics
                duration = time.perf_counter() - start_time
                rag_operation_counter.labels(operation="embed", source="remote").inc()
                inference_latency_histogram.labels(service="embedding").observe(duration)

                logger.info("✅ Remote embedding succeeded in %.2f ms", duration * 1000)
                return embeddings

            except Exception as e:
                # Log warning and fall back to local processing
                logger.warning("⚠️  Remote embedding failed: %s. Falling back to local model.", e)

                from services.metrics import rag_operation_counter
                rag_operation_counter.labels(operation="embed", source="fallback").inc()
                # Continue to local processing below
        else:
            logger.debug("Remote embedding service disabled (ENABLE_REMOTE_INFERENCE=false)")

    except ImportError:
        # inference_config not available, use local
        logger.debug("Remote inference configuration unavailable; defaulting to local model")

    # Local processing (default or fallback)
    logger.info("💻 Using local embedding model (texts=%s)", len(texts))

    model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    batch_size = len(texts)

    # Track embedding performance with the provided context manager
    with track_embedding(model_name, "local", batch_size):
        def _encode(batch: Sequence[str]) -> List[List[float]]:
            model = _get_embed_model()
            vectors = model.encode(
                list(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [vec.astype("float32").tolist() for vec in vectors]

        try:
            from services.metrics import rag_operation_counter
            rag_operation_counter.labels(operation="embed", source="local").inc()
        except ImportError:
            pass

        result = await run_in_threadpool(_encode, texts)

        # Approximate token count (rough heuristic: characters / 4)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4
        track_embedding_tokens(model_name, estimated_tokens)

        logger.info("✅ Local embedding succeeded (dimension=%s)", len(result[0]) if result else 0)
        return result


def _build_role_filter(role_id: Optional[int]) -> Tuple[str, Dict[str, Any]]:
    if role_id is None:
        return "", {}
    return "AND kd.role_id = :role_id", {"role_id": role_id}


async def _hybrid_candidates(
    db: AsyncSession,
    query_text: str,
    query_embedding: List[float],
    *,
    limit: int,
    role_id: Optional[int],
    vector_limit: int,
    text_limit: int,
    vector_weight: float,
    trigram_weight: float,
) -> List[Dict[str, Any]]:
    global _VECTOR_SEARCH_AVAILABLE
    role_clause, role_params = _build_role_filter(role_id)
    safe_schema = KNOWLEDGE_SCHEMA.replace('"', "").replace(";", "")
    schema_prefix = f'"{safe_schema}".' if safe_schema else ""
    role_filtered = role_id is not None

    async def _text_only_candidates() -> List[Dict[str, Any]]:
        # Track pure trigram-search performance
        with track_pgvector_query("text", role_filtered):
            text_stmt = text(
                f"""
                SELECT
                    kc.id,
                    kc.document_id,
                    kc.content,
                    kc.metadata,
                    kd.title AS document_title,
                    0::float AS vector_score,
                    similarity(kc.content, :query_text) AS trigram_score,
                    similarity(kc.content, :query_text) AS hybrid_score
                FROM {schema_prefix}knowledge_chunks kc
                JOIN {schema_prefix}knowledge_documents kd ON kd.id = kc.document_id
                WHERE kc.content % :query_text
                {role_clause}
                ORDER BY similarity(kc.content, :query_text) DESC
                LIMIT :limit
                """
            ).bindparams(bindparam("query_text"), bindparam("limit"))
            params = {"query_text": query_text, "limit": limit}
            params.update(role_params)
            result = await db.execute(text_stmt, params)
            candidates = list(result.mappings())

            # Record how many candidates were returned
            from services.metrics import pgvector_candidates_returned_histogram
            pgvector_candidates_returned_histogram.labels(query_type="text").observe(len(candidates))

            return candidates

    if _VECTOR_SEARCH_AVAILABLE and PgVectorType is not None:
        stmt = text(
            f"""
            WITH vector_candidates AS (
                SELECT
                    kc.id,
                    kc.document_id,
                    kc.content,
                    kc.metadata,
                    kd.title AS document_title,
                    1 - (kc.embedding <=> :query_embedding) AS vector_score,
                    similarity(kc.content, :query_text) AS trigram_score
                FROM {schema_prefix}knowledge_chunks kc
                JOIN {schema_prefix}knowledge_documents kd ON kd.id = kc.document_id
                WHERE kc.embedding IS NOT NULL
                {role_clause}
                ORDER BY kc.embedding <=> :query_embedding
                LIMIT :vector_limit
            ),
            text_candidates AS (
                SELECT
                    kc.id,
                    kc.document_id,
                    kc.content,
                    kc.metadata,
                    kd.title AS document_title,
                    NULL::float AS vector_score,
                    similarity(kc.content, :query_text) AS trigram_score
                FROM {schema_prefix}knowledge_chunks kc
                JOIN {schema_prefix}knowledge_documents kd ON kd.id = kc.document_id
                WHERE kc.content % :query_text
                {role_clause}
                ORDER BY similarity(kc.content, :query_text) DESC
                LIMIT :text_limit
            ),
            combined AS (
                SELECT * FROM vector_candidates
                UNION ALL
                SELECT * FROM text_candidates
            )
            SELECT
                id,
                document_id,
                content,
                metadata,
                COALESCE(vector_score, 0) AS vector_score,
                COALESCE(trigram_score, 0) AS trigram_score,
                document_title,
                (COALESCE(vector_score, 0) * :vector_weight + COALESCE(trigram_score, 0) * :trigram_weight) AS hybrid_score
            FROM combined
            ORDER BY hybrid_score DESC
            LIMIT :limit
            """
        )

        bind_params = [
            bindparam("query_embedding", type_=PgVectorType(len(query_embedding))),
            bindparam("query_text"),
            bindparam("vector_limit"),
            bindparam("text_limit"),
            bindparam("vector_weight"),
            bindparam("trigram_weight"),
            bindparam("limit"),
        ]
        if "role_id" in role_params:
            bind_params.append(bindparam("role_id"))
        stmt = stmt.bindparams(*bind_params)
        params: Dict[str, Any] = {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "vector_limit": vector_limit,
            "text_limit": text_limit,
            "vector_weight": vector_weight,
            "trigram_weight": trigram_weight,
            "limit": limit,
        }
        params.update(role_params)

        # Track hybrid search performance
        with track_pgvector_query("hybrid", role_filtered):
            try:
                result = await db.execute(stmt, params)
                candidates = list(result.mappings())

                # Record how many candidates were returned
                from services.metrics import pgvector_candidates_returned_histogram, pgvector_query_counter
                pgvector_candidates_returned_histogram.labels(query_type="hybrid").observe(len(candidates))
                pgvector_query_counter.labels(query_type="hybrid", status="success").inc()

                return candidates
            except Exception as exc:
                await db.rollback()
                from services.metrics import pgvector_query_counter
                if "embedding" in str(exc):
                    _VECTOR_SEARCH_AVAILABLE = False
                    pgvector_query_counter.labels(query_type="hybrid", status="fallback").inc()
                else:
                    pgvector_query_counter.labels(query_type="hybrid", status="error").inc()
                    raise

    return await _text_only_candidates()


async def _rerank(
    query: str,
    passages: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not passages:
        return []
    if not _reranker_enabled():
        from services.metrics import rerank_counter
        rerank_counter.labels(
            model=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"),
            source="local",
            status="skipped"
        ).inc()
        return list(passages)[:top_k]

    import logging
    logger = logging.getLogger(__name__)
    model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    candidate_count = len(passages)

    # Try remote inference service if enabled
    try:
        from knowledge_config.inference_config import inference_config

        if inference_config.ENABLE_REMOTE_INFERENCE:
            try:
                from services.inference_client import get_rerank_client
                from services.metrics import rag_operation_counter, inference_latency_histogram
                import time

                logger.info(
                    "📡 Using remote inference for rerank (candidates=%s, url=%s)",
                    len(passages),
                    inference_config.RERANK_SERVICE_URL,
                )

                start_time = time.perf_counter()
                client = get_rerank_client()
                documents = [p["content"] for p in passages]
                scores = await client.rerank(query, documents, model=model_name, top_k=top_k)

                # Record metrics
                duration = time.perf_counter() - start_time
                rag_operation_counter.labels(operation="rerank", source="remote").inc()
                inference_latency_histogram.labels(service="rerank").observe(duration)

                # Build scored results
                scored = []
                for item, score in zip(passages, scores):
                    enriched = dict(item)
                    enriched["rerank_score"] = float(score)
                    scored.append(enriched)

                scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                logger.info("✅ Remote rerank succeeded in %.2f ms", duration * 1000)
                return scored[:top_k]

            except Exception as e:
                # Log warning and fall back to local processing
                logger.warning("⚠️  Remote rerank failed: %s. Falling back to local model.", e)

                from services.metrics import rag_operation_counter
                rag_operation_counter.labels(operation="rerank", source="fallback").inc()
                # Continue to local processing below
        else:
            logger.debug("Remote rerank service disabled (ENABLE_REMOTE_INFERENCE=false)")

    except ImportError:
        # inference_config not available, use local
        logger.debug("Remote inference configuration unavailable; defaulting to local rerank")

    # Local processing (default or fallback)
    logger.info("💻 Using local reranker (candidates=%s)", len(passages))

    # Track rerank performance with the metrics context manager
    with track_rerank(model_name, "local", candidate_count) as record_scores:
        def _score() -> List[float]:
            model, tokenizer = _get_reranker()
            encoded = tokenizer(
                [query] * len(passages),
                [p["content"] for p in passages],
                padding=True,
                truncation=True,
                max_length=int(os.getenv("RERANKER_MAX_LENGTH", "512")),
                return_tensors="pt",
            )
            device = _get_rerank_device()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.squeeze(-1)
            return logits.detach().cpu().tolist()

        try:
            from services.metrics import rag_operation_counter
            rag_operation_counter.labels(operation="rerank", source="local").inc()

            scores = await run_in_threadpool(_score)

            # Record the score distribution for telemetry
            record_scores(scores)

        except Exception as e:
            # fallback: return original order if reranker fails (e.g., missing model files)
            from services.metrics import rerank_counter
            rerank_counter.labels(model=model_name, source="local", status="error").inc()
            import logging
            logging.getLogger(__name__).warning(f"Rerank failed: {e}, using original order")
            return list(passages)[:top_k]

        scored = []
        for item, score in zip(passages, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            scored.append(enriched)

        scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k]


async def _track_retrieval_access(db: AsyncSession, results: List[Dict[str, Any]]) -> None:
    """
    Track document and chunk access for cold/hot tier management.
    Also triggers automatic rehydration for cold documents.
    """
    if not results:
        return

    try:
        from services.knowledge_tier import track_document_access, track_chunk_usage, rehydrate_document
        from models import KnowledgeDocument
        from sqlalchemy import select

        # Collect unique document IDs and chunk IDs
        document_ids = set()
        chunk_ids = set()

        for result in results:
            if "document_id" in result:
                document_ids.add(result["document_id"])
            if "id" in result:
                chunk_ids.add(result["id"])

        # Check for cold documents and rehydrate if needed
        if document_ids:
            stmt = select(KnowledgeDocument).where(
                KnowledgeDocument.id.in_(document_ids),
                KnowledgeDocument.is_cold == True  # noqa: E712
            )
            result = await db.execute(stmt)
            cold_docs = result.scalars().all()

            # Rehydrate cold documents asynchronously (fire and forget)
            for doc in cold_docs:
                try:
                    await rehydrate_document(db, doc.id)
                except Exception as e:
                    # Log but don't fail the retrieval
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to rehydrate cold document {doc.id}: {e}"
                    )

        # Track access for all documents
        for doc_id in document_ids:
            try:
                await track_document_access(db, doc_id)
            except Exception:
                pass  # Don't fail retrieval if tracking fails

        # Track usage for all chunks
        for chunk_id in chunk_ids:
            try:
                await track_chunk_usage(db, chunk_id)
            except Exception:
                pass  # Don't fail retrieval if tracking fails

    except ImportError:
        # knowledge_tier module not available, skip tracking
        pass
    except Exception as e:
        # Log but don't fail the retrieval
        import logging
        logging.getLogger(__name__).warning(f"Failed to track retrieval access: {e}")


async def retrieve_context(
    db: AsyncSession,
    *,
    query: str,
    role_id: Optional[int],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval (pgvector + pg_trgm) and rerank the passages.
    Returns dictionaries containing content, metadata, and scores.

    This function also:
    - Tracks document/chunk access for cold/hot tier management
    - Automatically rehydrates cold documents when accessed
    """
    embedding = (await embed_texts([query]))[0]
    weight_vector = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
    weight_trgm = float(os.getenv("HYBRID_TRIGRAM_WEIGHT", "0.4"))
    vector_limit = int(os.getenv("HYBRID_VECTOR_LIMIT", "20"))
    text_limit = int(os.getenv("HYBRID_TEXT_LIMIT", "20"))
    candidate_limit = int(os.getenv("HYBRID_CANDIDATE_LIMIT", str(top_k * 4)))

    candidates = await _hybrid_candidates(
        db,
        query_text=query,
        query_embedding=embedding,
        limit=candidate_limit,
        role_id=role_id,
        vector_limit=vector_limit,
        text_limit=text_limit,
        vector_weight=weight_vector,
        trigram_weight=weight_trgm,
    )
    reranked = await _rerank(query, candidates, top_k)

    # Track access for cold/hot tier management
    await _track_retrieval_access(db, reranked)

    return reranked


def _split_text(
    text_value: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Simple splitter that works on sentence-like boundaries to avoid cutting words abruptly.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    paragraphs = [p.strip() for p in text_value.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for para in paragraphs:
        sentences = [s.strip() for s in para.replace("\n", " ").split("。") if s.strip()]
        for sent in sentences:
            token_estimate = len(sent)
            if current_len + token_estimate > chunk_size and buffer:
                chunks.append("".join(buffer).strip())
                overlap_text = "".join(buffer)[-chunk_overlap:] if chunk_overlap > 0 else ""
                buffer = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            buffer.append(sent + "。")
            current_len += token_estimate
        if buffer and buffer[-1].endswith("。"):
            buffer[-1] = buffer[-1]  # keep punctuation

    if buffer:
        chunks.append("".join(buffer).strip())

    return [chunk for chunk in chunks if chunk]


async def ingest_document(
    db: AsyncSession,
    *,
    title: str,
    text: str,
    owner_id: Optional[int] = None,
    role_id: Optional[int] = None,
    source: Optional[str] = None,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> KnowledgeDocument:
    """
    Break a document into chunks, embed them, and persist the records.
    """
    chunks = _split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Text is empty or cannot be split into valid segments")

    embeddings = await embed_texts(chunks)

    doc_metadata = dict(metadata or {})
    if source:
        doc_metadata.setdefault("source", source)

    document = KnowledgeDocument(
        title=title,
        description=doc_metadata.get("description"),
        owner_id=owner_id,
        role_id=role_id,
        source=source,
        language=language,
        meta=doc_metadata,
    )
    db.add(document)
    await db.flush()

    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_metadata = dict(doc_metadata)
        if source:
            chunk_metadata.setdefault("source", source)
        chunk = KnowledgeChunk(
            document_id=document.id,
            chunk_index=idx,
            content=chunk_text,
            embedding=embedding,
            meta=chunk_metadata,
            token_count=len(chunk_text),
        )
        db.add(chunk)

    await db.commit()
    await db.refresh(document)
    return document


__all__ = [
    "embed_texts",
    "retrieve_context",
    "ingest_document",
]
