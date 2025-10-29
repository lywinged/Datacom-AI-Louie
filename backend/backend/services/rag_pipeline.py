"""
RAG ingestion and retrieval pipeline backed by Qdrant.
"""
import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional

from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI
from qdrant_client.http import models as qdrant_models

from backend.config.settings import settings, OPENAI_CONFIG
from backend.models.rag_schemas import Citation, DocumentResponse, RAGResponse
from backend.services.metadata_index import search_by_title
from backend.config.knowledge_config.inference_config import inference_config
from backend.services.inference_client import get_embedding_client, get_rerank_client
from backend.services.onnx_inference import (
    get_embedding_model,
    get_reranker_model,
    reranker_is_cpu_only,
    switch_to_fallback_reranker,
    set_reranker_model_path,
    get_current_reranker_path,
    _has_cuda_available,
)
from backend.services.qdrant_client import ensure_collection, get_qdrant_client
from backend.services.token_counter import get_token_counter, TokenUsage
from backend.utils.text_splitter import split_text
from backend.utils.openai import sanitize_messages


COLLECTION_NAME = settings.QDRANT_COLLECTION

# Initialize OpenAI client for answer generation
_openai_client = None
logger = logging.getLogger(__name__)
_reranker_switch_locked = False
_AUTHOR_QUESTION_PATTERN = re.compile(
    r"\bwho\s+(?:wrote|is\s+the\s+author\s+of|authored)\b",
    flags=re.IGNORECASE,
)

VECTOR_LIMIT_MIN = 5
VECTOR_LIMIT_MAX = 20
CONTENT_CHAR_MIN = 150
CONTENT_CHAR_MAX = 1000
DEFAULT_CONTENT_CHAR_LIMIT = 300

_token_counter = get_token_counter()


def _get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY or OPENAI_CONFIG.get("api_key")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM answer generation")

        # Support both OPENAI_BASE_URL and OPENAI_BASE_URL
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or OPENAI_CONFIG.get("base_url")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        _openai_client = AsyncOpenAI(**client_kwargs)

    return _openai_client


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


async def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using ONNX runtime in a worker thread."""
    if inference_config.ENABLE_REMOTE_INFERENCE:
        client = get_embedding_client()
        return await client.embed(texts, normalize=True)

    def _encode() -> List[List[float]]:
        model = get_embedding_model()
        vectors = model.encode(texts)
        return vectors.astype("float32").tolist()

    return await run_in_threadpool(_encode)


def _get_vector_size() -> int:
    """Return the expected embedding vector size for the active pipeline."""
    if inference_config.ENABLE_REMOTE_INFERENCE:
        return settings.RAG_VECTOR_SIZE

    from backend.services.onnx_inference import get_embedding_model

    return get_embedding_model().vector_size


def _should_switch_reranker(latency_ms: float) -> bool:
    """Determine whether to switch to the fallback reranker."""
    if inference_config.ENABLE_REMOTE_INFERENCE:
        return False
    global _reranker_switch_locked
    if _reranker_switch_locked:
        return False
    if not settings.RERANK_FALLBACK_MODEL_PATH:
        _reranker_switch_locked = True
        return False
    if not reranker_is_cpu_only():
        _reranker_switch_locked = True
        return False
    return latency_ms >= settings.RERANK_CPU_SWITCH_THRESHOLD_MS


def _apply_reranker_override(choice: Optional[str]) -> str:
    """Apply manual reranker override if requested."""
    global _reranker_switch_locked

    if inference_config.ENABLE_REMOTE_INFERENCE:
        return "remote"

    if not choice:
        choice = "auto"
    choice_normalized = choice.strip().lower()

    def _maybe_switch(target_path: Optional[str]) -> bool:
        if not target_path:
            return False
        current = get_current_reranker_path()
        if current == target_path:
            return True
        try:
            set_reranker_model_path(target_path)
            return True
        except Exception as exc:
            logger.warning("Failed to switch reranker to %s: %s", target_path, exc)
            return False

    if choice_normalized in ("auto", ""):
        _reranker_switch_locked = False
        _maybe_switch(settings.ONNX_RERANK_MODEL_PATH)
        return "auto"

    if choice_normalized == "primary":
        if _maybe_switch(settings.ONNX_RERANK_MODEL_PATH):
            _reranker_switch_locked = True
            return "primary"
        return "auto"

    if choice_normalized == "fallback":
        if _maybe_switch(settings.RERANK_FALLBACK_MODEL_PATH):
            _reranker_switch_locked = True
            return "fallback"
        logger.warning("Fallback reranker requested but not configured.")
        return "auto"

    resolved = choice.strip()
    if resolved:
        if _maybe_switch(resolved):
            _reranker_switch_locked = True
            return "custom"
    return "auto"


async def _rerank(
    question: str,
    chunks: List[RetrievedChunk],
    override_choice: Optional[str] = None,
) -> Tuple[List[RetrievedChunk], float, str, str]:
    """Apply ONNX reranker to refine relevance ordering."""
    if not chunks:
        mode = "remote" if inference_config.ENABLE_REMOTE_INFERENCE else "auto"
        return [], 0.0, "", mode

    global _reranker_switch_locked

    docs = [chunk.content for chunk in chunks]

    if inference_config.ENABLE_REMOTE_INFERENCE:
        client = get_rerank_client()
        start = time.perf_counter()
        scores = await client.rerank(question, docs, top_k=len(docs))
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = "remote"
        reranker_mode = "remote"
    else:
        reranker_mode = _apply_reranker_override(override_choice)

        def _score(model) -> List[float]:
            scores_local = model.score(question, docs)
            return scores_local.tolist()

        model = get_reranker_model()
        start = time.perf_counter()
        scores = await run_in_threadpool(lambda: _score(model))
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = getattr(model, "resolved_model_path", getattr(model, "model_path", ""))

        if _should_switch_reranker(duration_ms):
            if switch_to_fallback_reranker():
                _reranker_switch_locked = True
                model = get_reranker_model()
                start = time.perf_counter()
                scores = await run_in_threadpool(lambda: _score(model))
                duration_ms = (time.perf_counter() - start) * 1000
                model_name = getattr(model, "resolved_model_path", getattr(model, "model_path", ""))
                logger.info(
                    "Switched to fallback reranker model '%s' after CPU latency %.1f ms.",
                    model_name or "<unknown>",
                    duration_ms,
                )
            else:
                _reranker_switch_locked = True

    reranked: List[RetrievedChunk] = []
    for chunk, score in zip(chunks, scores):
        reranked.append(
            RetrievedChunk(
                content=chunk.content,
                source=chunk.source,
                score=float(score),
                metadata={**chunk.metadata, "base_score": chunk.score},
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked, duration_ms, model_name, reranker_mode


async def ingest_document(
    title: str,
    content: str,
    *,
    source: str,
    metadata: Dict[str, Any] | None = None,
) -> DocumentResponse:
    """Chunk a document, embed and upsert into Qdrant."""
    vector_size = _get_vector_size()
    ensure_collection(vector_size)
    client = get_qdrant_client()

    document_id = int(time.time() * 1000)
    metadata = metadata or {}

    chunks = split_text(
        content,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    tic = time.perf_counter()
    embeddings = await _embed_texts(chunks)
    embed_duration_ms = (time.perf_counter() - tic) * 1000

    points = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = uuid.uuid4().hex
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "chunk_index": idx,
                    "content": chunk_text,
                    "title": title,
                    "source": source,
                    "metadata": metadata,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    return DocumentResponse(
        document_id=document_id,
        title=title,
        num_chunks=len(points),
        embedding_time_ms=embed_duration_ms,
    )


def _is_author_question(question: str) -> bool:
    return bool(_AUTHOR_QUESTION_PATTERN.search(question))


def _extract_title_from_question(question: str) -> str:
    cleaned = _AUTHOR_QUESTION_PATTERN.sub("", question.lower())
    cleaned = cleaned.replace("?", " ").strip()
    return cleaned


async def retrieve_chunks(
    question: str,
    *,
    top_k: int = 5,
    search_limit: int = 10,
    include_timings: bool = False,
    semantic_mode: bool = False,
    reranker_override: Optional[str] = None,
    vector_limit_override: Optional[int] = None,
    content_char_limit: Optional[int] = None,
) -> Union[
    Tuple[List[RetrievedChunk], float],
    Tuple[List[RetrievedChunk], float, Dict[str, Any]],
]:
    """Retrieve relevant chunks from Qdrant and rerank."""
    vector_size = _get_vector_size()
    ensure_collection(vector_size)
    client = get_qdrant_client()

    tic_total = time.perf_counter()
    candidate_limit = max(top_k, search_limit)
    # Use 5 as minimum for better CPU performance (less reranking work)
    vector_limit = max(5, min(8, candidate_limit))

    if inference_config.ENABLE_REMOTE_INFERENCE:
        embed_model_path = inference_config.EMBEDDING_SERVICE_URL or "remote"
    else:
        embedding_model = get_embedding_model()
        embed_model_path = getattr(
            embedding_model,
            "resolved_model_path",
            getattr(embedding_model, "configured_path", settings.ONNX_EMBED_MODEL_PATH),
        )

    if inference_config.ENABLE_REMOTE_INFERENCE:
        has_gpu = False
    else:
        has_gpu = _has_cuda_available()

    # CPU optimization: limit candidates to save memory and processing time
    # GPU: can handle more candidates with better performance
    if vector_limit_override is not None:
        vector_limit = max(
            top_k,
            min(
                VECTOR_LIMIT_MAX,
                max(VECTOR_LIMIT_MIN, int(vector_limit_override)),
            ),
        )
    elif not has_gpu and semantic_mode:
        vector_limit = max(top_k, min(9, vector_limit))

    char_limit_applied: Optional[int] = None
    if content_char_limit is not None:
        char_limit_applied = min(
            CONTENT_CHAR_MAX,
            max(CONTENT_CHAR_MIN, int(content_char_limit)),
        )
    elif not has_gpu and semantic_mode:
        char_limit_applied = DEFAULT_CONTENT_CHAR_LIMIT

    embed_start = time.perf_counter()
    query_embedding = (await _embed_texts([question]))[0]
    embed_ms = (time.perf_counter() - embed_start) * 1000

    vector_start = time.perf_counter()
    base_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=vector_limit,
        with_payload=["text", "content", "source", "title", "document_id", "chunk_index", "authors", "subjects"],
        # Fetch authors and subjects for proper metadata display
    )
    vector_ms = (time.perf_counter() - vector_start) * 1000
    candidates: dict[str, RetrievedChunk] = {}

    candidate_start = time.perf_counter()
    for point in base_results:
        payload = point.payload or {}
        text_content = (
            payload.get("text")
            or payload.get("content")
            or ""
        )
        content = (
            text_content[:char_limit_applied]
            if char_limit_applied is not None
            else text_content
        )

        retrieved = RetrievedChunk(
            content=content,
            source=payload.get("source") or payload.get("title", "Unknown"),
            score=float(point.score or 0.0),
            metadata={
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "title": payload.get("title"),
                "point_id": str(point.id),
                "retrieval_source": "vector",
                "authors": payload.get("authors"),
                "subjects": payload.get("subjects"),
            },
        )
        candidates[retrieved.metadata.get("point_id") or uuid.uuid4().hex] = retrieved

    # Disabled author question optimization - it adds 300ms latency
    # if _is_author_question(question):
    #     title_query = _extract_title_from_question(question)
    #     metadata_entries = search_by_title(
    #         title_query,
    #         limit=settings.METADATA_TITLE_MATCH_LIMIT,
    #     )
    #     for entry in metadata_entries:
    #         key = entry.point_id
    #         if key in candidates:
    #             continue
    #         candidates[key] = RetrievedChunk(
    #             content=entry.content,
    #             source=entry.source or entry.title or "Unknown",
    #             score=1.0,
    #             metadata={
    #                 "title": entry.title,
    #                 "authors": entry.authors,
    #                 "retrieval_source": "metadata",
    #                 "point_id": entry.point_id,
    #             },
    #         )

    candidate_list = list(candidates.values())[:candidate_limit]
    candidate_prep_ms = (time.perf_counter() - candidate_start) * 1000

    pre_rerank_ms = (time.perf_counter() - tic_total) * 1000

    reranked, rerank_ms, reranker_model_path, reranker_mode = await _rerank(
        question,
        candidate_list,
        override_choice=reranker_override,
    )
    total_ms = (time.perf_counter() - tic_total) * 1000

    if include_timings:
        timings = {
            "embed_ms": embed_ms,
            "vector_ms": vector_ms,
            "candidate_prep_ms": candidate_prep_ms,
            "pre_rerank_ms": pre_rerank_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
            "reranker_model_path": reranker_model_path,
            "embedding_model_path": embed_model_path,
            "vector_limit_used": vector_limit,
            "content_char_limit_used": char_limit_applied,
            "reranker_mode": reranker_mode,
        }
        # Return total_ms (includes rerank) instead of pre_rerank_ms
        return reranked[:top_k], total_ms, timings

    # Return total_ms (includes rerank) instead of pre_rerank_ms
    return reranked[:top_k], total_ms


async def _generate_answer_with_llm(
    question: str,
    chunks: List[RetrievedChunk],
    *,
    model: str = "Gpt4o"
) -> Tuple[str, Optional[Dict[str, int]], float]:
    """Generate answer using LLM with retrieved context."""
    if not chunks:
        return (
            "I could not find relevant information in the knowledge base to answer your question.",
            None,
            0.0,
        )

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):  # Use top 5 chunks
        source = chunk.source or "Unknown"

        # Include metadata (title, authors) if available
        metadata_lines = []
        if chunk.metadata:
            title = chunk.metadata.get("title")
            authors = chunk.metadata.get("authors")
            if title and title != source:
                metadata_lines.append(f"Title: {title}")
            if authors:
                metadata_lines.append(f"Authors: {authors}")

        # Build context entry with metadata + content
        header = f"[{i}] Source: {source}"
        if metadata_lines:
            header += "\n" + "\n".join(metadata_lines)

        context_parts.append(f"{header}\n{chunk.content}")

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a helpful assistant answering questions based on retrieved documents.

Context (Retrieved Documents):
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Include inline citations using [1], [2], [3] format referring to the sources above
4. Be concise but complete
5. If multiple sources support your answer, cite all relevant ones

Answer:"""

    try:
        client = _get_openai_client()
        messages = sanitize_messages([
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources using [1], [2], [3] format."},
            {"role": "user", "content": prompt}
        ])

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        if usage:
            usage_dict = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }
            usage_obj = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                model=model,
                timestamp=datetime.utcnow(),
            )
            cost = _token_counter.estimate_cost(usage_obj)
        else:
            usage_dict = None
            cost = 0.0

        return answer, usage_dict, cost

    except Exception as e:
        # Fallback to simple concatenation if LLM fails
        fallback = f"Error generating answer: {str(e)}. Context: {chunks[0].content[:200]}..."
        return fallback, None, 0.0


async def answer_question(
    question: str,
    *,
    top_k: int = 5,
    use_llm: bool = True,
    include_timings: bool = True,
    reranker_override: Optional[str] = None,
    vector_limit: Optional[int] = None,
    content_char_limit: Optional[int] = None,
) -> RAGResponse:
    """
    High-level RAG pipeline: retrieve chunks and generate answer with LLM.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve
        use_llm: If True, use LLM to generate answer; if False, just concatenate chunks

    Returns:
        RAGResponse with answer, citations, and timing info
    """
    tic_total = time.perf_counter()

    search_limit = max(top_k * 2, 10)

    retrieval_kwargs = dict(
        question=question,
        top_k=top_k,
        search_limit=search_limit,
        include_timings=include_timings,
        reranker_override=reranker_override,
        vector_limit_override=vector_limit,
        content_char_limit=content_char_limit,
    )

    if include_timings:
        chunks, retrieval_ms, timings = await retrieve_chunks(**retrieval_kwargs)
    else:
        chunks, retrieval_ms = await retrieve_chunks(**{**retrieval_kwargs, "include_timings": False})
        timings = {}

    if not chunks:
        if inference_config.ENABLE_REMOTE_INFERENCE:
            embedding_model_path = inference_config.EMBEDDING_SERVICE_URL or "remote"
            reranker_model_path = inference_config.RERANK_SERVICE_URL or "remote"
            reranker_mode = "remote"
        else:
            embedding_model_path = timings.get("embedding_model_path") if timings else getattr(
                get_embedding_model(),
                "resolved_model_path",
                getattr(get_embedding_model(), "configured_path", settings.ONNX_EMBED_MODEL_PATH),
            )
            reranker_model_path = timings.get("reranker_model_path") if timings else get_current_reranker_path()
            reranker_mode = timings.get("reranker_mode") if timings else None

        vector_limit_used = timings.get("vector_limit_used") if timings else vector_limit
        content_char_limit_used = timings.get("content_char_limit_used") if timings else content_char_limit

        return RAGResponse(
            answer="I could not find relevant information in the knowledge base.",
            citations=[],
            retrieval_time_ms=retrieval_ms,
            confidence=0.0,
            num_chunks_retrieved=0,
            llm_time_ms=0.0,
            total_time_ms=(time.perf_counter() - tic_total) * 1000,
            timings=timings or None,
            models={
                "embedding": embedding_model_path,
                "reranker": reranker_model_path,
                "llm": settings.OPENAI_MODEL if use_llm else "disabled",
            },
            token_usage=None,
            token_cost_usd=0.0,
            llm_used=False,
            reranker_mode=reranker_mode,
            vector_limit_used=vector_limit_used,
            content_char_limit_used=content_char_limit_used,
        )

    # Generate answer with LLM or simple concatenation
    llm_model = os.getenv("OPENAI_MODEL") or settings.OPENAI_MODEL or "Gpt4o"
    token_usage = None
    token_cost_usd = 0.0
    llm_used = use_llm
    if use_llm:
        tic_llm = time.perf_counter()
        answer, token_usage, token_cost_usd = await _generate_answer_with_llm(
            question,
            chunks,
            model=llm_model,
        )
        llm_time_ms = (time.perf_counter() - tic_llm) * 1000
        llm_used = token_usage is not None
    else:
        # Simple concatenation (for evaluation/debugging)
        answer_parts = [chunk.content for chunk in chunks[: min(5, len(chunks))]]
        answer = " ".join(answer_parts)
        llm_time_ms = 0.0
        llm_used = False

    citations = [
        Citation(
            source=chunk.source,
            content=chunk.content,
            score=chunk.score,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]

    top_confidence = max(chunk.score for chunk in chunks)
    total_time_ms = (time.perf_counter() - tic_total) * 1000
    timings = timings or {}
    timings.update({
        "llm_ms": llm_time_ms,
        "end_to_end_ms": total_time_ms,
    })

    if inference_config.ENABLE_REMOTE_INFERENCE:
        embedding_model_path = inference_config.EMBEDDING_SERVICE_URL or "remote"
        reranker_model_path = inference_config.RERANK_SERVICE_URL or "remote"
        reranker_mode = "remote"
    else:
        embedding_model_path = timings.get("embedding_model_path") or getattr(
            get_embedding_model(),
            "resolved_model_path",
            getattr(get_embedding_model(), "configured_path", settings.ONNX_EMBED_MODEL_PATH),
        )
        reranker_model_path = timings.get("reranker_model_path") or getattr(
            get_reranker_model(),
            "resolved_model_path",
            getattr(get_reranker_model(), "model_path", settings.ONNX_RERANK_MODEL_PATH),
        )
        reranker_mode = timings.get("reranker_mode")

    vector_limit_used = timings.get("vector_limit_used")
    content_char_limit_used = timings.get("content_char_limit_used")

    return RAGResponse(
        answer=answer,
        citations=citations,
        retrieval_time_ms=retrieval_ms,
        confidence=top_confidence,
        num_chunks_retrieved=len(chunks),
        llm_time_ms=llm_time_ms,
        total_time_ms=total_time_ms,
        timings=timings or None,
        models={
            "embedding": embedding_model_path,
            "reranker": reranker_model_path,
            "llm": llm_model if use_llm else "disabled",
        },
        token_usage=token_usage,
        token_cost_usd=token_cost_usd,
        llm_used=llm_used,
        reranker_mode=reranker_mode,
        vector_limit_used=vector_limit_used,
        content_char_limit_used=content_char_limit_used,
    )


async def ingest_documents_batch(documents: List[Tuple[str, str, str]]) -> List[DocumentResponse]:
    """Helper to ingest multiple documents concurrently."""
    tasks = [
        ingest_document(title=title, content=content, source=source)
        for title, content, source in documents
    ]
    return await asyncio.gather(*tasks)
