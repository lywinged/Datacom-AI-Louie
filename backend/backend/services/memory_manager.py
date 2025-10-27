from __future__ import annotations

import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, List, Sequence

from sqlalchemy import func, select, or_, literal, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import KnowledgeChunk, KnowledgeDocument
from utils.rag import embed_texts, ingest_document
from services.metrics import (
    role_profile_base_sync_counter,
    role_profile_diary_sync_counter,
    role_profile_section_sync_counter,
)
from services.summarizer import get_summarizer
from knowledge_config.knowledge_config import knowledge_config

logger = logging.getLogger(__name__)
sync_logger = logging.getLogger("knowledge.sync")


def _json_text(column, key: str):
    return column.op("->>")(literal(key))


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash for deduplication."""
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _check_duplicate_chunk(
    db: AsyncSession,
    content_hash: str,
    document_id: int | None = None,
    scope: str = "document"
) -> KnowledgeChunk | None:
    """
    Check if a chunk with this content hash already exists.

    Args:
        db: Database session
        content_hash: Content hash to check
        document_id: Document ID for document-level deduplication
        scope: "document", "role", or "global"

    Returns:
        Existing chunk if found, None otherwise
    """
    if scope == "document" and document_id:
        stmt = select(KnowledgeChunk).where(
            KnowledgeChunk.content_hash == content_hash,
            KnowledgeChunk.document_id == document_id
        ).limit(1)
    elif scope == "global":
        stmt = select(KnowledgeChunk).where(
            KnowledgeChunk.content_hash == content_hash
        ).limit(1)
    else:
        # For role-level, would need to join with document
        # Not implemented here for simplicity
        return None

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _maybe_summarize_content(content: str) -> tuple[str, bool, float]:
    """
    Apply summarization if configured and content is long.

    Returns:
        Tuple of (content, was_summarized, cost)
    """
    if not knowledge_config.KNOWLEDGE_SUMMARY_ENABLED:
        return content, False, 0.0

    summarizer = get_summarizer()

    # Check if content should be summarized
    if not summarizer.should_summarize(
        content,
        min_tokens=knowledge_config.SUMMARY_TOKEN_THRESHOLD
    ):
        return content, False, 0.0

    # Perform summarization
    if knowledge_config.SUMMARY_METHOD == "extractive":
        summary = summarizer.extractive_summary(
            content,
            max_tokens=knowledge_config.SUMMARY_MAX_TOKENS
        )
        return summary, True, 0.0
    elif knowledge_config.SUMMARY_METHOD == "llm":
        summary, cost = await summarizer.llm_summary(
            content,
            model=knowledge_config.SUMMARY_LLM_MODEL,
            max_tokens=knowledge_config.SUMMARY_MAX_TOKENS,
            style=knowledge_config.SUMMARY_STYLE
        )
        return summary, True, cost
    else:  # hybrid
        summary, cost, _ = await summarizer.hybrid_summary(
            content,
            token_threshold=knowledge_config.SUMMARY_TOKEN_THRESHOLD,
            target_tokens=knowledge_config.SUMMARY_MAX_TOKENS,
            use_llm_for_long=knowledge_config.SUMMARY_USE_LLM_FOR_LONG
        )
        return summary, True, cost

async def _get_or_create_conversation_document(
    db: AsyncSession,
    *,
    conversation_id: int,
    role_id: Optional[int],
    owner_user_id: Optional[int],
    context_type: str,
) -> KnowledgeDocument:
    stmt = (
        select(KnowledgeDocument)
        .where(
            _json_text(KnowledgeDocument.meta, "type") == context_type,
            _json_text(KnowledgeDocument.meta, "conversation_id")
            == str(conversation_id),
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    document = result.scalars().first()
    if document:
        return document

    document = KnowledgeDocument(
        title=f"{context_type.title()} #{conversation_id}",
        description="Auto-ingested conversation memory",
        owner_id=owner_user_id,
        role_id=role_id,
        source=context_type,
        meta={
            "type": context_type,
            "conversation_id": conversation_id,
            "role_id": role_id,
        },
    )
    db.add(document)
    await db.flush()
    return document


async def _chunk_exists(
    db: AsyncSession,
    *,
    document_id: int,
    user_message_id: Optional[int],
    assistant_message_id: Optional[int],
) -> bool:
    conditions = [KnowledgeChunk.document_id == document_id]
    if user_message_id is not None:
        conditions.append(
            _json_text(KnowledgeChunk.meta, "user_message_id")
            == str(user_message_id)
        )
    if assistant_message_id is not None:
        conditions.append(
            _json_text(KnowledgeChunk.meta, "assistant_message_id")
            == str(assistant_message_id)
        )
    if len(conditions) == 1:
        return False
    stmt = select(func.count(KnowledgeChunk.id)).where(*conditions)
    result = await db.execute(stmt)
    return (result.scalar() or 0) > 0


def _format_turn_content(
    *,
    user_message: Optional[str],
    user_label: str,
    user_timestamp: Optional[datetime],
    assistant_message: Optional[str],
    assistant_label: str,
    assistant_timestamp: Optional[datetime],
) -> Optional[str]:
    segments = []
    if user_message:
        time_str = (
            user_timestamp.isoformat() if isinstance(user_timestamp, datetime) else ""
        )
        header = f"{user_label}{f' @ {time_str}' if time_str else ''}"
        segments.append(f"{header}:\n{user_message.strip()}")
    if assistant_message:
        time_str = (
            assistant_timestamp.isoformat()
            if isinstance(assistant_timestamp, datetime)
            else ""
        )
        header = f"{assistant_label}{f' @ {time_str}' if time_str else ''}"
        segments.append(f"{header}:\n{assistant_message.strip()}")

    if not segments:
        return None
    return "\n\n".join(segments)


async def store_turn(
    db: AsyncSession,
    *,
    context_type: str,
    conversation_id: int,
    role_id: Optional[int],
    owner_user_id: Optional[int],
    user_message: Optional[str],
    user_message_id: Optional[int],
    user_timestamp: Optional[datetime],
    assistant_message: Optional[str],
    assistant_message_id: Optional[int],
    assistant_timestamp: Optional[datetime],
    user_label: str = "User",
    assistant_label: str = "Assistant",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a single conversation turn into the knowledge base as long-term memory.
    """
    content = _format_turn_content(
        user_message=user_message,
        user_label=user_label,
        user_timestamp=user_timestamp,
        assistant_message=assistant_message,
        assistant_label=assistant_label,
        assistant_timestamp=assistant_timestamp,
    )
    if not content:
        logger.debug(
            "Vector store skipped: empty content conversation=%s type=%s",
            conversation_id,
            context_type,
        )
        return

    document = await _get_or_create_conversation_document(
        db,
        conversation_id=conversation_id,
        role_id=role_id,
        owner_user_id=owner_user_id,
        context_type=context_type,
    )

    if await _chunk_exists(
        db,
        document_id=document.id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    ):
        logger.debug(
            "Vector store skipped: chunk already exists conversation=%s user_msg=%s assistant_msg=%s",
            conversation_id,
            user_message_id,
            assistant_message_id,
        )
        return

    result = await db.execute(
        select(func.max(KnowledgeChunk.chunk_index)).where(
            KnowledgeChunk.document_id == document.id
        )
    )
    next_index = (result.scalar() or -1) + 1

    # Compute content hash for deduplication
    content_hash = _compute_content_hash(content)

    # Check for duplicates if enabled
    if knowledge_config.KNOWLEDGE_DEDUPE_ENABLED:
        existing_chunk = await _check_duplicate_chunk(
            db,
            content_hash=content_hash,
            document_id=document.id,
            scope=knowledge_config.DEDUPE_SCOPE
        )
        if existing_chunk:
            logger.info(
                "Duplicate chunk detected - skipping. Hash=%s existing_id=%s",
                content_hash[:8],
                existing_chunk.id
            )
            return

    # Apply summarization if enabled
    summarize_cost = 0.0
    is_summary = False
    original_content = content
    if knowledge_config.KNOWLEDGE_SUMMARY_ENABLED:
        content, is_summary, summarize_cost = await _maybe_summarize_content(content)
        if is_summary:
            logger.info(
                "Content summarized: original=%d chars, summary=%d chars, cost=$%.6f",
                len(original_content),
                len(content),
                summarize_cost
            )

    embedding = None
    try:
        embedding = (await embed_texts([content]))[0]
    except Exception as exc:  # pragma: no cover - embedding optional
        logger.exception("Failed to embed conversation memory: %s", exc)

    metadata: Dict[str, Any] = {
        "type": context_type,
        "conversation_id": conversation_id,
        "role_id": role_id,
        "owner_user_id": owner_user_id,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "user_label": user_label,
        "assistant_label": assistant_label,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    # Add summarization metadata
    if is_summary:
        metadata["summarized"] = True
        metadata["summary_cost"] = summarize_cost
        metadata["original_length"] = len(original_content)

    chunk = KnowledgeChunk(
        document_id=document.id,
        chunk_index=next_index,
        content=content,
        content_hash=content_hash,
        is_summary=is_summary,
        embedding=embedding,
        meta=metadata,
        token_count=len(content),
    )
    db.add(chunk)
    try:
        await db.commit()
        logger.info(
            "Vector stored type=%s conversation=%s role_id=%s owner=%s chunk_id=%s document_id=%s user_msg=%s assistant_msg=%s",
            context_type,
            conversation_id,
            role_id,
            owner_user_id,
            chunk.id,
            document.id,
            user_message_id,
            assistant_message_id,
        )
    except Exception:
        logger.exception("Failed to persist conversation memory chunk")
        await db.rollback()


async def store_role_chat_turn(
    db: AsyncSession,
    *,
    conversation_id: int,
    role_id: int,
    user_id: int,
    user_message: str,
    user_message_id: int,
    user_timestamp: datetime,
    assistant_message: Optional[str],
    assistant_message_id: Optional[int],
    assistant_timestamp: Optional[datetime],
    role_name: Optional[str],
) -> None:
    assistant_label = role_name or f"Role#{role_id}"
    await store_turn(
        db,
        context_type="role_chat",
        conversation_id=conversation_id,
        role_id=role_id,
        owner_user_id=user_id,
        user_message=user_message,
        user_message_id=user_message_id,
        user_timestamp=user_timestamp,
        assistant_message=assistant_message,
        assistant_message_id=assistant_message_id,
        assistant_timestamp=assistant_timestamp,
        assistant_label=assistant_label,
    )


async def store_story_chat_turn(
    db: AsyncSession,
    *,
    conversation_id: int,
    story_id: int,
    user_id: int,
    user_message: str,
    user_message_id: int,
    user_timestamp: datetime,
    assistant_message: Optional[str],
    assistant_message_id: Optional[int],
    assistant_timestamp: Optional[datetime],
    playing_role_index: Optional[int],
    assistant_role_label: Optional[str],
) -> None:
    extra_meta = {
        "story_id": story_id,
        "playing_role_index": playing_role_index,
    }
    await store_turn(
        db,
        context_type="story_chat",
        conversation_id=conversation_id,
        role_id=None,
        owner_user_id=user_id,
        user_message=user_message,
        user_message_id=user_message_id,
        user_timestamp=user_timestamp,
        assistant_message=assistant_message,
        assistant_message_id=assistant_message_id,
        assistant_timestamp=assistant_timestamp,
        user_label="StoryUser",
        assistant_label=assistant_role_label or "StoryAgent",
        extra_metadata=extra_meta,
    )


async def knowledge_available_for_role(
    db: AsyncSession,
    role_id: Optional[Any],
) -> bool:
    role_ids: list[Optional[int]]
    if isinstance(role_id, (list, tuple, set)):
        role_ids = list(role_id)
    else:
        role_ids = [role_id]

    conditions = [KnowledgeDocument.role_id.is_(None)]
    for rid in role_ids:
        conditions.append(KnowledgeDocument.role_id == rid)

    stmt = (
        select(func.count(KnowledgeChunk.id))
        .join(
            KnowledgeDocument,
            KnowledgeDocument.id == KnowledgeChunk.document_id,
        )
        .where(or_(*conditions))
    )
    result = await db.execute(stmt)
    return (result.scalar() or 0) > 0


async def _delete_chunks(
    db: AsyncSession,
    condition,
) -> None:
    result = await db.execute(
        select(KnowledgeChunk.id, KnowledgeChunk.document_id).where(condition)
    )
    rows = result.all()
    if not rows:
        return

    chunk_ids = [row[0] for row in rows]
    document_ids = {row[1] for row in rows}

    await db.execute(
        delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids))
    )
    await db.commit()

    empty_docs: list[int] = []
    for doc_id in document_ids:
        res = await db.execute(
            select(func.count(KnowledgeChunk.id)).where(
                KnowledgeChunk.document_id == doc_id
            )
        )
        if (res.scalar() or 0) == 0:
            empty_docs.append(doc_id)

    if empty_docs:
        await db.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.id.in_(empty_docs)
            )
        )
        await db.commit()
    logger.info(
        "Vector chunks deleted chunk_ids=%s empty_documents=%s",
        chunk_ids,
        empty_docs,
    )


async def remove_role_chat_vectors(
    db: AsyncSession,
    *,
    message_ids: Optional[List[int]] = None,
    conversation_id: Optional[int] = None,
) -> None:
    filters = []
    if message_ids:
        str_ids = [str(mid) for mid in message_ids]
        filters.append(_json_text(KnowledgeChunk.meta, "user_message_id").in_(str_ids))
        filters.append(
            _json_text(KnowledgeChunk.meta, "assistant_message_id").in_(str_ids)
        )
    if conversation_id is not None:
        filters.append(
            _json_text(KnowledgeChunk.meta, "conversation_id")
            == str(conversation_id)
        )

    if not filters:
        return

    await _delete_chunks(db, or_(*filters))
    logger.info(
        "Role chat vectors removed conversation_id=%s message_ids=%s",
        conversation_id,
        message_ids,
    )


async def remove_story_chat_vectors(
    db: AsyncSession,
    *,
    message_ids: Optional[List[int]] = None,
    conversation_id: Optional[int] = None,
    story_id: Optional[int] = None,
) -> None:
    filters = []
    if message_ids:
        str_ids = [str(mid) for mid in message_ids]
        filters.append(_json_text(KnowledgeChunk.meta, "user_message_id").in_(str_ids))
        filters.append(
            _json_text(KnowledgeChunk.meta, "assistant_message_id").in_(str_ids)
        )
    if conversation_id is not None:
        filters.append(
            _json_text(KnowledgeChunk.meta, "conversation_id")
            == str(conversation_id)
        )
    if story_id is not None:
        filters.append(_json_text(KnowledgeChunk.meta, "story_id") == str(story_id))

    if not filters:
        return

    await _delete_chunks(db, or_(*filters))
    logger.info(
        "Story chat vectors removed story_id=%s conversation_id=%s message_ids=%s",
        story_id,
        conversation_id,
        message_ids,
    )


async def upsert_profile_document(
    db: AsyncSession,
    *,
    doc_type: str,
    title: str,
    content: str,
    owner_user_id: Optional[int] = None,
    role_id: Optional[int] = None,
    story_id: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    metadata: Dict[str, Any] = dict(meta or {})
    metadata["type"] = doc_type
    if role_id is not None:
        metadata["role_id"] = role_id
    if story_id is not None:
        metadata["story_id"] = story_id

    stmt = select(KnowledgeDocument).where(
        _json_text(KnowledgeDocument.meta, "type") == doc_type
    )
    if role_id is not None:
        stmt = stmt.where(KnowledgeDocument.role_id == role_id)
    if story_id is not None:
        stmt = stmt.where(
            _json_text(KnowledgeDocument.meta, "story_id") == str(story_id)
        )

    result = await db.execute(stmt.limit(1))
    document = result.scalars().first()

    if not content.strip():
        if document:
            await _delete_chunks(db, KnowledgeChunk.document_id == document.id)
        else:
            logger.debug(
                "Profile document remove skipped (no existing doc) type=%s role_id=%s story_id=%s",
                doc_type,
                role_id,
                story_id,
            )
        return

    if document is None:
        document = KnowledgeDocument(
            title=title,
            description=f"{doc_type} summary",
            owner_id=owner_user_id,
            role_id=role_id,
            meta=metadata,
        )
        db.add(document)
        await db.flush()
    else:
        document.title = title
        document.meta = metadata
        if owner_user_id is not None:
            document.owner_id = owner_user_id
        if role_id is not None:
            document.role_id = role_id

        await db.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.document_id == document.id
            )
        )

    embedding = None
    try:
        embedding = (await embed_texts([content]))[0]
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to embed profile document type=%s: %s", doc_type, exc)

    chunk = KnowledgeChunk(
        document_id=document.id,
        chunk_index=0,
        content=content,
        embedding=embedding,
        meta=metadata,
        token_count=len(content),
    )
    db.add(chunk)
    await db.commit()
    logger.info(
        "Profile document stored type=%s role_id=%s story_id=%s document_id=%s",
        doc_type,
        role_id,
        story_id,
        document.id,
    )


async def _clone_document_with_chunks(
    db: AsyncSession,
    *,
    source_doc: KnowledgeDocument,
    source_chunks: Optional[Sequence[KnowledgeChunk]] = None,
    title: str,
    owner_user_id: Optional[int],
    role_id: Optional[int] = None,
    meta_override: Optional[Dict[str, Any]] = None,
) -> KnowledgeDocument:
    base_meta = dict(source_doc.meta or {})
    if meta_override:
        base_meta.update(meta_override)

    cloned_doc = KnowledgeDocument(
        title=title,
        description=source_doc.description,
        owner_id=owner_user_id,
        role_id=role_id if role_id is not None else source_doc.role_id,
        source=source_doc.source,
        language=source_doc.language,
        meta=base_meta,
    )
    db.add(cloned_doc)
    await db.flush()

    if source_chunks is None:
        chunks_stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == source_doc.id)
            .order_by(KnowledgeChunk.chunk_index)
        )
        result = await db.execute(chunks_stmt)
        source_chunks = list(result.scalars())

    for chunk in source_chunks:
        chunk_meta = dict(chunk.meta or {})
        if meta_override:
            chunk_meta.update(meta_override)
        cloned_chunk = KnowledgeChunk(
            document_id=cloned_doc.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=chunk.embedding,
            meta=chunk_meta,
            token_count=chunk.token_count,
        )
        db.add(cloned_chunk)

    await db.flush()
    return cloned_doc


async def clone_role_profile_document_if_same(
    db: AsyncSession,
    *,
    source_role_id: Optional[int],
    target_role_id: int,
    owner_user_id: Optional[int],
    title: str,
    content: str,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """Clone existing role profile document (and its chunks) if content matches.

    Returns True when cloning succeeded, False to indicate normal upsert path should run.
    """
    if source_role_id is None:
        return False

    stmt = (
        select(KnowledgeDocument)
        .where(
            KnowledgeDocument.role_id == source_role_id,
            _json_text(KnowledgeDocument.meta, "type") == "role_profile",
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    source_doc = result.scalars().first()
    if not source_doc:
        return False

    chunks_result = await db.execute(
        select(KnowledgeChunk)
        .where(KnowledgeChunk.document_id == source_doc.id)
        .order_by(KnowledgeChunk.chunk_index)
    )
    source_chunks: List[KnowledgeChunk] = list(chunks_result.scalars())
    if not source_chunks:
        return False

    source_content = "\n".join(chunk.content for chunk in source_chunks)
    if source_content != content:
        return False

    merged_meta: Dict[str, Any] = dict(source_doc.meta or {})
    if meta:
        merged_meta.update(meta)
    merged_meta["type"] = "role_profile"
    merged_meta["cloned_from_role_id"] = source_role_id

    cloned_doc = await _clone_document_with_chunks(
        db,
        source_doc=source_doc,
        source_chunks=source_chunks,
        title=title,
        owner_user_id=owner_user_id,
        role_id=target_role_id,
        meta_override=merged_meta,
    )
    await db.commit()
    logger.info(
        "Role profile cloned source_role_id=%s target_role_id=%s document_id=%s",
        source_role_id,
        target_role_id,
        cloned_doc.id,
    )
    sync_logger.info(
        "Role profile cloned role_id=%s source_role_id=%s document_id=%s",
        target_role_id,
        source_role_id,
        cloned_doc.id,
    )
    role_profile_base_sync_counter.labels(action="cloned").inc()
    return True


async def _delete_document_with_chunks(
    db: AsyncSession,
    document: KnowledgeDocument,
) -> None:
    await _delete_chunks(db, KnowledgeChunk.document_id == document.id)
    await db.delete(document)


async def _load_role_docs_by_type(
    db: AsyncSession,
    role_id: int,
    doc_type: str,
) -> List[KnowledgeDocument]:
    stmt = (
        select(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .where(
            KnowledgeDocument.role_id == role_id,
            _json_text(KnowledgeDocument.meta, "type") == doc_type,
        )
    )
    result = await db.execute(stmt)
    return list(result.scalars())


async def _sync_entry_documents(
    db: AsyncSession,
    *,
    entries: Sequence[Dict[str, Any]] | None,
    role_id: int,
    owner_user_id: Optional[int],
    role_meta: Dict[str, Any],
    doc_type: str,
    key_meta: str,
    title_meta: str,
    default_title_prefix: str,
    counter,
    log_category: str,
    source_role_id: Optional[int],
    extra_meta_map: Optional[Dict[str, str]] = None,
) -> None:
    extra_meta_map = extra_meta_map or {}
    normalized_entries: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for idx, entry in enumerate(entries or []):
        content = str(entry.get("content") or "").strip()
        if not content:
            continue
        raw_key = entry.get("key") or f"{doc_type}-{idx}"
        key = str(raw_key)
        if key in seen_keys:
            key = f"{key}-{idx}"
        seen_keys.add(key)
        title = entry.get("title") or f"{default_title_prefix} {idx + 1}"
        entry_hash = entry.get("hash")
        if not entry_hash:
            entry_hash = _hash_text(content)
        normalized: Dict[str, Any] = {
            "key": key,
            "title": title,
            "content": content,
            "hash": entry_hash,
        }
        for field in extra_meta_map.keys():
            if field in entry and entry[field] is not None:
                normalized[field] = entry[field]
        normalized_entries.append(normalized)

    existing_docs = await _load_role_docs_by_type(db, role_id, doc_type)
    docs_to_remove = set(existing_docs)
    existing_by_key: Dict[str, KnowledgeDocument] = {}
    for doc in existing_docs:
        meta = doc.meta or {}
        stored_key = meta.get(key_meta)
        if stored_key:
            existing_by_key[str(stored_key)] = doc

    source_docs_by_hash: Dict[str, KnowledgeDocument] = {}
    if source_role_id:
        source_docs = await _load_role_docs_by_type(db, source_role_id, doc_type)
        for doc in source_docs:
            meta = doc.meta or {}
            hash_value = meta.get("content_hash")
            if not hash_value or not isinstance(hash_value, str):
                joined = "\n".join(chunk.content for chunk in doc.chunks)
                hash_value = _hash_text(joined)
            source_docs_by_hash[hash_value] = doc

    for entry in normalized_entries:
        key = entry["key"]
        existing_doc = existing_by_key.get(key)
        entry_meta = {
            "role_name": role_meta.get("role_name"),
            "card_name": role_meta.get("card_name"),
            "creator": role_meta.get("creator"),
            "type": doc_type,
            key_meta: key,
            title_meta: entry["title"],
            "content_hash": entry["hash"],
        }
        for field, meta_name in extra_meta_map.items():
            value = entry.get(field)
            if value:
                entry_meta[meta_name] = value

        if existing_doc:
            docs_to_remove.discard(existing_doc)
            existing_hash = (existing_doc.meta or {}).get("content_hash")
            if existing_hash == entry["hash"]:
                new_meta = dict(existing_doc.meta or {})
                new_meta.update(entry_meta)
                existing_doc.meta = new_meta
                existing_doc.title = entry["title"]
                counter.labels(action="reused").inc()
                sync_logger.info(
                    "Role %s reused role_id=%s %s=%s hash=%s",
                    log_category,
                    role_id,
                    key_meta,
                    key,
                    entry["hash"],
                )
                continue
            await _delete_document_with_chunks(db, existing_doc)
            await db.flush()
            counter.labels(action="deleted").inc()
            sync_logger.info(
                "Role %s removed before refresh role_id=%s %s=%s old_hash=%s",
                log_category,
                role_id,
                key_meta,
                key,
                existing_hash,
            )

        cloned_doc = None
        source_doc = source_docs_by_hash.get(entry["hash"])
        if source_doc:
            cloned_meta = dict(source_doc.meta or {})
            cloned_meta.update(entry_meta)
            cloned_meta["cloned_from_role_id"] = source_role_id
            cloned_doc = await _clone_document_with_chunks(
                db,
                source_doc=source_doc,
                source_chunks=list(source_doc.chunks),
                title=entry["title"],
                owner_user_id=owner_user_id,
                role_id=role_id,
                meta_override=cloned_meta,
            )
            docs_to_remove.discard(cloned_doc)

        if cloned_doc:
            counter.labels(action="cloned").inc()
            sync_logger.info(
                "Role %s cloned role_id=%s %s=%s hash=%s source_role_id=%s",
                log_category,
                role_id,
                key_meta,
                key,
                entry["hash"],
                source_role_id,
            )
            continue

        await ingest_document(
            db,
            title=entry["title"],
            text=entry["content"],
            owner_id=owner_user_id,
            role_id=role_id,
            metadata=entry_meta,
        )
        counter.labels(action="embedded").inc()
        sync_logger.info(
            "Role %s embedded role_id=%s %s=%s hash=%s",
            log_category,
            role_id,
            key_meta,
            key,
            entry["hash"],
        )

    if docs_to_remove:
        for doc in docs_to_remove:
            meta = doc.meta or {}
            key_value = meta.get(key_meta)
            await _delete_document_with_chunks(db, doc)
            counter.labels(action="deleted").inc()
            sync_logger.info(
                "Role %s deleted role_id=%s %s=%s",
                log_category,
                doc.role_id,
                key_meta,
                key_value,
            )


async def sync_role_profile_documents(
    db: AsyncSession,
    *,
    role_id: int,
    owner_user_id: Optional[int],
    role_meta: Dict[str, Any],
    base_content: str,
    diary_entries: Sequence[Dict[str, Any]],
    section_entries: Sequence[Dict[str, Any]] | None = None,
    worldbook_entries: Sequence[Dict[str, Any]] | None = None,
    status_entries: Sequence[Dict[str, Any]] | None = None,
    source_role_id: Optional[int] = None,
) -> None:
    # 1) core profile (excluding diary entries)
    base_meta = dict(role_meta)
    cloned = await clone_role_profile_document_if_same(
        db,
        source_role_id=source_role_id,
        target_role_id=role_id,
        owner_user_id=owner_user_id,
        title=f"Role Profile #{role_id}",
        content=base_content,
        meta=base_meta,
    )
    if not cloned:
        await upsert_profile_document(
            db,
            doc_type="role_profile",
            title=f"Role Profile #{role_id}",
            content=base_content,
            role_id=role_id,
            owner_user_id=owner_user_id,
            meta=base_meta,
        )
        role_profile_base_sync_counter.labels(action="embedded").inc()
        sync_logger.info(
            "Role profile embedded role_id=%s document_type=core",
            role_id,
        )

    await _sync_entry_documents(
        db,
        entries=diary_entries,
        role_id=role_id,
        owner_user_id=owner_user_id,
        role_meta=role_meta,
        doc_type="role_diary",
        key_meta="entry_key",
        title_meta="entry_title",
        default_title_prefix="Diary",
        counter=role_profile_diary_sync_counter,
        log_category="diary",
        source_role_id=source_role_id,
        extra_meta_map={"date": "entry_date"},
    )

    await _sync_entry_documents(
        db,
        entries=section_entries,
        role_id=role_id,
        owner_user_id=owner_user_id,
        role_meta=role_meta,
        doc_type="role_profile_section",
        key_meta="section_key",
        title_meta="section_title",
        default_title_prefix="Section",
        counter=role_profile_section_sync_counter,
        log_category="section",
        source_role_id=source_role_id,
    )

    await _sync_entry_documents(
        db,
        entries=worldbook_entries,
        role_id=role_id,
        owner_user_id=owner_user_id,
        role_meta=role_meta,
        doc_type="role_worldbook",
        key_meta="worldbook_key",
        title_meta="worldbook_title",
        default_title_prefix="Worldview",
        counter=role_profile_section_sync_counter,
        log_category="worldbook",
        source_role_id=source_role_id,
    )

    await _sync_entry_documents(
        db,
        entries=status_entries,
        role_id=role_id,
        owner_user_id=owner_user_id,
        role_meta=role_meta,
        doc_type="role_status",
        key_meta="status_key",
        title_meta="status_title",
        default_title_prefix="Status",
        counter=role_profile_section_sync_counter,
        log_category="status",
        source_role_id=source_role_id,
    )

    await db.commit()
