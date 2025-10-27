"""
Knowledge Cold/Hot Tier Management Service

Handles automatic archival of cold data and lazy rehydration of accessed cold documents.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update, func, and_, or_, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from models import KnowledgeDocument, KnowledgeChunk
from services.metrics import (
    knowledge_tier_operations_counter,
    knowledge_cold_documents_gauge,
    knowledge_hot_documents_gauge,
    knowledge_cold_chunks_gauge,
)
from utils.rag import embed_texts

logger = logging.getLogger(__name__)

# Configuration from environment variables
COLD_THRESHOLD_DAYS = int(os.getenv("KNOWLEDGE_COLD_THRESHOLD_DAYS", "30"))  # Days of inactivity before archival
ARCHIVE_BATCH_SIZE = int(os.getenv("KNOWLEDGE_ARCHIVE_BATCH_SIZE", "100"))  # Batch size for archival
REHYDRATE_BATCH_SIZE = int(os.getenv("KNOWLEDGE_REHYDRATE_BATCH_SIZE", "50"))  # Batch size for rehydration


async def track_document_access(
    db: AsyncSession,
    document_id: int,
) -> None:
    """
    Track document access by incrementing access_count and updating last_access_at.

    This should be called whenever a document is retrieved in RAG search.
    """
    try:
        await db.execute(
            update(KnowledgeDocument)
            .where(KnowledgeDocument.id == document_id)
            .values(
                access_count=KnowledgeDocument.access_count + 1,
                last_access_at=func.now(),
            )
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to track document access for document_id={document_id}: {e}")
        await db.rollback()


async def track_chunk_usage(
    db: AsyncSession,
    chunk_id: int,
) -> None:
    """
    Track chunk usage by updating last_used_at.

    This should be called whenever a chunk is returned in search results.
    """
    try:
        await db.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id == chunk_id)
            .values(last_used_at=func.now())
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to track chunk usage for chunk_id={chunk_id}: {e}")
        await db.rollback()


async def archive_cold_documents(
    db: AsyncSession,
    cold_threshold_days: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Archive documents that haven't been accessed in the specified threshold period.

    Archival process:
    1. Mark document as is_cold = TRUE
    2. Set chunk embeddings to NULL to save vector storage space
    3. Keep full-text content for fallback search

    Args:
        db: Database session
        cold_threshold_days: Days of inactivity before archival (default from env)
        batch_size: Maximum number of documents to archive in one run
        dry_run: If True, only count documents without actually archiving

    Returns:
        dict with archival statistics
    """
    threshold_days = cold_threshold_days or COLD_THRESHOLD_DAYS
    batch = batch_size or ARCHIVE_BATCH_SIZE

    threshold_date = datetime.utcnow() - timedelta(days=threshold_days)

    # Find documents to archive
    query = select(KnowledgeDocument).where(
        and_(
            KnowledgeDocument.is_cold == False,  # noqa: E712
            or_(
                KnowledgeDocument.last_access_at < threshold_date,
                KnowledgeDocument.last_access_at.is_(None),  # Never accessed
            ),
        )
    ).limit(batch)

    result = await db.execute(query)
    documents_to_archive = result.scalars().all()

    if dry_run:
        return {
            "dry_run": True,
            "documents_found": len(documents_to_archive),
            "threshold_days": threshold_days,
            "threshold_date": threshold_date.isoformat(),
        }

    archived_count = 0
    chunks_archived = 0
    errors = 0

    for doc in documents_to_archive:
        try:
            # Mark document as cold
            doc.is_cold = True

            # Set chunk embeddings to NULL
            result = await db.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.document_id == doc.id)
                .values(embedding=None)
                .returning(func.count())
            )
            chunk_count = result.scalar() or 0
            chunks_archived += chunk_count

            await db.commit()
            archived_count += 1

            knowledge_tier_operations_counter.labels(operation="archive", outcome="success").inc()

            logger.info(
                f"Archived document_id={doc.id} ('{doc.title}'), "
                f"nullified {chunk_count} chunk embeddings"
            )

        except Exception as e:
            logger.error(f"Failed to archive document_id={doc.id}: {e}")
            await db.rollback()
            errors += 1
            knowledge_tier_operations_counter.labels(operation="archive", outcome="error").inc()

    # Update gauges
    await update_tier_metrics(db)

    return {
        "archived_documents": archived_count,
        "chunks_archived": chunks_archived,
        "errors": errors,
        "threshold_days": threshold_days,
        "threshold_date": threshold_date.isoformat(),
    }


async def rehydrate_document(
    db: AsyncSession,
    document_id: int,
) -> dict:
    """
    Rehydrate a cold document by regenerating embeddings for its chunks.

    This is called when a cold document is accessed and needs to be activated.

    Args:
        db: Database session
        document_id: ID of the document to rehydrate

    Returns:
        dict with rehydration statistics
    """
    try:
        # Get document
        result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == document_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            return {"error": "Document not found", "document_id": document_id}

        if not doc.is_cold:
            return {"status": "already_hot", "document_id": document_id}

        # Get chunks with null embeddings
        result = await db.execute(
            select(KnowledgeChunk).where(
                and_(
                    KnowledgeChunk.document_id == document_id,
                    KnowledgeChunk.embedding.is_(None),
                )
            )
        )
        chunks = result.scalars().all()

        if not chunks:
            # No chunks need rehydration, just mark as hot
            doc.is_cold = False
            doc.last_access_at = func.now()
            await db.commit()
            return {
                "status": "rehydrated",
                "document_id": document_id,
                "chunks_rehydrated": 0,
            }

        # Regenerate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = embed_texts(texts)

        rehydrated_count = 0
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.last_used_at = func.now()
            rehydrated_count += 1

        # Mark document as hot
        doc.is_cold = False
        doc.last_access_at = func.now()
        doc.access_count = KnowledgeDocument.access_count + 1

        await db.commit()

        knowledge_tier_operations_counter.labels(operation="rehydrate", outcome="success").inc()

        # Update gauges
        await update_tier_metrics(db)

        logger.info(
            f"Rehydrated document_id={document_id} ('{doc.title}'), "
            f"regenerated {rehydrated_count} chunk embeddings"
        )

        return {
            "status": "rehydrated",
            "document_id": document_id,
            "chunks_rehydrated": rehydrated_count,
        }

    except Exception as e:
        logger.error(f"Failed to rehydrate document_id={document_id}: {e}")
        await db.rollback()
        knowledge_tier_operations_counter.labels(operation="rehydrate", outcome="error").inc()
        return {"error": str(e), "document_id": document_id}


async def update_tier_metrics(db: AsyncSession) -> None:
    """Update Prometheus gauge metrics for cold/hot tier monitoring."""
    try:
        # Count cold documents
        result = await db.execute(
            select(func.count()).select_from(KnowledgeDocument).where(KnowledgeDocument.is_cold == True)  # noqa: E712
        )
        cold_docs = result.scalar() or 0
        knowledge_cold_documents_gauge.set(cold_docs)

        # Count hot documents
        result = await db.execute(
            select(func.count()).select_from(KnowledgeDocument).where(KnowledgeDocument.is_cold == False)  # noqa: E712
        )
        hot_docs = result.scalar() or 0
        knowledge_hot_documents_gauge.set(hot_docs)

        # Count cold chunks (with null embeddings)
        result = await db.execute(
            select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.embedding.is_(None))
        )
        cold_chunks = result.scalar() or 0
        knowledge_cold_chunks_gauge.set(cold_chunks)

    except Exception as e:
        logger.error(f"Failed to update tier metrics: {e}")


async def get_tier_statistics(db: AsyncSession) -> dict:
    """Get current cold/hot tier statistics."""
    try:
        # Document statistics
        result = await db.execute(
            select(
                func.count().label("total"),
                func.sum(func.cast(KnowledgeDocument.is_cold, Integer)).label("cold"),
                func.avg(KnowledgeDocument.access_count).label("avg_access_count"),
            ).select_from(KnowledgeDocument)
        )
        doc_stats = result.one()

        # Chunk statistics
        result = await db.execute(
            select(
                func.count().label("total"),
                func.sum(
                    func.case(
                        (KnowledgeChunk.embedding.is_(None), 1),
                        else_=0,
                    )
                ).label("cold"),
            ).select_from(KnowledgeChunk)
        )
        chunk_stats = result.one()

        return {
            "documents": {
                "total": doc_stats.total or 0,
                "hot": (doc_stats.total or 0) - (doc_stats.cold or 0),
                "cold": doc_stats.cold or 0,
                "avg_access_count": float(doc_stats.avg_access_count or 0),
            },
            "chunks": {
                "total": chunk_stats.total or 0,
                "hot": (chunk_stats.total or 0) - (chunk_stats.cold or 0),
                "cold": chunk_stats.cold or 0,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get tier statistics: {e}")
        return {"error": str(e)}
