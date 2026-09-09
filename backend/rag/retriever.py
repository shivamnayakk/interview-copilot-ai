import math
import uuid
from typing import List, Optional, Dict
from sqlmodel import Session, select
from backend.database.models import ResumeEmbedding, KnowledgeEmbedding
from backend.rag.embeddings import get_embedding
from backend.utils.logger import logger


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors (Python fallback for SQLite)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _pgvector_available(session: Session) -> bool:
    """Check if we are connected to PostgreSQL (pgvector capable)."""
    return session.bind.dialect.name == "postgresql"


def search_resume(
    session: Session,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    Find the top_k most relevant resume chunks for a given query.
    Uses pgvector cosine similarity on PostgreSQL, Python fallback on SQLite.

    Returns list of dicts: [{'chunk_index': int, 'chunk_text': str, 'score': float}]
    """
    logger.info(f"Searching resume embeddings for user {user_id}, query='{query[:60]}...'")
    query_vector = get_embedding(query)

    # Fetch all user's resume embeddings
    rows = session.exec(
        select(ResumeEmbedding).where(ResumeEmbedding.user_id == user_id)
    ).all()

    if not rows:
        logger.warning(f"No resume embeddings found for user {user_id}.")
        return []

    # Score each chunk
    scored = []
    for row in rows:
        if row.embedding:
            score = _cosine_similarity(query_vector, row.embedding)
            scored.append({
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "score": round(score, 6),
            })

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    logger.info(f"Resume search returned {len(results)} chunks (top score: {results[0]['score'] if results else 'N/A'}).")
    return results


def search_knowledge_base(
    session: Session,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 3,
    document_id: Optional[uuid.UUID] = None,
) -> List[Dict]:
    """
    Find the top_k most relevant knowledge document chunks for a given query.
    Optionally filter by a specific document_id.

    Returns list of dicts: [{'chunk_index': int, 'chunk_text': str, 'score': float, 'document_id': str}]
    """
    logger.info(f"Searching knowledge base for user {user_id}, query='{query[:60]}...'")
    query_vector = get_embedding(query)

    # Build query — optionally filter by document
    stmt = select(KnowledgeEmbedding).where(KnowledgeEmbedding.user_id == user_id)
    if document_id:
        stmt = stmt.where(KnowledgeEmbedding.document_id == document_id)

    rows = session.exec(stmt).all()

    if not rows:
        logger.warning(f"No knowledge embeddings found for user {user_id}.")
        return []

    # Score each chunk
    scored = []
    for row in rows:
        if row.embedding:
            score = _cosine_similarity(query_vector, row.embedding)
            scored.append({
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "score": round(score, 6),
                "document_id": str(row.document_id),
            })

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    logger.info(f"Knowledge base search returned {len(results)} chunks (top score: {results[0]['score'] if results else 'N/A'}).")
    return results
