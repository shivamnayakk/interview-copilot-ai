import uuid
from typing import List
from sqlmodel import Session, select
from backend.database.models import (
    ResumeData,
    ResumeEmbedding,
    KnowledgeDocument,
    KnowledgeEmbedding,
)
from backend.utils.pdf_parser import extract_text_from_pdf, chunk_text
from backend.rag.embeddings import get_embeddings_batch
from backend.utils.logger import logger


def index_resume(
    session: Session,
    user_id: uuid.UUID,
    pdf_bytes: bytes,
    session_id: uuid.UUID = None,
) -> ResumeData:
    """
    Full pipeline: PDF bytes → extract text → chunk → embed → persist to DB.
    Returns the created ResumeData record.
    """
    logger.info(f"Indexing resume for user {user_id}...")

    # Step 1: Extract clean text from PDF
    raw_text = extract_text_from_pdf(pdf_bytes)

    # Step 2: Persist raw resume data
    resume_data = ResumeData(
        user_id=user_id,
        raw_text=raw_text,
        parsed_json="{}",
        resume_score=0.0,
    )
    session.add(resume_data)
    session.commit()
    session.refresh(resume_data)

    # Step 3: Chunk the text
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    if not chunks:
        logger.warning("Resume text yielded no chunks — skipping embedding.")
        return resume_data

    # Step 4: Batch embed all chunks
    texts = [c["chunk_text"] for c in chunks]
    embeddings = get_embeddings_batch(texts)

    # Step 5: Delete old embeddings for this user (re-index)
    old_embeddings = session.exec(
        select(ResumeEmbedding).where(ResumeEmbedding.user_id == user_id)
    ).all()
    for old in old_embeddings:
        session.delete(old)
    session.commit()

    # Step 6: Persist new embeddings
    for chunk, embedding in zip(chunks, embeddings):
        resume_embed = ResumeEmbedding(
            user_id=user_id,
            session_id=session_id,
            chunk_text=chunk["chunk_text"],
            chunk_index=chunk["chunk_index"],
            embedding=embedding,
        )
        session.add(resume_embed)

    session.commit()
    logger.info(f"Indexed {len(chunks)} resume chunks for user {user_id}.")
    return resume_data


def index_knowledge_document(
    session: Session,
    user_id: uuid.UUID,
    document_title: str,
    pdf_bytes: bytes,
) -> KnowledgeDocument:
    """
    Full pipeline: PDF bytes → extract text → chunk → embed → persist to DB.
    Returns the created KnowledgeDocument record.
    """
    logger.info(f"Indexing knowledge document '{document_title}' for user {user_id}...")

    # Step 1: Extract clean text
    raw_text = extract_text_from_pdf(pdf_bytes)

    # Step 2: Chunk text
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)

    # Step 3: Create KnowledgeDocument record
    doc = KnowledgeDocument(
        user_id=user_id,
        document_title=document_title,
        total_chunks=len(chunks),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    if not chunks:
        logger.warning(f"Document '{document_title}' yielded no chunks.")
        return doc

    # Step 4: Batch embed
    texts = [c["chunk_text"] for c in chunks]
    embeddings = get_embeddings_batch(texts)

    # Step 5: Persist embeddings
    for chunk, embedding in zip(chunks, embeddings):
        knowledge_embed = KnowledgeEmbedding(
            document_id=doc.id,
            user_id=user_id,
            chunk_text=chunk["chunk_text"],
            chunk_index=chunk["chunk_index"],
            embedding=embedding,
        )
        session.add(knowledge_embed)

    session.commit()
    logger.info(f"Indexed {len(chunks)} chunks for document '{document_title}'.")
    return doc
