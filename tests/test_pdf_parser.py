import io
import pytest
from pypdf import PdfWriter
from backend.utils.pdf_parser import (
    clean_text,
    extract_text_from_pdf,
    chunk_text,
    extract_resume_sections,
)


def create_sample_pdf_bytes(text_pages: list[str]) -> bytes:
    """Helper to create an in-memory PDF byte stream for testing."""
    writer = PdfWriter()
    for text in text_pages:
        page = writer.add_blank_page(width=612, height=792)
        # Note: PdfWriter blank page can be saved; for testing basic structure:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_clean_text():
    raw_text = "  Hello \r\n\r\n World! \n\n\n\n  This is a   test. \x00 "
    cleaned = clean_text(raw_text)
    assert "Hello" in cleaned
    assert "World!" in cleaned
    assert "\x00" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_short():
    text = "Short sentence for testing chunker."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_text"] == text


def test_chunk_text_long():
    text = (
        "FastAPI is a modern web framework for Python. "
        "LangChain helps build LLM applications. "
        "LangGraph enables complex multi-agent workflows. "
        "pgvector allows fast vector similarity search in PostgreSQL. "
        "Next.js provides a high-performance frontend framework. "
    ) * 10

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert len(chunk["chunk_text"]) <= 350
        assert isinstance(chunk["chunk_text"], str)


def test_extract_resume_sections():
    sample_resume = """
SUMMARY
Experienced AI Engineer specializing in LLMs and FastAPI.

SKILLS
Python, FastAPI, LangChain, PostgreSQL, React, Docker

WORK EXPERIENCE
Senior AI Developer - Tech Corp
Built multi-agent chat system using LangGraph and pgvector.

EDUCATION
B.Tech in Computer Science - State University

PROJECTS
Interview Copilot AI - Fullstack multi-agent platform.
"""
    sections = extract_resume_sections(sample_resume)
    assert "Experienced AI Engineer" in sections["summary"]
    assert "Python, FastAPI" in sections["skills"]
    assert "Senior AI Developer" in sections["experience"]
    assert "B.Tech in Computer Science" in sections["education"]
    assert "Interview Copilot AI" in sections["projects"]
