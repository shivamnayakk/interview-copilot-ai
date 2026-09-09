import io
import re
from typing import BinaryIO, List, Dict, Union
from pypdf import PdfReader
from backend.utils.logger import logger


def clean_text(text: str) -> str:
    """Clean extracted text by normalizing whitespace and removing non-printable characters."""
    if not text:
        return ""
    # Normalize newline formats
    text = re.sub(r"\r\n|\r", "\n", text)
    # Remove null characters and non-printable control codes
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Consolidate 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def extract_text_from_pdf(pdf_source: Union[bytes, BinaryIO, str]) -> str:
    """
    Extract clean, structured text from a PDF file (bytes, file-like object, or file path).
    """
    try:
        if isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
            reader = PdfReader(stream)
        elif isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
        else:
            reader = PdfReader(pdf_source)

        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        full_text = "\n\n".join(extracted_pages)
        cleaned = clean_text(full_text)
        logger.info(f"Successfully extracted {len(cleaned)} characters from PDF ({len(reader.pages)} pages).")
        return cleaned
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to parse PDF file: {e}")


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[Dict[str, Union[int, str]]]:
    """
    Recursively split text into overlapping chunks for vector embedding storage.
    
    Returns a list of dicts: [{'chunk_index': int, 'chunk_text': str}]
    """
    if not text or not text.strip():
        return []

    cleaned = clean_text(text)
    if len(cleaned) <= chunk_size:
        return [{"chunk_index": 0, "chunk_text": cleaned}]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(cleaned):
        end = start + chunk_size

        if end >= len(cleaned):
            chunk = cleaned[start:].strip()
            if chunk:
                chunks.append({"chunk_index": chunk_index, "chunk_text": chunk})
            break

        # Prefer breaking at a newline or period within reasonable distance
        break_point = cleaned.rfind("\n", start + chunk_size // 2, end)
        if break_point == -1:
            break_point = cleaned.rfind(". ", start + chunk_size // 2, end)
            if break_point != -1:
                break_point += 1  # Include the period

        if break_point == -1:
            break_point = cleaned.rfind(" ", start + chunk_size // 2, end)

        if break_point == -1 or break_point <= start:
            break_point = end

        chunk = cleaned[start:break_point].strip()
        if chunk:
            chunks.append({"chunk_index": chunk_index, "chunk_text": chunk})
            chunk_index += 1

        # Advance start index with overlap
        start = break_point - overlap if (break_point - overlap > start) else break_point

    return chunks


def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Categorize resume text into key sections (summary, skills, experience, education, projects).
    """
    sections = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "other": "",
    }

    if not text:
        return sections

    section_keywords = {
        "summary": r"(?:summary|objective|about\s+me|profile)",
        "skills": r"(?:skills|technical\s+skills|core\s+competencies|technologies)",
        "experience": r"(?:work\s+experience|experience|employment\s+history|work\s+history)",
        "education": r"(?:education|academic\s+background|qualifications)",
        "projects": r"(?:projects|key\s+projects|personal\s+projects)",
    }

    all_headers_pattern = r"(?i)^(?:[#*=\-\s]*)(?P<header>" + "|".join(section_keywords.values()) + r")\b[:\-\s]*$"

    lines = text.split("\n")
    current_section = "summary"
    buffer = []

    for line in lines:
        match = re.match(all_headers_pattern, line.strip())
        if match:
            if buffer:
                sections[current_section] = (sections.get(current_section, "") + "\n" + "\n".join(buffer)).strip()
                buffer = []

            matched_header = match.group("header").lower()
            found_key = "other"
            for key, pattern in section_keywords.items():
                if re.search(r"\b" + pattern + r"\b", matched_header, re.IGNORECASE):
                    found_key = key
                    break
            current_section = found_key
        else:
            buffer.append(line)

    if buffer:
        sections[current_section] = (sections.get(current_section, "") + "\n" + "\n".join(buffer)).strip()

    if not any(sections[k] for k in ["skills", "experience", "education", "projects"]):
        sections["summary"] = text.strip()

    return sections
