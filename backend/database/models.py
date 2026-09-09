import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column
from sqlmodel import SQLModel, Field

from backend.database.types import EmbeddingVector


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserProfile(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(default="Candidate")
    email: Optional[str] = Field(default=None)
    target_role: str = Field(default="Software Engineer")
    skills_json: str = Field(default="[]")
    weaknesses_json: str = Field(default="[]")
    created_at: datetime = Field(default_factory=utc_now)


class InterviewSession(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="userprofile.id")
    session_type: str = Field(default="interview")
    job_role: str = Field(default="Software Engineer")
    job_description: Optional[str] = Field(default=None)
    status: str = Field(default="active")
    overall_score: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=utc_now)


class Message(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="interviewsession.id")
    role: str = Field(default="user")
    content: str
    agent_type: str = Field(default="supervisor")
    score: Optional[float] = Field(default=None)
    timestamp: datetime = Field(default_factory=utc_now)


class ResumeData(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="userprofile.id")
    raw_text: str
    parsed_json: str = Field(default="{}")
    resume_score: float = Field(default=0.0)
    uploaded_at: datetime = Field(default_factory=utc_now)


class ResumeEmbedding(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="userprofile.id")
    session_id: Optional[uuid.UUID] = Field(default=None, foreign_key="interviewsession.id")
    chunk_text: str
    chunk_index: int = Field(default=0)
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(EmbeddingVector()))


class KnowledgeDocument(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="userprofile.id")
    document_title: str
    total_chunks: int = Field(default=0)
    uploaded_at: datetime = Field(default_factory=utc_now)


class KnowledgeEmbedding(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="knowledgedocument.id")
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="userprofile.id")
    chunk_text: str
    chunk_index: int = Field(default=0)
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(EmbeddingVector()))
