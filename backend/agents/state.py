"""
Shared Agent State for Interview Copilot AI.

All LangGraph agents read from and write to this single AgentState,
ensuring every agent has access to the full conversation context,
user profile, resume data, and knowledge base context.

Architecture:
    USER → Supervisor → [Resume | Interview | Code | Learning | Knowledge] Agent
                                         ↑
                                   AgentState (shared)
"""

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserContext(TypedDict):
    """
    Persistent user profile data populated after login / resume upload.
    Shared across all agents so every agent knows who it's talking to.
    """
    user_id: str              # UUID of the UserProfile row
    name: str                 # Candidate's display name
    skills: List[str]         # e.g. ["Python", "FastAPI", "LangGraph"]
    weaknesses: List[str]     # e.g. ["System Design", "Dynamic Programming"]
    target_role: str          # e.g. "Senior AI Engineer"


class ResumeContext(TypedDict):
    """
    Populated by the Resume Agent after a resume PDF is uploaded & parsed.
    Downstream agents (Interview, Code) use this to personalise questions.
    """
    raw_text: str             # Full extracted resume text
    parsed_data: dict         # Structured extraction: skills, experience, education, etc.
    resume_score: float       # ATS/quality score 0.0–10.0


class KnowledgeContext(TypedDict):
    """
    Populated by the RAG Knowledge Agent when the user uploads custom
    study documents (PDFs, notes). Tracks the active document and the
    chunks retrieved for the current query.
    """
    active_document_id: Optional[str]     # UUID of the active KnowledgeDocument
    active_document_title: Optional[str]  # Human-readable doc name shown in UI
    retrieved_chunks: List[str]           # Top-k chunk texts from last retrieval
    questions_generated: List[str]        # Document-grounded questions generated so far
    doc_scores: List[float]               # Similarity scores for each retrieved chunk


class AgentState(TypedDict):
    """
    The single source of truth flowing through every LangGraph node.

    `messages` uses LangGraph's `add_messages` reducer so that each node
    appends rather than overwrites — giving a full conversation history.

    All other fields are plain last-write-wins so agents can update them
    freely as new context is discovered.
    """
    # Sub-contexts populated progressively as the session advances
    user: UserContext
    resume: ResumeContext
    knowledge: KnowledgeContext

    # Full conversation history — add_messages handles deduplication & merging
    messages: Annotated[List[BaseMessage], add_messages]

    # Routing / control fields written by the Supervisor and read by nodes
    current_agent: str    # "supervisor" | "resume" | "interview" | "code" | "learning" | "knowledge"
    current_task: str     # Short description of the active task (for logging / UI)
    session_id: str       # UUID of the active InterviewSession row

    # Final assembled response string — populated by each terminal agent node
    final_response: str


# --------------------------------------------------------------------------- #
# Helper: build a minimal valid AgentState with safe defaults
# --------------------------------------------------------------------------- #

def make_default_state(
    session_id: str = "",
    user_id: str = "",
    name: str = "Candidate",
    target_role: str = "Software Engineer",
) -> AgentState:
    """
    Return an AgentState with every field initialised to a safe default.
    Useful for bootstrapping a new session before any agent has run.

    Args:
        session_id:   InterviewSession UUID (can be set later).
        user_id:      UserProfile UUID (can be set later).
        name:         Candidate name for personalisation.
        target_role:  Job role the candidate is preparing for.

    Returns:
        A fully-initialised AgentState TypedDict.
    """
    return AgentState(
        user=UserContext(
            user_id=user_id,
            name=name,
            skills=[],
            weaknesses=[],
            target_role=target_role,
        ),
        resume=ResumeContext(
            raw_text="",
            parsed_data={},
            resume_score=0.0,
        ),
        knowledge=KnowledgeContext(
            active_document_id=None,
            active_document_title=None,
            retrieved_chunks=[],
            questions_generated=[],
            doc_scores=[],
        ),
        messages=[],
        current_agent="supervisor",
        current_task="",
        session_id=session_id,
        final_response="",
    )
