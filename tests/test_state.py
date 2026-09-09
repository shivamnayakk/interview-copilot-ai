"""
tests/test_state.py
===================
Unit tests for Phase 5 — Shared Agent State (LangGraph Foundation).

Covers:
- AgentState construction via make_default_state
- Default field values
- Sub-context field access (UserContext, ResumeContext, KnowledgeContext)
- Message accumulation using LangGraph add_messages reducer
- Direct AgentState construction with custom values
- __init__.py public exports
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agents import (
    AgentState,
    KnowledgeContext,
    ResumeContext,
    UserContext,
    make_default_state,
)
from langgraph.graph.message import add_messages


# --------------------------------------------------------------------------- #
# make_default_state — defaults
# --------------------------------------------------------------------------- #

class TestMakeDefaultState:
    def test_returns_agent_state_type(self):
        state = make_default_state()
        assert isinstance(state, dict)

    def test_default_session_id_empty(self):
        state = make_default_state()
        assert state["session_id"] == ""

    def test_default_current_agent_is_supervisor(self):
        state = make_default_state()
        assert state["current_agent"] == "supervisor"

    def test_default_current_task_empty(self):
        state = make_default_state()
        assert state["current_task"] == ""

    def test_default_final_response_empty(self):
        state = make_default_state()
        assert state["final_response"] == ""

    def test_default_messages_empty_list(self):
        state = make_default_state()
        assert state["messages"] == []

    def test_custom_session_id(self):
        state = make_default_state(session_id="abc-123")
        assert state["session_id"] == "abc-123"

    def test_custom_user_fields(self):
        state = make_default_state(user_id="u-1", name="Alice", target_role="ML Engineer")
        assert state["user"]["user_id"] == "u-1"
        assert state["user"]["name"] == "Alice"
        assert state["user"]["target_role"] == "ML Engineer"


# --------------------------------------------------------------------------- #
# UserContext defaults
# --------------------------------------------------------------------------- #

class TestUserContext:
    def test_default_name(self):
        state = make_default_state()
        assert state["user"]["name"] == "Candidate"

    def test_default_target_role(self):
        state = make_default_state()
        assert state["user"]["target_role"] == "Software Engineer"

    def test_default_skills_empty(self):
        state = make_default_state()
        assert state["user"]["skills"] == []

    def test_default_weaknesses_empty(self):
        state = make_default_state()
        assert state["user"]["weaknesses"] == []

    def test_user_context_mutation(self):
        state = make_default_state()
        state["user"]["skills"] = ["Python", "FastAPI"]
        state["user"]["weaknesses"] = ["System Design"]
        assert state["user"]["skills"] == ["Python", "FastAPI"]
        assert state["user"]["weaknesses"] == ["System Design"]


# --------------------------------------------------------------------------- #
# ResumeContext defaults
# --------------------------------------------------------------------------- #

class TestResumeContext:
    def test_default_raw_text_empty(self):
        state = make_default_state()
        assert state["resume"]["raw_text"] == ""

    def test_default_parsed_data_empty_dict(self):
        state = make_default_state()
        assert state["resume"]["parsed_data"] == {}

    def test_default_resume_score_zero(self):
        state = make_default_state()
        assert state["resume"]["resume_score"] == 0.0

    def test_resume_context_mutation(self):
        state = make_default_state()
        state["resume"]["raw_text"] = "John Doe | Python Developer"
        state["resume"]["resume_score"] = 8.5
        state["resume"]["parsed_data"] = {"skills": ["Python"]}
        assert state["resume"]["raw_text"] == "John Doe | Python Developer"
        assert state["resume"]["resume_score"] == 8.5
        assert state["resume"]["parsed_data"]["skills"] == ["Python"]


# --------------------------------------------------------------------------- #
# KnowledgeContext defaults
# --------------------------------------------------------------------------- #

class TestKnowledgeContext:
    def test_default_active_document_id_none(self):
        state = make_default_state()
        assert state["knowledge"]["active_document_id"] is None

    def test_default_active_document_title_none(self):
        state = make_default_state()
        assert state["knowledge"]["active_document_title"] is None

    def test_default_retrieved_chunks_empty(self):
        state = make_default_state()
        assert state["knowledge"]["retrieved_chunks"] == []

    def test_default_questions_generated_empty(self):
        state = make_default_state()
        assert state["knowledge"]["questions_generated"] == []

    def test_default_doc_scores_empty(self):
        state = make_default_state()
        assert state["knowledge"]["doc_scores"] == []

    def test_knowledge_context_mutation(self):
        state = make_default_state()
        state["knowledge"]["active_document_id"] = "doc-uuid-1"
        state["knowledge"]["active_document_title"] = "System Design Notes.pdf"
        state["knowledge"]["retrieved_chunks"] = ["Load balancers distribute traffic."]
        state["knowledge"]["doc_scores"] = [0.92]
        assert state["knowledge"]["active_document_id"] == "doc-uuid-1"
        assert state["knowledge"]["retrieved_chunks"][0] == "Load balancers distribute traffic."
        assert state["knowledge"]["doc_scores"][0] == 0.92


# --------------------------------------------------------------------------- #
# Message accumulation via add_messages reducer
# --------------------------------------------------------------------------- #

class TestMessageAccumulation:
    def test_add_messages_appends_to_empty(self):
        existing = []
        new_msgs = [HumanMessage(content="Hello")]
        result = add_messages(existing, new_msgs)
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_add_messages_appends_multiple(self):
        existing = [HumanMessage(content="Hello")]
        new_msgs = [AIMessage(content="Hi there!"), HumanMessage(content="Tell me about FastAPI")]
        result = add_messages(existing, new_msgs)
        assert len(result) == 3
        assert result[1].content == "Hi there!"
        assert result[2].content == "Tell me about FastAPI"

    def test_add_messages_preserves_message_types(self):
        msgs = [
            SystemMessage(content="You are an interview coach."),
            HumanMessage(content="Start the interview"),
            AIMessage(content="Sure! Let's begin."),
        ]
        result = add_messages([], msgs)
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_state_messages_accumulate_across_updates(self):
        state = make_default_state()
        # Simulate first agent turn
        state["messages"] = add_messages(state["messages"], [HumanMessage(content="Analyse my resume")])
        assert len(state["messages"]) == 1
        # Simulate second agent turn
        state["messages"] = add_messages(state["messages"], [AIMessage(content="Your resume score is 7.5/10")])
        assert len(state["messages"]) == 2
        assert state["messages"][0].content == "Analyse my resume"
        assert state["messages"][1].content == "Your resume score is 7.5/10"

    def test_message_ids_are_unique_by_default(self):
        msgs = [HumanMessage(content="Q1"), HumanMessage(content="Q2")]
        result = add_messages([], msgs)
        ids = [m.id for m in result if m.id]
        # If IDs are assigned, they should be unique
        assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Direct AgentState construction (without make_default_state)
# --------------------------------------------------------------------------- #

class TestDirectConstruction:
    def test_full_agent_state_construction(self):
        state = AgentState(
            user=UserContext(
                user_id="u-99",
                name="Bob",
                skills=["Go", "Kubernetes"],
                weaknesses=["Frontend"],
                target_role="DevOps Engineer",
            ),
            resume=ResumeContext(
                raw_text="Bob Smith | DevOps",
                parsed_data={"skills": ["Go", "Kubernetes"]},
                resume_score=9.0,
            ),
            knowledge=KnowledgeContext(
                active_document_id="doc-42",
                active_document_title="K8s Notes.pdf",
                retrieved_chunks=["Kubernetes orchestrates containers."],
                questions_generated=["What is a Pod?"],
                doc_scores=[0.88],
            ),
            messages=[HumanMessage(content="Let's do a DevOps interview")],
            current_agent="interview",
            current_task="generate_question",
            session_id="sess-001",
            final_response="",
        )

        assert state["user"]["name"] == "Bob"
        assert state["resume"]["resume_score"] == 9.0
        assert state["knowledge"]["active_document_title"] == "K8s Notes.pdf"
        assert state["messages"][0].content == "Let's do a DevOps interview"
        assert state["current_agent"] == "interview"
        assert state["session_id"] == "sess-001"

    def test_current_agent_valid_values(self):
        valid_agents = ["supervisor", "resume", "interview", "code", "learning", "knowledge"]
        for agent in valid_agents:
            state = make_default_state()
            state["current_agent"] = agent
            assert state["current_agent"] == agent


# --------------------------------------------------------------------------- #
# __init__.py public exports
# --------------------------------------------------------------------------- #

class TestPublicExports:
    def test_agent_state_exported(self):
        from backend.agents import AgentState
        assert AgentState is not None

    def test_user_context_exported(self):
        from backend.agents import UserContext
        assert UserContext is not None

    def test_resume_context_exported(self):
        from backend.agents import ResumeContext
        assert ResumeContext is not None

    def test_knowledge_context_exported(self):
        from backend.agents import KnowledgeContext
        assert KnowledgeContext is not None

    def test_make_default_state_exported(self):
        from backend.agents import make_default_state
        assert callable(make_default_state)
