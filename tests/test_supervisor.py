"""
tests/test_supervisor.py
========================
Unit + integration tests for Phase 6 — Supervisor Agent + LangGraph Workflow.

Covers:
- keyword fallback routing for all 5 intents
- empty / whitespace message handling
- detect_intent always returns a valid intent
- supervisor_node updates AgentState correctly
- route_to_agent returns correct node names
- unknown current_agent fallback in route_to_agent
- full workflow graph invocation end-to-end for all 5 routes
- workflow produces final_response and correct current_agent
- workflow appends AIMessage to state messages
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from backend.agents.state import make_default_state
from backend.agents.supervisor import (
    VALID_INTENTS,
    _keyword_fallback,
    detect_intent,
    route_to_agent,
    supervisor_node,
)
from backend.agents.workflow import (
    build_workflow,
    code_node,
    interview_node,
    knowledge_node,
    learning_node,
    resume_node,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _state_with_message(message: str, **kwargs) -> dict:
    """Build an AgentState with a single HumanMessage."""
    state = make_default_state(**kwargs)
    state["messages"] = [HumanMessage(content=message)]
    return state


# --------------------------------------------------------------------------- #
# Keyword fallback — all 5 intents
# --------------------------------------------------------------------------- #

class TestKeywordFallback:
    def test_resume_keywords(self):
        for msg in ["analyse my resume", "upload my cv", "ats score", "resume feedback"]:
            assert _keyword_fallback(msg) == "resume", f"Failed for: {msg}"

    def test_code_keywords(self):
        for msg in ["review my code", "debug this function", "what is the complexity"]:
            assert _keyword_fallback(msg) == "code", f"Failed for: {msg}"

    def test_learning_keywords(self):
        for msg in ["give me a study roadmap", "recommend resources", "learning plan for python"]:
            assert _keyword_fallback(msg) == "learning", f"Failed for: {msg}"

    def test_knowledge_keywords(self):
        for msg in ["quiz me from my pdf", "based on my uploaded notes", "from the document"]:
            assert _keyword_fallback(msg) == "knowledge", f"Failed for: {msg}"

    def test_interview_keywords(self):
        for msg in ["start a mock interview", "practice questions", "what is a load balancer"]:
            assert _keyword_fallback(msg) == "interview", f"Failed for: {msg}"

    def test_default_fallback_is_interview(self):
        assert _keyword_fallback("xyzzy unknown gibberish 12345") == "interview"

    def test_case_insensitive(self):
        assert _keyword_fallback("RESUME ANALYSIS") == "resume"
        assert _keyword_fallback("CODE REVIEW") == "code"


# --------------------------------------------------------------------------- #
# detect_intent — mock mode (no real API key)
# --------------------------------------------------------------------------- #

class TestDetectIntentMockMode:
    """In mock mode detect_intent always uses keyword fallback (no API call)."""

    def test_resume_intent(self):
        assert detect_intent("analyse my resume please") == "resume"

    def test_interview_intent(self):
        assert detect_intent("start a mock technical interview") == "interview"

    def test_code_intent(self):
        assert detect_intent("please review my code snippet") == "code"

    def test_learning_intent(self):
        assert detect_intent("give me a study roadmap for system design") == "learning"

    def test_knowledge_intent(self):
        assert detect_intent("quiz me based on my uploaded pdf") == "knowledge"

    def test_empty_message_returns_interview(self):
        assert detect_intent("") == "interview"

    def test_whitespace_only_returns_interview(self):
        assert detect_intent("   \n\t  ") == "interview"

    def test_always_returns_valid_intent(self):
        messages = [
            "tell me something", "what should I do", "help me", "random text",
            "review", "notes", "document", "code", "resume",
        ]
        for msg in messages:
            result = detect_intent(msg)
            assert result in VALID_INTENTS, f"Invalid intent '{result}' for: {msg}"


# --------------------------------------------------------------------------- #
# detect_intent — LLM path (mocked)
# --------------------------------------------------------------------------- #

class TestDetectIntentLLM:
    """Test the LLM path by patching _is_mock_mode and ChatOpenAI."""

    def _mock_llm_response(self, intent: str):
        import json
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({"intent": intent, "route_to": f"{intent}_agent"})
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg
        return mock_llm

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_resume_intent(self, MockLLM, _):
        MockLLM.return_value = self._mock_llm_response("resume")
        result = detect_intent("score my resume")
        assert result == "resume"

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_interview_intent(self, MockLLM, _):
        MockLLM.return_value = self._mock_llm_response("interview")
        result = detect_intent("ask me a Python question")
        assert result == "interview"

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_code_intent(self, MockLLM, _):
        MockLLM.return_value = self._mock_llm_response("code")
        result = detect_intent("review this function")
        assert result == "code"

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_learning_intent(self, MockLLM, _):
        MockLLM.return_value = self._mock_llm_response("learning")
        result = detect_intent("build me a learning roadmap")
        assert result == "learning"

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_knowledge_intent(self, MockLLM, _):
        MockLLM.return_value = self._mock_llm_response("knowledge")
        result = detect_intent("quiz me from my notes")
        assert result == "knowledge"

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_unknown_intent_falls_back_to_keyword(self, MockLLM, _):
        """LLM returns invalid intent → keyword fallback takes over."""
        bad_msg = MagicMock()
        bad_msg.content = '{"intent": "invalid_thing", "route_to": "invalid_agent"}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_msg
        MockLLM.return_value = mock_llm
        result = detect_intent("start a mock interview session")
        assert result in VALID_INTENTS

    @patch("backend.agents.supervisor._is_mock_mode", return_value=False)
    @patch("backend.agents.supervisor.ChatOpenAI")
    def test_llm_exception_falls_back_to_keyword(self, MockLLM, _):
        """LLM raises an exception → keyword fallback takes over."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        MockLLM.return_value = mock_llm
        result = detect_intent("review my code please")
        assert result in VALID_INTENTS


# --------------------------------------------------------------------------- #
# supervisor_node
# --------------------------------------------------------------------------- #

class TestSupervisorNode:
    def test_sets_current_agent(self):
        state = _state_with_message("analyse my resume")
        updated = supervisor_node(state)
        assert updated["current_agent"] == "resume"

    def test_sets_current_task(self):
        state = _state_with_message("start a mock interview")
        updated = supervisor_node(state)
        assert updated["current_task"] == "route_to_interview_agent"

    def test_all_five_intents_routed(self):
        cases = [
            ("review this code function", "code"),
            ("build a study roadmap", "learning"),
            ("quiz from my uploaded pdf", "knowledge"),
            ("score my cv", "resume"),
            ("practice interview questions", "interview"),
        ]
        for msg, expected in cases:
            state = _state_with_message(msg)
            updated = supervisor_node(state)
            assert updated["current_agent"] == expected, (
                f"msg='{msg}' expected='{expected}' got='{updated['current_agent']}'"
            )

    def test_no_human_message_defaults_to_interview(self):
        state = make_default_state()
        state["messages"] = []
        updated = supervisor_node(state)
        assert updated["current_agent"] == "interview"

    def test_does_not_alter_other_state_fields(self):
        state = _state_with_message("resume feedback", session_id="s-99", name="Alice")
        updated = supervisor_node(state)
        assert updated["session_id"] == "s-99"
        assert updated["user"]["name"] == "Alice"


# --------------------------------------------------------------------------- #
# route_to_agent
# --------------------------------------------------------------------------- #

class TestRouteToAgent:
    @pytest.mark.parametrize("intent,expected_node", [
        ("resume", "resume_agent"),
        ("interview", "interview_agent"),
        ("code", "code_agent"),
        ("learning", "learning_agent"),
        ("knowledge", "knowledge_agent"),
    ])
    def test_all_valid_intents(self, intent, expected_node):
        state = make_default_state()
        state["current_agent"] = intent
        assert route_to_agent(state) == expected_node

    def test_unknown_agent_defaults_to_interview(self):
        state = make_default_state()
        state["current_agent"] = "nonexistent"
        assert route_to_agent(state) == "interview_agent"

    def test_empty_current_agent_defaults_to_interview(self):
        state = make_default_state()
        state["current_agent"] = ""
        assert route_to_agent(state) == "interview_agent"


# --------------------------------------------------------------------------- #
# Stub specialist nodes
# --------------------------------------------------------------------------- #

class TestStubNodes:
    def test_resume_node_sets_final_response(self):
        state = make_default_state(session_id="s-1")
        updated = resume_node(state)
        assert updated["current_agent"] == "resume"
        assert len(updated["final_response"]) > 0
        assert isinstance(updated["messages"][-1], AIMessage)

    def test_interview_node_includes_target_role(self):
        state = make_default_state(target_role="ML Engineer")
        updated = interview_node(state)
        assert updated["current_agent"] == "interview"
        assert "ML Engineer" in updated["final_response"]

    def test_code_node_sets_final_response(self):
        state = make_default_state()
        updated = code_node(state)
        assert updated["current_agent"] == "code"
        assert len(updated["final_response"]) > 0

    def test_learning_node_sets_final_response(self):
        state = make_default_state()
        updated = learning_node(state)
        assert updated["current_agent"] == "learning"
        assert len(updated["final_response"]) > 0

    def test_knowledge_node_uses_doc_title(self):
        state = make_default_state()
        state["knowledge"]["active_document_title"] = "System Design Notes.pdf"
        updated = knowledge_node(state)
        assert updated["current_agent"] == "knowledge"
        assert "System Design Notes.pdf" in updated["final_response"]

    def test_knowledge_node_default_doc_title(self):
        state = make_default_state()
        updated = knowledge_node(state)
        assert "your uploaded document" in updated["final_response"]

    def test_all_nodes_append_ai_message(self):
        nodes = [resume_node, interview_node, code_node, learning_node, knowledge_node]
        for node_fn in nodes:
            state = make_default_state()
            state["messages"] = [HumanMessage(content="hello")]
            updated = node_fn(state)
            assert len(updated["messages"]) == 2
            assert isinstance(updated["messages"][-1], AIMessage)


# --------------------------------------------------------------------------- #
# Full workflow integration — end-to-end graph invocation
# --------------------------------------------------------------------------- #

class TestWorkflowIntegration:
    """Invoke the compiled workflow graph and assert correct routing end-to-end."""

    @pytest.fixture(scope="class")
    def graph(self):
        return build_workflow()

    def _invoke(self, graph, message: str, **state_kwargs) -> dict:
        state = make_default_state(**state_kwargs)
        state["messages"] = [HumanMessage(content=message)]
        return graph.invoke(state)

    def test_resume_route_end_to_end(self, graph):
        result = self._invoke(graph, "please analyse my resume")
        assert result["current_agent"] == "resume"
        assert len(result["final_response"]) > 0

    def test_interview_route_end_to_end(self, graph):
        result = self._invoke(graph, "start a mock technical interview")
        assert result["current_agent"] == "interview"
        assert len(result["final_response"]) > 0

    def test_code_route_end_to_end(self, graph):
        result = self._invoke(graph, "review my code for bugs")
        assert result["current_agent"] == "code"
        assert len(result["final_response"]) > 0

    def test_learning_route_end_to_end(self, graph):
        result = self._invoke(graph, "give me a study roadmap")
        assert result["current_agent"] == "learning"
        assert len(result["final_response"]) > 0

    def test_knowledge_route_end_to_end(self, graph):
        result = self._invoke(graph, "quiz me from my uploaded pdf notes")
        assert result["current_agent"] == "knowledge"
        assert len(result["final_response"]) > 0

    def test_workflow_appends_ai_message(self, graph):
        result = self._invoke(graph, "start a mock interview")
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

    def test_workflow_preserves_session_id(self, graph):
        result = self._invoke(graph, "review my code", session_id="sess-42")
        assert result["session_id"] == "sess-42"

    def test_workflow_preserves_user_context(self, graph):
        result = self._invoke(graph, "practice questions", name="Carol", target_role="DevOps Engineer")
        assert result["user"]["name"] == "Carol"
        assert result["user"]["target_role"] == "DevOps Engineer"
