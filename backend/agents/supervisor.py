"""
Supervisor Agent — Intent Detection + Routing
=============================================
The entry point for every user message. Reads the last HumanMessage,
classifies it into one of 5 intents, updates AgentState.current_agent,
and returns the state for LangGraph to route to the correct specialist.

Routing table:
    "resume"     → Resume Agent    (ATS scoring, skill extraction)
    "interview"  → Interview Agent (mock Q&A, technical interview)
    "code"       → Code Agent      (code review, debugging)
    "learning"   → Learning Agent  (study roadmap, topic recommendations)
    "knowledge"  → Knowledge Agent (RAG Q&A from uploaded docs/notes)

OpenAI is used when a real API key is present; otherwise a lightweight
keyword-based fallback handles routing without any API call.
"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from backend.agents.state import AgentState
from backend.config import settings
from backend.utils.logger import logger

# ------------------------------------------------------------------ #
# Types
# ------------------------------------------------------------------ #

Intent = Literal["resume", "interview", "code", "learning", "knowledge"]

VALID_INTENTS: set[str] = {"resume", "interview", "code", "learning", "knowledge"}

# ------------------------------------------------------------------ #
# Supervisor prompt
# ------------------------------------------------------------------ #

SUPERVISOR_PROMPT = """You are the Supervisor Agent for Interview Copilot AI.

Your only job is to classify the user's message into exactly one of these 5 intents and return a JSON object.

Intents:
- "resume"     → Resume upload, analysis, ATS scoring, skill extraction, resume feedback
- "interview"  → Mock technical interview, practice questions, role-based Q&A
- "code"       → Code review, debugging, complexity analysis, code explanation
- "learning"   → Study roadmap, topic recommendations, learning plan, resources
- "knowledge"  → Q&A or interview practice based on user-uploaded custom docs/PDFs/notes

Rules:
1. Return ONLY valid JSON — no markdown, no explanation.
2. Choose the single best-matching intent.
3. If ambiguous, prefer "interview".

Response format:
{
  "intent": "<one of: resume | interview | code | learning | knowledge>",
  "route_to": "<intent>_agent"
}

User message: {user_message}"""

# ------------------------------------------------------------------ #
# Keyword fallback (no API key required)
# ------------------------------------------------------------------ #

_KEYWORD_MAP: list[tuple[Intent, list[str]]] = [
    ("resume", [
        "resume", "cv", "ats", "skill extract", "upload resume", "parse resume",
        "resume score", "resume feedback", "resume analysis",
    ]),
    ("code", [
        "code", "debug", "review code", "function", "algorithm", "complexity",
        "big o", "refactor", "bug", "syntax", "class", "method", "snippet",
    ]),
    ("learning", [
        "learn", "roadmap", "study plan", "recommend", "resource", "course",
        "topic", "what should i", "how do i prepare", "syllabus",
    ]),
    ("knowledge", [
        "document", "pdf", "notes", "uploaded", "my doc", "based on",
        "from the doc", "quiz me", "knowledge base", "custom doc",
    ]),
    ("interview", [
        "interview", "question", "mock", "practice", "prepare", "tell me about",
        "explain", "what is", "how does", "describe",
    ]),
]


def _keyword_fallback(user_message: str) -> Intent:
    """
    Classify intent from keyword matching.
    Iterates _KEYWORD_MAP in priority order; first match wins.
    Falls back to 'interview' if nothing matches.
    """
    msg_lower = user_message.lower()
    for intent, keywords in _KEYWORD_MAP:
        if any(kw in msg_lower for kw in keywords):
            logger.debug(f"Keyword fallback matched intent='{intent}' for message: {msg_lower[:80]}")
            return intent
    logger.debug("Keyword fallback found no match — defaulting to 'interview'.")
    return "interview"


# ------------------------------------------------------------------ #
# Mock mode check
# ------------------------------------------------------------------ #

def _is_mock_mode() -> bool:
    """Return True when no real OpenAI key is configured."""
    key = settings.OPENAI_API_KEY or ""
    return key.startswith("sk-your") or key == "sk-placeholder" or len(key) < 20


# ------------------------------------------------------------------ #
# LLM-based intent detection
# ------------------------------------------------------------------ #

def _detect_intent_with_llm(user_message: str) -> Intent:
    """
    Call OpenAI to classify intent.
    Returns one of the 5 valid intent strings.
    Raises on error — caller handles fallback.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )
    prompt = SUPERVISOR_PROMPT.format(user_message=user_message)
    response = llm.invoke([SystemMessage(content=prompt)])
    raw = response.content.strip()

    # Strip optional markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    data = json.loads(raw)
    intent = data.get("intent", "").strip().lower()

    if intent not in VALID_INTENTS:
        logger.warning(f"LLM returned unknown intent '{intent}', falling back to keyword.")
        return _keyword_fallback(user_message)

    return intent  # type: ignore[return-value]


def detect_intent(user_message: str) -> Intent:
    """
    Public entry point for intent detection.

    Uses OpenAI when a real API key is configured; otherwise uses the
    keyword-based fallback so tests and local dev work without credentials.

    Args:
        user_message: The raw user input string.

    Returns:
        One of: "resume" | "interview" | "code" | "learning" | "knowledge"
    """
    if not user_message or not user_message.strip():
        logger.warning("Empty user message received by supervisor — defaulting to 'interview'.")
        return "interview"

    if _is_mock_mode():
        logger.debug("Supervisor mock mode: using keyword fallback for intent detection.")
        return _keyword_fallback(user_message)

    try:
        intent = _detect_intent_with_llm(user_message)
        logger.info(f"Supervisor LLM intent='{intent}' for: {user_message[:80]}")
        return intent
    except Exception as e:
        logger.warning(f"LLM intent detection failed ({e}), using keyword fallback.")
        return _keyword_fallback(user_message)


# ------------------------------------------------------------------ #
# LangGraph node: supervisor_node
# ------------------------------------------------------------------ #

def _get_last_user_message(state: AgentState) -> str:
    """Extract the content of the most recent HumanMessage from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def supervisor_node(state: AgentState) -> AgentState:
    """
    LangGraph node function for the Supervisor Agent.

    Reads the last HumanMessage, detects intent, and writes
    `current_agent` back into state for `route_to_agent` to route on.

    Args:
        state: Current AgentState flowing through the graph.

    Returns:
        Updated AgentState with `current_agent` set to the detected intent.
    """
    user_message = _get_last_user_message(state)
    intent = detect_intent(user_message)

    logger.info(
        f"Supervisor routing → '{intent}' | session={state.get('session_id', '')} "
        f"| user='{state['user'].get('name', 'unknown')}'"
    )

    state["current_agent"] = intent
    state["current_task"] = f"route_to_{intent}_agent"
    return state


# ------------------------------------------------------------------ #
# LangGraph conditional edge: route_to_agent
# ------------------------------------------------------------------ #

def route_to_agent(state: AgentState) -> str:
    """
    LangGraph conditional edge function.

    Called after `supervisor_node` completes; reads `current_agent` from
    state and returns the graph node name to transition to.

    Returns one of:
        "resume_agent" | "interview_agent" | "code_agent" |
        "learning_agent" | "knowledge_agent"
    """
    agent = state.get("current_agent", "interview")
    if agent not in VALID_INTENTS:
        logger.warning(f"route_to_agent received unknown agent='{agent}', defaulting to interview.")
        agent = "interview"
    return f"{agent}_agent"
