"""
LangGraph Workflow — 5-Agent StateGraph
========================================
Assembles the full Interview Copilot AI multi-agent graph:

    supervisor_node
         │
         └─ route_to_agent (conditional edge)
              ├── resume_agent
              ├── interview_agent
              ├── code_agent
              ├── learning_agent
              └── knowledge_agent
                       │
                      END

Each specialist node is a stub that sets `final_response` and
`current_agent`. Full implementations are added in Phase 7.

Usage:
    from backend.agents.workflow import build_workflow

    graph = build_workflow()
    result = graph.invoke(state)
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from backend.agents.state import AgentState
from backend.agents.supervisor import route_to_agent, supervisor_node
from backend.utils.logger import logger


# ------------------------------------------------------------------ #
# Stub specialist nodes (Phase 7 will implement full logic)
# ------------------------------------------------------------------ #

def resume_node(state: AgentState) -> AgentState:
    """
    Resume Agent stub.
    Phase 7: parse PDF, extract skills, run ATS scoring, populate ResumeContext.
    """
    logger.info(f"[ResumeAgent] session={state.get('session_id', '')} — stub response.")
    state["current_agent"] = "resume"
    state["final_response"] = (
        "📄 Resume Agent is analysing your resume. "
        "I'll extract your skills, score it against ATS criteria, "
        "and identify areas for improvement. (stub — full logic in Phase 7)"
    )
    state["messages"] = state["messages"] + [
        AIMessage(content=state["final_response"], name="resume_agent")
    ]
    return state


def interview_node(state: AgentState) -> AgentState:
    """
    Interview Agent stub.
    Phase 7: generate role-specific technical questions, evaluate answers.
    """
    logger.info(f"[InterviewAgent] session={state.get('session_id', '')} — stub response.")
    state["current_agent"] = "interview"
    state["final_response"] = (
        "🎙️ Interview Agent is ready. "
        "I'll generate tailored technical questions based on your target role "
        f"({state['user'].get('target_role', 'Software Engineer')}) and resume. "
        "(stub — full logic in Phase 7)"
    )
    state["messages"] = state["messages"] + [
        AIMessage(content=state["final_response"], name="interview_agent")
    ]
    return state


def code_node(state: AgentState) -> AgentState:
    """
    Code Review Agent stub.
    Phase 7: review code snippets, identify bugs, analyse complexity.
    """
    logger.info(f"[CodeAgent] session={state.get('session_id', '')} — stub response.")
    state["current_agent"] = "code"
    state["final_response"] = (
        "💻 Code Review Agent is ready. "
        "Paste your code and I'll review it for correctness, complexity, "
        "and best practices. (stub — full logic in Phase 7)"
    )
    state["messages"] = state["messages"] + [
        AIMessage(content=state["final_response"], name="code_agent")
    ]
    return state


def learning_node(state: AgentState) -> AgentState:
    """
    Learning Agent stub.
    Phase 7: build personalised study roadmap based on skills and target role.
    """
    logger.info(f"[LearningAgent] session={state.get('session_id', '')} — stub response.")
    state["current_agent"] = "learning"
    state["final_response"] = (
        "📚 Learning Agent is ready. "
        "I'll build a personalised study roadmap and recommend resources "
        "to close the gap between your current skills and your target role. "
        "(stub — full logic in Phase 7)"
    )
    state["messages"] = state["messages"] + [
        AIMessage(content=state["final_response"], name="learning_agent")
    ]
    return state


def knowledge_node(state: AgentState) -> AgentState:
    """
    RAG Knowledge Agent stub.
    Phase 7: retrieve top-k chunks from pgvector, generate doc-grounded questions,
    evaluate answers with source citations.
    """
    logger.info(f"[KnowledgeAgent] session={state.get('session_id', '')} — stub response.")
    doc_title = state["knowledge"].get("active_document_title") or "your uploaded document"
    state["current_agent"] = "knowledge"
    state["final_response"] = (
        f"🧠 Knowledge Agent is ready. "
        f"I'll quiz you on '{doc_title}' by retrieving the most relevant sections "
        "and asking deep, doc-grounded questions. (stub — full logic in Phase 7)"
    )
    state["messages"] = state["messages"] + [
        AIMessage(content=state["final_response"], name="knowledge_agent")
    ]
    return state


# ------------------------------------------------------------------ #
# Graph builder
# ------------------------------------------------------------------ #

def build_workflow() -> StateGraph:
    """
    Assemble and compile the full LangGraph StateGraph.

    Graph topology:
        START → supervisor → (conditional) → specialist → END

    Returns:
        A compiled LangGraph graph ready for .invoke() / .stream().
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("resume_agent", resume_node)
    graph.add_node("interview_agent", interview_node)
    graph.add_node("code_agent", code_node)
    graph.add_node("learning_agent", learning_node)
    graph.add_node("knowledge_agent", knowledge_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor → one of 5 specialists
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "resume_agent": "resume_agent",
            "interview_agent": "interview_agent",
            "code_agent": "code_agent",
            "learning_agent": "learning_agent",
            "knowledge_agent": "knowledge_agent",
        },
    )

    # All specialist nodes terminate at END
    for agent_node in [
        "resume_agent",
        "interview_agent",
        "code_agent",
        "learning_agent",
        "knowledge_agent",
    ]:
        graph.add_edge(agent_node, END)

    return graph.compile()


# Module-level compiled graph — import this for one-time compilation
workflow = build_workflow()
