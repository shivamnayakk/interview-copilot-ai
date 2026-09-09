"""
backend.agents
==============
Public surface of the agents package.

Import from here to keep downstream code insulated from internal layout:

    from backend.agents import AgentState, UserContext, make_default_state
"""

from backend.agents.state import (
    AgentState,
    KnowledgeContext,
    ResumeContext,
    UserContext,
    make_default_state,
)

__all__ = [
    "AgentState",
    "KnowledgeContext",
    "ResumeContext",
    "UserContext",
    "make_default_state",
]
