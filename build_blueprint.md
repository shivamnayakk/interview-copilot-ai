# 🚀 Interview Copilot AI — Complete Build Blueprint (Updated with RAG Knowledge Agent)

> **Role**: Senior AI Engineer + System Architect  
> **Goal**: Hackathon-winning MVP → 30-40 LPA Portfolio Project  
> **Stack**: FastAPI + LangGraph + LangChain + pgvector + Next.js

---

## 🏛️ Updated Multi-Agent Architecture (5 Core Agents)

```
                    USER (Browser / Next.js)
                             │
                     Conversation Layer
                             │
                     SUPERVISOR AGENT
               (Intent Detection + Routing)
                             │
 ┌──────────────┬────────────┼────────────┬──────────────┐
 │              │            │            │              │
Resume      Interview    Learning     Code Review     RAG Knowledge
Agent        Agent        Agent         Agent            Agent
 │              │            │            │              │
 └──────────────┴────────────┼────────────┴──────────────┘
                             │
                    SHARED AGENT STATE
                             │
             Memory + Persistence Layer
            (PostgreSQL + pgvector Database)
```

---

# 📋 SECTION 1: 10-Phase Development Roadmap

---

## PHASE 1 — Foundation Layer
**Goal**: Project scaffold + local dev environment ready

### Features:
- FastAPI server with CORS, health check
- PostgreSQL + pgvector via Docker
- Environment configuration (`.env`)
- Loguru structured logging

---

## PHASE 2 — Data Layer (Models + Persistence)
**Goal**: Define database schema for all 5 agents & RAG storage

### Models to Create (`backend/database/models.py`):
```python
class UserProfile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    skills: List[str]
    weaknesses: List[str]
    target_role: str

class InterviewSession(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID
    session_type: str            # "resume" | "interview" | "code" | "learning" | "knowledge"
    status: str                  # "active" | "completed"
    overall_score: float = 0.0

class Message(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID
    role: str                    # "user" | "assistant" | "system"
    content: str
    agent_type: str              # "supervisor" | "resume" | "interview" | "code" | "learning" | "knowledge"
    score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ResumeEmbedding(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID
    chunk_text: str
    embedding: List[float]       # pgvector column (1536 dims)

class KnowledgeDocument(SQLModel, table=True):
    """Store metadata for user-uploaded custom study notes/docs/PDFs"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID
    document_title: str
    total_chunks: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeEmbedding(SQLModel, table=True):
    """pgvector embeddings for custom uploaded knowledge base docs"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="knowledgedocument.id")
    user_id: UUID
    chunk_text: str
    chunk_index: int
    embedding: List[float]       # pgvector column (1536 dims)
```

---

## PHASE 3 — PDF Parser + Resume/Doc Extractor
**Goal**: Extract clean text from Resume PDFs & Custom Knowledge PDFs/Notes

### Features:
- Generic `pdf_parser.py` supporting both Resumes & Custom Study Documents
- Chunking helper (`chunk_text`) for document indexing

---

## PHASE 4 — RAG Pipeline Engine (Embeddings + Vector Search)
**Goal**: Common vector retrieval engine for both Resume context & Knowledge Base docs

### Features:
- `embeddings.py` (OpenAI `text-embedding-3-small` wrapper)
- `pipeline.py` (Index resume AND custom notes/PDFs into pgvector)
- `retriever.py` (Cosine similarity search across `KnowledgeEmbedding` & `ResumeEmbedding`)

```python
# retriever.py
def search_knowledge_base(user_id: str, query: str, top_k: int = 3, document_id: Optional[str] = None):
    query_vector = get_embedding(query)
    # Cosine similarity query on KnowledgeEmbedding table in pgvector
    return top_chunks
```

---

## PHASE 5 — Shared Agent State (LangGraph Foundation)
**Goal**: Updated `AgentState` schema to include RAG Knowledge Agent context

```python
# backend/agents/state.py
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class UserContext(TypedDict):
    user_id: str
    name: str
    skills: List[str]
    weaknesses: List[str]
    target_role: str

class ResumeContext(TypedDict):
    raw_text: str
    parsed_data: dict
    resume_score: float

class KnowledgeContext(TypedDict):
    active_document_id: Optional[str]
    active_document_title: Optional[str]
    retrieved_chunks: List[str]
    questions_generated: List[str]
    doc_scores: List[float]

class AgentState(TypedDict):
    user: UserContext
    resume: ResumeContext
    knowledge: KnowledgeContext
    messages: Annotated[List, add_messages]
    current_agent: str          # "supervisor" | "resume" | "interview" | "code" | "learning" | "knowledge"
    current_task: str
    session_id: str
    final_response: str
```

---

## PHASE 6 — Supervisor Agent (Intent Detection + Routing)
**Goal**: Route user requests to 5 specialized agents including RAG Knowledge Agent

### Routing Logic:
```python
SUPERVISOR_PROMPT = """
You are the Supervisor Agent for Interview Copilot AI.

Route user intent to one of the 5 specialized agents:
- "resume" → Resume analysis, skill extraction, ATS scoring
- "interview" → Mock technical interview, Q&A on role/skills
- "code" → Code review, debugging, complexity analysis
- "learning" → Study roadmap, learning recommendations
- "knowledge" → Q&A/Interview based on user-uploaded custom docs/notes/PDFs

User message: {user_message}

Return JSON:
{
  "intent": "resume | interview | code | learning | knowledge",
  "route_to": "knowledge_agent"
}
"""
```

### LangGraph StateGraph with 5 Agents:
```python
def build_workflow():
    graph = StateGraph(AgentState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("resume_agent", resume_node)
    graph.add_node("interview_agent", interview_node)
    graph.add_node("code_agent", code_node)
    graph.add_node("learning_agent", learning_node)
    graph.add_node("knowledge_agent", knowledge_node)  # NEW: RAG Agent node
    
    graph.set_entry_point("supervisor")
    
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "resume": "resume_agent",
            "interview": "interview_agent",
            "code": "code_agent",
            "learning": "learning_agent",
            "knowledge": "knowledge_agent",
        }
    )
    
    for agent in ["resume_agent", "interview_agent", "code_agent", "learning_agent", "knowledge_agent"]:
        graph.add_edge(agent, END)
        
    return graph.compile()
```

---

## PHASE 7 — Specialized Agents (Including RAG Knowledge Agent)
**Goal**: Implementation of the new RAG Knowledge Agent alongside existing 4 agents

### 🧠 RAG Knowledge Agent Logic (`backend/agents/knowledge_agent.py`):
1. **Document Upload / Selection**: User uploads or selects a study doc (e.g., "System Design Notes.pdf").
2. **Context Retrieval**: Retrieve top relevant chunks matching current interview topic or user query.
3. **Document Question Generation**: Generate deep technical questions directly grounded in the retrieved chunks.
4. **Grounded Answer Evaluation**: Evaluate user's answer strictly against the retrieved document context.

```python
# knowledge_agent.py
def knowledge_node(state: AgentState) -> AgentState:
    user_msg = get_last_user_message(state['messages'])
    active_doc_id = state['knowledge'].get('active_document_id')
    
    # Retrieve top-3 chunks from pgvector for user's document
    context_chunks = search_knowledge_base(
        user_id=state['user']['user_id'],
        query=user_msg,
        top_k=3,
        document_id=active_doc_id
    )
    
    # Evaluate user's previous answer using retrieved context & generate next doc-based question
    evaluation_and_next_q = evaluate_with_doc_context(
        user_answer=user_msg,
        context_chunks=context_chunks
    )
    
    state['knowledge']['retrieved_chunks'] = context_chunks
    state['messages'].append(AIMessage(content=evaluation_and_next_q))
    return state
```

---

## PHASE 8 — REST API Endpoints (Adding Knowledge Endpoints)

#### Knowledge Agent Endpoints (`/api/knowledge/`):
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/upload` | Upload custom study doc/PDF, chunk & embed into pgvector |
| GET | `/api/knowledge/documents` | List user's uploaded knowledge documents |
| POST | `/api/knowledge/start-quiz` | Start a quiz session from an uploaded document |
| POST | `/api/knowledge/answer` | Submit answer to document-grounded question |

---

## PHASE 9 — Next.js Frontend Dashboard (Updated)

- Add **"Knowledge Hub" / "Custom Notes Practice"** tab/page.
- Drag & Drop custom PDFs/Notes.
- Document-based Q&A mode UI with chunk reference badges.

---

## PHASE 10 — Testing, Streaming & Deployment

- Test RAG Retrieval accuracy & fallback when no document uploaded.
- Stream RAG Knowledge Agent responses via SSE (`StreamingResponse`).

---
---

# ⏱️ SECTION 2: Updated 20-Hour Hackathon Execution Timeline

```
HOUR  0-2  │ FOUNDATION & DATABASE
            │ - Models (User, Session, Message, ResumeEmbedding, KnowledgeEmbedding)
            │ - Docker postgres + pgvector setup
            │ - Verify FastAPI health check
            │
HOUR  2-4  │ PARSER & RAG ENGINE
            │ - pdf_parser.py (Resume + Custom Docs)
            │ - embeddings.py + retriever.py (pgvector search)
            │
HOUR  4-7  │ SHARED STATE & SUPERVISOR AGENT
            │ - state.py (AgentState with KnowledgeContext)
            │ - supervisor.py (5-intent routing logic)
            │ - LangGraph workflow assembly (5 nodes)
            │
HOUR  7-11 │ 5 CORE AGENTS IMPLEMENTATION
            │ - resume_agent.py
            │ - interview_agent.py
            │ - code_agent.py
            │ - learning_agent.py
            │ - knowledge_agent.py (RAG document Q&A + evaluation)
            │
HOUR 11-13 │ REST API ENDPOINTS
            │ - /api/resume, /api/interview, /api/code, /api/learning, /api/knowledge
            │ - Unified /api/chat router with Supervisor
            │
HOUR 13-17 │ FRONTEND DASHBOARD
            │ - Landing page, Resume page, Interview room, Code editor, Knowledge hub
            │
HOUR 17-19 │ STREAMING & INTEGRATION
            │ - SSE streaming for all 5 agents
            │ - Connect frontend components to backend APIs
            │
HOUR 19-20 │ POLISH & DEMO PREP
            │ - Test document upload → RAG quiz flow → Scorecard
            │ - Final demo recording
```

---

# 🎬 SECTION 3: 3-Minute Hackathon Demo Flow (With RAG Knowledge Agent)

1. **0:00 - 0:30 (Problem & Vision)**: "Generic interview prep sucks. We built a 5-agent AI Copilot."
2. **0:30 - 1:00 (Resume & Multi-Agent State)**: Upload Resume → Resume Agent scores it & populates shared state.
3. **1:00 - 1:45 (Mock Technical Interview & Supervisor Routing)**: Ask question → Supervisor routes to Interview Agent → Tailored interview streaming.
4. **1:45 - 2:30 (RAG Knowledge Agent Demo)**: Upload "System Design Notes.pdf" → RAG Agent chunks & embeds into pgvector → Asks question directly grounded in the notes → Evaluates user answer with source citation!
5. **2:30 - 3:00 (Code Review & Roadmap)**: Code agent review + Learning agent roadmap + Architecture closing.

---
