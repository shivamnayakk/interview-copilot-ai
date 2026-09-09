from sqlmodel import create_engine, Session, SQLModel, select
from backend.database.models import (
    UserProfile,
    InterviewSession,
    KnowledgeDocument,
    KnowledgeEmbedding,
)


test_engine = create_engine("sqlite:///:memory:")


def setup_module():
    SQLModel.metadata.create_all(test_engine)


def test_create_user_profile_and_session():
    with Session(test_engine) as session:
        user = UserProfile(
            name="John Doe",
            email="john@example.com",
            target_role="AI Engineer",
            skills_json='["Python", "FastAPI", "LangChain"]',
            weaknesses_json='["System Design"]',
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        assert user.id is not None
        assert user.name == "John Doe"

        interview_session = InterviewSession(
            user_id=user.id,
            session_type="interview",
            job_role="AI Engineer",
            status="active",
        )
        session.add(interview_session)
        session.commit()
        session.refresh(interview_session)

        assert interview_session.id is not None
        assert interview_session.user_id == user.id


def test_create_knowledge_embedding():
    with Session(test_engine) as session:
        user = UserProfile(name="Jane")
        session.add(user)
        session.commit()
        session.refresh(user)

        doc = KnowledgeDocument(
            user_id=user.id,
            document_title="System Design Notes.pdf",
            total_chunks=1,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)

        chunk = KnowledgeEmbedding(
            document_id=doc.id,
            user_id=user.id,
            chunk_text="Load balancer distributes incoming traffic.",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        session.add(chunk)
        session.commit()

        results = session.exec(
            select(KnowledgeEmbedding).where(KnowledgeEmbedding.document_id == doc.id)
        ).all()
        assert len(results) == 1
        assert results[0].chunk_text == "Load balancer distributes incoming traffic."
        assert results[0].embedding == [0.1, 0.2, 0.3]
