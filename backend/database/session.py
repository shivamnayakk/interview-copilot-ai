from sqlalchemy import text
from sqlmodel import create_engine, Session, SQLModel
from backend.config import settings
import backend.database.models  # Register models with SQLModel metadata

engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)


def get_session():
    """FastAPI dependency to yield database sessions."""
    with Session(engine) as session:
        yield session


def init_db():
    """Enable pgvector (Postgres) and create tables."""
    with engine.connect() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
    SQLModel.metadata.create_all(engine)
