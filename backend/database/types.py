import json
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy.types import TEXT, TypeDecorator


class EmbeddingVector(TypeDecorator):
    """pgvector on PostgreSQL; JSON text on SQLite (unit tests)."""

    impl = TEXT
    cache_ok = True
    dimensions = 1536

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value: Optional[list[float]], dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect) -> Optional[list[float]]:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        if isinstance(value, str):
            return json.loads(value)
        return list(value)
