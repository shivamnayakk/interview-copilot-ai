import hashlib
import math
from typing import List
from backend.config import settings
from backend.utils.logger import logger


def _is_mock_mode() -> bool:
    """Return True if no real OpenAI key is configured."""
    key = settings.OPENAI_API_KEY or ""
    return key.startswith("sk-your") or key == "sk-placeholder" or len(key) < 20


def _mock_embedding(text: str, dims: int = 1536) -> List[float]:
    """Generate a deterministic unit-vector embedding from text hash (no API call)."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    rng_state = seed
    raw = []
    for _ in range(dims):
        rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw.append((rng_state / 0xFFFFFFFF) * 2 - 1)

    # Normalize to unit vector
    magnitude = math.sqrt(sum(v * v for v in raw))
    if magnitude == 0:
        return [0.0] * dims
    return [v / magnitude for v in raw]


def get_embedding(text: str) -> List[float]:
    """
    Convert text to a 1536-dim vector using OpenAI text-embedding-3-small.
    Falls back to deterministic mock embedding if API key is not configured.
    """
    if _is_mock_mode():
        logger.debug("Mock mode: generating deterministic embedding without API call.")
        return _mock_embedding(text)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip(),
        )
        embedding = response.data[0].embedding
        logger.debug(f"OpenAI embedding generated: {len(embedding)} dims.")
        return embedding
    except Exception as e:
        logger.warning(f"OpenAI embedding failed, using mock fallback: {e}")
        return _mock_embedding(text)


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently (single API call).
    Falls back to mock mode if no real API key is configured.
    """
    if not texts:
        return []

    if _is_mock_mode():
        logger.debug(f"Mock mode: generating {len(texts)} deterministic embeddings.")
        return [_mock_embedding(t) for t in texts]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        cleaned = [t.strip() for t in texts]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned,
        )
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        logger.info(f"OpenAI batch embeddings generated: {len(embeddings)} vectors.")
        return embeddings
    except Exception as e:
        logger.warning(f"OpenAI batch embedding failed, using mock fallback: {e}")
        return [_mock_embedding(t) for t in texts]
