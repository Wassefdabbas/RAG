"""
Text embedding using sentence-transformers.
Loads the model once at import time and reuses it for every call.
"""

from sentence_transformers import SentenceTransformer
from src.core.config import settings

# Loaded once, reused across calls — loading the model is the slow part,
# so we don't want to do it on every embed_text() call.
_model = SentenceTransformer(settings.embedding_model_name)


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text, returns a plain list of floats
    (ready to insert into a pgvector column)."""
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts at once — faster than calling embed_text()
    in a loop since the model can batch them."""
    vectors = _model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()