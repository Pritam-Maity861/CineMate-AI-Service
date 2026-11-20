import hashlib
import logging
import re
from typing import Any, Dict, List, Optional
import numpy as np
from app.config import settings
from app.mcp.tools import node_client

logger = logging.getLogger(__name__)

# Dimension for dense embedding vectors
VECTOR_DIM = 256


def generate_local_embedding(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Generates a deterministic 256-dimensional unit dense vector from text
    using hashed n-grams and term frequencies.
    Acts as a zero-dependency local embedding engine when external cloud APIs are unavailable.
    """
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in clean.split() if len(w) > 1]

    vec = np.zeros(dim, dtype=np.float32)

    # 1. Word-level hashing
    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign

    # 2. Bi-gram hashing for phrases (e.g. "dark comedy", "mind bending")
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += 1.5 * sign

    # Normalize vector to unit length (L2 norm)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# Global cache for FastEmbed model instance
_fastembed_model = None


async def get_embedding(text: str) -> np.ndarray:
    """
    Returns an embedding vector:
    1. FastEmbed (100% Free, Local, No API key, runs on CPU via ONNX).
    2. Google Gemini Free Tier (text-embedding-004 if GEMINI_API_KEY is present).
    3. Fallback: Deterministic local dense vector engine (zero dependencies, 100% free).
    """
    global _fastembed_model

    # 1. Try FastEmbed (Free local neural model)
    try:
        from fastembed import TextEmbedding

        if _fastembed_model is None:
            _fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embeddings = list(_fastembed_model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)
    except (ImportError, Exception):
        pass

    # 2. Try Gemini Free API (if key provided)
    if settings.GEMINI_API_KEY:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embedder = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GEMINI_API_KEY,
                model=settings.EMBEDDING_MODEL,
            )
            emb = await embedder.aembed_query(text)
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Gemini embedding fallback: {e}")

    # 3. Try OpenAI (if key provided)
    if settings.OPENAI_API_KEY and settings.DEFAULT_MODEL_PROVIDER == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            embedder = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            emb = await embedder.aembed_query(text)
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.warning(f"OpenAI embedding fallback: {e}")

    # 4. Zero-dependency local dense vector fallback (Immediate & Free)
    return generate_local_embedding(text)


async def semantic_movie_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs semantic vector similarity search against the CineMate catalog.
    Matches movie vibes, themes, and plot synopses (e.g. 'mind-bending sci-fi with dark humor').
    """
    # 1. Fetch current approved movies catalog from Node.js backend
    try:
        catalog = await node_client.search_movies(limit=50)
        movies = catalog.get("movies", [])
    except Exception as e:
        logger.error(f"Failed to fetch catalog for semantic search: {e}")
        return []

    if not movies:
        return []

    # 2. Compute query embedding
    query_vec = generate_local_embedding(query)

    scored_movies = []
    for m in movies:
        genres = ", ".join([g.get("name", "") for g in m.get("genres", [])])
        doc_text = (
            f"Title: {m.get('title', '')}. "
            f"Genres: {genres}. "
            f"Synopsis: {m.get('description', '')}. "
            f"Duration: {m.get('duration', 0)} mins."
        )
        doc_vec = generate_local_embedding(doc_text)

        # Cosine similarity between unit vectors
        similarity = float(np.dot(query_vec, doc_vec))

        scored_movies.append(
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "genres": genres,
                "description": m.get("description"),
                "rating": m.get("rating"),
                "duration": m.get("duration"),
                "score": round(similarity, 3),
            }
        )

    # Sort descending by semantic similarity score
    scored_movies.sort(key=lambda x: x["score"], reverse=True)

    # Return top_k
    return scored_movies[:top_k]

