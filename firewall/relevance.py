"""
Context Rot Firewall — Relevance Scorer
Two-tier scoring:
  Tier 1 (all chunks): cosine similarity between chunk embedding and sub-goal embedding
  Tier 2 (borderline only): LLM-judged 0-10 relevance score (normalized to 0-1)

Embeddings are cached per chunk_id to avoid recomputing on every turn.
"""
from __future__ import annotations

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk
from .config_loader import cfg

_EMBED_MODEL: SentenceTransformer | None = None
_EMBED_CACHE: dict[str, np.ndarray] = {}   # chunk_id → embedding


def _get_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        model_name = cfg["relevance"]["embedding_model"]
        print(f"[Relevance] Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def embed(text: str) -> np.ndarray:
    """Compute a normalized embedding for a text string."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors → [0, 1]."""
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def score_tier1(chunks: list[Chunk], subgoal: str) -> None:
    """
    Score all chunks against the current sub-goal using embedding cosine similarity.
    Updates chunk.relevance_score in place.
    Caches embeddings by chunk_id.
    """
    if not subgoal.strip():
        # No sub-goal detected — give everything a neutral score
        for chunk in chunks:
            chunk.relevance_score = 0.6
        return

    subgoal_emb = embed(subgoal)

    for chunk in chunks:
        if chunk.is_protected:
            chunk.relevance_score = 1.0
            continue

        if chunk.chunk_id not in _EMBED_CACHE:
            _EMBED_CACHE[chunk.chunk_id] = embed(chunk.content)

        chunk.relevance_score = cosine_similarity(_EMBED_CACHE[chunk.chunk_id], subgoal_emb)


def score_tier2_llm(chunk: Chunk, subgoal: str) -> float:
    """
    Ask the local Ollama LLM to judge relevance (0-10) for a borderline chunk.
    Returns a normalized score in [0, 1]. Falls back to previous score on error.
    """
    if not cfg["relevance"].get("llm_judge_enabled", True):
        return chunk.relevance_score

    prompt = f"""You are a context relevance judge.

Current task/sub-goal:
{subgoal[:400]}

Context chunk:
{chunk.content[:600]}

On a scale of 0 to 10, how relevant is the chunk to the current sub-goal?
- 10 = directly needed to accomplish the sub-goal
- 0  = completely unrelated, old, or noise

Reply with ONLY the integer score, nothing else."""

    try:
        resp = httpx.post(
            f"{cfg['llm']['ollama_url']}/api/generate",
            json={
                "model": cfg["llm"]["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 4},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "5").strip()
        score = int("".join(c for c in raw if c.isdigit()) or "5")
        return min(max(score, 0), 10) / 10.0
    except Exception as e:
        print(f"[Relevance][Tier2] LLM judge failed: {e} — using Tier-1 score")
        return chunk.relevance_score


def score_all(chunks: list[Chunk], subgoal: str) -> None:
    """
    Full two-tier scoring pipeline.
    After this call, every chunk.relevance_score is set.
    """
    high = cfg["relevance"]["high_threshold"]
    low = cfg["relevance"]["low_threshold"]

    # Tier 1: fast embedding scoring for everyone
    score_tier1(chunks, subgoal)

    # Tier 2: LLM judge only for borderline chunks (between low and high threshold)
    borderline = [c for c in chunks if not c.is_protected and low <= c.relevance_score < high]
    for chunk in borderline:
        chunk.relevance_score = score_tier2_llm(chunk, subgoal)
