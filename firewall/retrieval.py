"""
Context Rot Firewall — Retrieval
Embeds the current sub-goal and queries cold storage for the most relevant
previously-archived chunks, then injects them back into the active context.
"""
import numpy as np
from .chunker import Chunk
from .relevance import embed
from .storage import retrieve_chunks
from .config_loader import cfg


def retrieve_relevant(subgoal: str, existing_chunk_ids: set[str]) -> list[dict]:
    """
    Retrieve top-k archived chunks most relevant to the current sub-goal.
    Skips chunks that are already in the active context.

    Args:
        subgoal: Current sub-goal text.
        existing_chunk_ids: Set of chunk_ids already in active context.

    Returns:
        List of retrieved chunk dicts (chunk_id, content, distance, metadata).
    """
    if not subgoal.strip():
        return []

    top_k = cfg["relevance"]["top_k_retrieval"]
    subgoal_emb = embed(subgoal).tolist()
    candidates = retrieve_chunks(subgoal_emb, top_k=top_k * 2)  # over-fetch, then filter

    results = []
    for c in candidates:
        if c["chunk_id"] in existing_chunk_ids:
            continue
        # Only inject if distance is good enough (cosine distance < 0.5 means similarity > 0.5)
        if c["distance"] < 0.5:
            results.append(c)
        if len(results) >= top_k:
            break

    return results


def build_retrieval_messages(retrieved: list[dict]) -> list[dict]:
    """
    Convert retrieved chunk dicts into message dicts to prepend to context.
    Tagged with [RETRIEVED] so the model knows these came from cold storage.
    """
    messages = []
    for r in retrieved:
        messages.append({
            "role": "user",
            "content": f"[RETRIEVED FROM CONTEXT ARCHIVE — chunk_id: {r['chunk_id']}]\n{r['content']}",
        })
    return messages
