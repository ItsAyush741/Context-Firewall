"""
Context Rot Firewall — Adaptive Policy Engine
Assigns a policy decision (keep / compress / archive / retrieve) to each chunk.

Rules (in priority order):
  1. Protected chunks → KEEP always
  2. relevance_score >= high_threshold → KEEP
  3. relevance_score >= low_threshold  → COMPRESS (summarize, archive original)
  4. relevance_score <  low_threshold  → ARCHIVE (full text to cold storage)

Budget override: if total kept tokens > budget × pressure_ratio,
  downgrade KEEP→COMPRESS for the lowest-scoring kept chunks until budget is satisfied.
"""
from .chunker import Chunk
from .config_loader import cfg


def apply_policy(chunks: list[Chunk], token_budget: int | None = None) -> dict:
    """
    Apply the adaptive policy to a list of scored chunks.
    Updates chunk.policy_decision in place.

    Returns a summary dict with counts per decision.
    """
    budget = token_budget or cfg["context"]["token_budget"]
    high = cfg["relevance"]["high_threshold"]
    low = cfg["relevance"]["low_threshold"]
    pressure = cfg["context"]["budget_pressure_ratio"]

    # Step 1: Initial rule-based assignment
    for chunk in chunks:
        if chunk.archived:
            # Already in cold storage; don't re-archive unless retrieved
            chunk.policy_decision = "archived"
            continue

        if chunk.is_protected:
            chunk.policy_decision = "keep"
        elif chunk.relevance_score >= high:
            chunk.policy_decision = "keep"
        elif chunk.relevance_score >= low:
            chunk.policy_decision = "compress"
        else:
            chunk.policy_decision = "archive"

    # Step 2: Budget pressure override
    kept_chunks = [c for c in chunks if c.policy_decision == "keep"]
    kept_tokens = sum(c.metadata.get("token_count", 0) for c in kept_chunks)

    if kept_tokens > budget * pressure:
        _apply_budget_override(chunks, kept_tokens, budget, pressure)

    return _summary(chunks)


def _apply_budget_override(chunks: list[Chunk], kept_tokens: int, budget: int, pressure: float) -> None:
    """
    Downgrade non-protected KEEP chunks to COMPRESS, starting from lowest relevance,
    until kept_tokens <= budget * pressure.
    """
    target = int(budget * pressure)
    # Sort non-protected kept chunks by relevance ascending (lowest first = most expendable)
    expendable = sorted(
        [c for c in chunks if c.policy_decision == "keep" and not c.is_protected],
        key=lambda c: c.relevance_score,
    )

    for chunk in expendable:
        if kept_tokens <= target:
            break
        chunk.policy_decision = "compress"
        chunk.metadata["budget_override"] = True
        kept_tokens -= chunk.metadata.get("token_count", 0)


def _summary(chunks: list[Chunk]) -> dict:
    from collections import Counter
    counts = Counter(c.policy_decision for c in chunks)
    return {
        "keep": counts.get("keep", 0),
        "compress": counts.get("compress", 0),
        "archive": counts.get("archive", 0),
        "archived": counts.get("archived", 0),
        "total": len(chunks),
    }
