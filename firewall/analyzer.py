"""
Context Rot Firewall — Analyzer
Token counting, sub-goal extraction, dependency detection,
and protection tagging for Chunks.
"""
import re
import tiktoken
from .chunker import Chunk


_ENCODER = tiktoken.get_encoding("cl100k_base")

# Phrases that signal a new sub-goal when found in user messages
_SUBGOAL_MARKERS = [
    r"now\s+(let'?s|we'?ll|we should|I want|I need|please)",
    r"next,?\s+(let'?s|we|please)",
    r"step\s+\d+",
    r"task\s*:\s*",
    r"goal\s*:\s*",
    r"objective\s*:\s*",
]
_SUBGOAL_RE = re.compile("|".join(_SUBGOAL_MARKERS), re.IGNORECASE)

# Keywords that signal a chunk should be protected (never evicted)
_PROTECTION_KEYWORDS = [
    "must", "requirement", "constraint", "always", "never", "you must",
    "do not", "forbidden", "required", "mandatory", "rule:",
]


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base encoding."""
    return len(_ENCODER.encode(text, disallowed_special=()))


def count_chunk_tokens(chunk: Chunk) -> int:
    """Count tokens for a chunk (full content)."""
    return count_tokens(chunk.content)


def count_messages_tokens(messages: list[dict]) -> int:
    """Estimate tokens for a list of messages (adds per-message overhead)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content) + 4  # role + framing overhead
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += count_tokens(block["text"]) + 4
    return total


def extract_current_subgoal(chunks: list[Chunk]) -> str:
    """
    Extract the most recent sub-goal from the chunk list.
    Searches recent user messages for goal-signalling phrases.
    Falls back to the last user message content.
    """
    # Search recent user chunks (last 10) for sub-goal markers
    user_chunks = [c for c in chunks if c.role in ("user",)][-10:]
    for chunk in reversed(user_chunks):
        if _SUBGOAL_RE.search(chunk.content):
            return chunk.content[:512]

    # Fallback: last user message
    for chunk in reversed(chunks):
        if chunk.role == "user":
            return chunk.content[:512]

    return ""


def tag_protected_chunks(chunks: list[Chunk]) -> None:
    """
    Mark chunks as protected (is_protected=True) if they:
    - Have a protected role (system), already set by chunker
    - Contain hard constraint / requirement language
    - Are explicitly flagged in metadata
    """
    for chunk in chunks:
        if chunk.is_protected:
            continue
        text_lower = chunk.content.lower()
        if any(kw in text_lower for kw in _PROTECTION_KEYWORDS):
            chunk.is_protected = True
            chunk.metadata["protection_reason"] = "contains_constraint_language"


def tag_token_counts(chunks: list[Chunk]) -> None:
    """Attach token counts to each chunk's metadata."""
    for chunk in chunks:
        chunk.metadata["token_count"] = count_chunk_tokens(chunk)


def analyze(chunks: list[Chunk]) -> dict:
    """
    Run all analysis steps on the chunk list.
    Returns a summary dict with key context stats.
    """
    tag_token_counts(chunks)
    tag_protected_chunks(chunks)
    subgoal = extract_current_subgoal(chunks)

    total_tokens = sum(c.metadata.get("token_count", 0) for c in chunks)
    protected_count = sum(1 for c in chunks if c.is_protected)

    return {
        "total_tokens": total_tokens,
        "chunk_count": len(chunks),
        "protected_count": protected_count,
        "current_subgoal": subgoal,
    }
