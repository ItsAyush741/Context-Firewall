"""
Context Rot Firewall — Chunker
Converts a flat message list into Chunks with stable IDs and metadata.
Each Chunk groups one logical unit: a user/assistant message, or a
tool_use + tool_result pair (kept together — never separated).
"""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str                  # Stable hash-based ID
    role: str                      # "system" | "user" | "assistant" | "tool_pair"
    content: str                   # Full text content (for token counting + embedding)
    messages: list[dict]           # Original message dicts (preserved for LLM)
    turn_index: int                # Which turn this chunk was created on
    timestamp: float = field(default_factory=time.time)
    is_protected: bool = False     # Protected chunks are never evicted
    relevance_score: float = 1.0   # Set by relevance module; 1.0 = always keep
    policy_decision: str = "keep"  # keep | compress | archive | retrieve
    summary: str | None = None     # Set if compressed
    archived: bool = False         # True after moved to cold storage
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "role": self.role,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "turn_index": self.turn_index,
            "is_protected": self.is_protected,
            "relevance_score": round(self.relevance_score, 4),
            "policy_decision": self.policy_decision,
            "archived": self.archived,
        }


def _make_chunk_id(role: str, content: str, turn_index: int) -> str:
    """Stable deterministic ID based on content hash."""
    raw = f"{turn_index}:{role}:{content[:512]}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_chunks(messages: list[dict], turn_index: int, protected_roles: list[str]) -> list[Chunk]:
    """
    Convert a flat message list into Chunks.

    Tool-call pairs (assistant message with tool_use + subsequent tool_result)
    are grouped into a single Chunk to preserve their relationship.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        turn_index: Current agent turn number.
        protected_roles: Roles that are always marked is_protected=True.

    Returns:
        Ordered list of Chunk objects.
    """
    chunks: list[Chunk] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")
        content = _extract_text(msg)

        # Group tool-use with its paired tool_result
        if role == "assistant" and _has_tool_use(msg) and i + 1 < len(messages):
            next_msg = messages[i + 1]
            if next_msg.get("role") == "tool":
                paired_content = content + "\n[TOOL RESULT]\n" + _extract_text(next_msg)
                chunk = Chunk(
                    chunk_id=_make_chunk_id("tool_pair", paired_content, turn_index),
                    role="tool_pair",
                    content=paired_content,
                    messages=[msg, next_msg],
                    turn_index=turn_index,
                    is_protected=False,  # Tool pairs can be archived
                )
                chunks.append(chunk)
                i += 2
                continue

        chunk = Chunk(
            chunk_id=_make_chunk_id(role, content, turn_index),
            role=role,
            content=content,
            messages=[msg],
            turn_index=turn_index,
            is_protected=(role in protected_roles),
        )
        chunks.append(chunk)
        i += 1

    return chunks


def _extract_text(msg: dict) -> str:
    """Extract plain text from a message dict (handles str and list content)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[TOOL: {block.get('name')}] Input: {block.get('input', {})}")
                elif block.get("type") == "tool_result":
                    parts.append(f"[TOOL RESULT]: {block.get('content', '')}")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _has_tool_use(msg: dict) -> bool:
    content = msg.get("content", "")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False
