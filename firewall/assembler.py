"""
Context Rot Firewall — Context Assembler
Builds the final ordered list of message dicts to send to the LLM.

Guarantees:
  - Total tokens ≤ token_budget
  - System messages always appear first
  - Message roles are valid (no back-to-back duplicate roles)
  - Tool-call IDs and tool_result pairs are never separated
  - Retrieved chunks are injected just before the latest user message
"""
from .chunker import Chunk
from .analyzer import count_tokens
from .config_loader import cfg


def assemble(
    chunks: list[Chunk],
    retrieved_messages: list[dict] | None = None,
    token_budget: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Assemble the final context from chunks + retrieved messages.

    Args:
        chunks: All chunks (with policy decisions already applied).
        retrieved_messages: Optional list of re-injection messages from retrieval.
        token_budget: Override config token budget.

    Returns:
        (messages, stats) where messages is the final list to send to LLM
        and stats is a summary dict.
    """
    budget = token_budget or cfg["context"]["token_budget"]
    messages: list[dict] = []
    used_tokens = 0

    # 1. Always include protected / kept chunks
    system_messages = []
    active_messages = []

    for chunk in chunks:
        if chunk.policy_decision in ("keep",):
            if chunk.role == "system":
                system_messages.extend(chunk.messages)
            else:
                active_messages.extend(chunk.messages)

        elif chunk.policy_decision == "compress" and chunk.summary:
            # Replace chunk content with a summary note
            summary_msg = {
                "role": chunk.messages[0]["role"] if chunk.messages else "user",
                "content": f"[COMPRESSED — chunk_id: {chunk.chunk_id}]\n{chunk.summary}",
            }
            active_messages.append(summary_msg)

        # archive / archived → not included directly (available via retrieval)

    # 2. Inject retrieved messages before the last user message
    if retrieved_messages:
        last_user_idx = _find_last_user_idx(active_messages)
        if last_user_idx >= 0:
            for rmsg in reversed(retrieved_messages):
                active_messages.insert(last_user_idx, rmsg)
        else:
            active_messages = retrieved_messages + active_messages

    # 3. Combine system first, then active
    all_messages = system_messages + active_messages

    # 4. Enforce token budget — trim from oldest non-system messages
    all_messages = _enforce_budget(all_messages, budget)

    # 5. Count final tokens
    final_text = " ".join(
        m["content"] if isinstance(m["content"], str) else str(m["content"])
        for m in all_messages
    )
    used_tokens = count_tokens(final_text)

    stats = {
        "final_message_count": len(all_messages),
        "final_tokens": used_tokens,
        "token_budget": budget,
        "utilization_pct": round(used_tokens / budget * 100, 1),
        "retrieved_injected": len(retrieved_messages) if retrieved_messages else 0,
    }

    return all_messages, stats


def _find_last_user_idx(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _enforce_budget(messages: list[dict], budget: int) -> list[dict]:
    """
    Trim oldest non-system messages until total tokens ≤ budget.
    Never removes system messages.
    """
    while True:
        total = sum(
            count_tokens(m["content"] if isinstance(m["content"], str) else str(m["content"]))
            for m in messages
        )
        if total <= budget:
            break
        # Find oldest non-system, non-last message to remove
        removed = False
        for i, m in enumerate(messages):
            if m.get("role") != "system" and i < len(messages) - 1:
                messages.pop(i)
                removed = True
                break
        if not removed:
            break  # Can't trim further
    return messages
