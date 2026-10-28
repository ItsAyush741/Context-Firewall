"""
Context Rot Firewall — Test Suite
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── test_chunker.py ──────────────────────────────────────────────────────────
from firewall.chunker import build_chunks, Chunk


def test_system_message_is_protected():
    messages = [{"role": "system", "content": "You must always follow these rules."}]
    chunks = build_chunks(messages, turn_index=1, protected_roles=["system"])
    assert chunks[0].is_protected is True


def test_user_message_not_protected_by_default():
    messages = [{"role": "user", "content": "Hello, solve this task."}]
    chunks = build_chunks(messages, turn_index=1, protected_roles=["system"])
    assert chunks[0].is_protected is False


def test_tool_pair_grouped():
    messages = [
        {"role": "user", "content": "Do the task."},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "test.py"}}],
        },
        {"role": "tool", "content": "print('hello')"},
    ]
    chunks = build_chunks(messages, turn_index=1, protected_roles=["system"])
    roles = [c.role for c in chunks]
    assert "tool_pair" in roles


def test_chunk_ids_are_stable():
    messages = [{"role": "user", "content": "Exactly the same content"}]
    chunks1 = build_chunks(messages, turn_index=5, protected_roles=[])
    chunks2 = build_chunks(messages, turn_index=5, protected_roles=[])
    assert chunks1[0].chunk_id == chunks2[0].chunk_id


# ─── test_analyzer.py ─────────────────────────────────────────────────────────
from firewall.analyzer import count_tokens, tag_protected_chunks, tag_token_counts, extract_current_subgoal


def test_token_count_nonzero():
    assert count_tokens("Hello, world!") > 0


def test_token_count_empty():
    assert count_tokens("") == 0


def test_protection_by_constraint_language():
    messages = [{"role": "user", "content": "Requirement: you must never delete files."}]
    chunks = build_chunks(messages, turn_index=1, protected_roles=["system"])
    tag_protected_chunks(chunks)
    assert chunks[0].is_protected is True


def test_subgoal_extraction_fallback():
    messages = [{"role": "user", "content": "Write a hello world script."}]
    chunks = build_chunks(messages, turn_index=1, protected_roles=[])
    subgoal = extract_current_subgoal(chunks)
    assert "hello world" in subgoal.lower()


# ─── test_policy.py ───────────────────────────────────────────────────────────
from firewall.adaptive_policy import apply_policy


def _make_chunk(score: float, protected: bool = False, tokens: int = 100) -> Chunk:
    c = Chunk(
        chunk_id=f"test_{score}_{protected}",
        role="user",
        content="test content " * 20,
        messages=[{"role": "user", "content": "test"}],
        turn_index=1,
        is_protected=protected,
        relevance_score=score,
    )
    c.metadata["token_count"] = tokens
    return c


def test_protected_chunk_always_kept():
    chunk = _make_chunk(score=0.1, protected=True)
    apply_policy([chunk], token_budget=4096)
    assert chunk.policy_decision == "keep"


def test_high_score_kept():
    chunk = _make_chunk(score=0.9)
    apply_policy([chunk], token_budget=4096)
    assert chunk.policy_decision == "keep"


def test_low_score_archived():
    chunk = _make_chunk(score=0.1)
    apply_policy([chunk], token_budget=4096)
    assert chunk.policy_decision == "archive"


def test_mid_score_compressed():
    chunk = _make_chunk(score=0.55)
    apply_policy([chunk], token_budget=4096)
    assert chunk.policy_decision == "compress"


def test_budget_override_downgrades_keep():
    # 20 chunks × 300 tokens = 6000 > budget(4096) × 0.8 = 3276
    chunks = [_make_chunk(score=0.76, tokens=300) for _ in range(20)]
    apply_policy(chunks, token_budget=4096)
    decisions = [c.policy_decision for c in chunks]
    assert "compress" in decisions  # Some should be downgraded


# ─── test_assembler.py ────────────────────────────────────────────────────────
from firewall.assembler import assemble


def _kept_chunk(content: str, role: str = "user") -> Chunk:
    c = Chunk(
        chunk_id=f"asm_{role}_{hash(content)}",
        role=role,
        content=content,
        messages=[{"role": role, "content": content}],
        turn_index=1,
    )
    c.policy_decision = "keep"
    c.metadata["token_count"] = count_tokens(content)
    return c


def test_assembled_respects_budget():
    # Create chunks that total >> budget
    big_chunks = [_kept_chunk("word " * 500) for _ in range(5)]
    messages, stats = assemble(big_chunks, token_budget=512)
    assert stats["final_tokens"] <= 512 + 50  # allow small overflow from last message


def test_system_message_always_present():
    sys_chunk = _kept_chunk("You are a helpful assistant.", role="system")
    sys_chunk.is_protected = True
    user_chunk = _kept_chunk("Do the task.")
    messages, stats = assemble([sys_chunk, user_chunk], token_budget=4096)
    roles = [m["role"] for m in messages]
    assert "system" in roles


def test_retrieved_messages_injected():
    user_chunk = _kept_chunk("Current task: do something.")
    retrieved = [{"role": "user", "content": "[RETRIEVED] Earlier important finding."}]
    messages, stats = assemble([user_chunk], retrieved_messages=retrieved, token_budget=4096)
    contents = " ".join(m.get("content", "") for m in messages)
    assert "RETRIEVED" in contents
    assert stats["retrieved_injected"] == 1
