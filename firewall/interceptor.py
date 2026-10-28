"""
Context Rot Firewall — Interceptor
The main entry point. Wraps every LLM call made by the agent.

Usage:
    from firewall.interceptor import FirewallInterceptor
    firewall = FirewallInterceptor()
    messages, stats = firewall.process(messages, turn_index)

The interceptor runs the full pipeline:
  Chunk → Analyze → Score → Policy → Compress/Archive → Retrieve → Assemble
"""
import json
import time
from pathlib import Path

from .chunker import build_chunks, Chunk
from .analyzer import analyze
from .relevance import score_all
from .adaptive_policy import apply_policy
from .compressor import compress_chunk
from .storage import archive_chunk
from .retrieval import retrieve_relevant, build_retrieval_messages
from .assembler import assemble
from .relevance import embed
from .config_loader import cfg


class FirewallInterceptor:
    """
    The Context Rot Firewall.

    Maintains a persistent chunk registry across turns so chunk embeddings
    are not recomputed unless the content changes.
    """

    def __init__(self, enabled: bool = True, log_dir: str | None = None):
        self.enabled = enabled
        self.chunk_registry: dict[str, Chunk] = {}  # chunk_id → Chunk (persists across turns)
        self.log_dir = Path(log_dir or cfg["storage"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.turn_logs: list[dict] = []

    def process(
        self,
        messages: list[dict],
        turn_index: int,
        token_budget: int | None = None,
    ) -> tuple[list[dict], dict]:
        """
        Process the current message list through the firewall.

        Args:
            messages: Full message history to send to LLM (before management).
            turn_index: Current agent turn number.
            token_budget: Optional override for token budget.

        Returns:
            (managed_messages, stats) — managed_messages is what to actually send to LLM.
        """
        t_start = time.time()

        if not self.enabled:
            return messages, {"firewall": "disabled", "turn": turn_index}

        protected_roles = cfg["context"]["protected_roles"]

        # 1. Chunk the incoming messages
        new_chunks = build_chunks(messages, turn_index, protected_roles)

        # Merge with registry (update existing, add new)
        for chunk in new_chunks:
            if chunk.chunk_id not in self.chunk_registry:
                self.chunk_registry[chunk.chunk_id] = chunk
            else:
                # Preserve archived status from registry
                existing = self.chunk_registry[chunk.chunk_id]
                chunk.archived = existing.archived
                chunk.summary = existing.summary
                self.chunk_registry[chunk.chunk_id] = chunk

        all_chunks = list(self.chunk_registry.values())

        # 2. Analyze
        analysis = analyze(all_chunks)
        subgoal = analysis["current_subgoal"]

        # 3. Score relevance
        score_all(all_chunks, subgoal)

        # 4. Apply policy
        policy_summary = apply_policy(all_chunks, token_budget)

        # 5. Compress + Archive chunks assigned to compress/archive
        for chunk in all_chunks:
            if chunk.archived:
                continue  # Already in cold storage
            if chunk.policy_decision in ("compress", "archive"):
                # Compress first (get summary), then archive original
                if not chunk.summary:
                    compress_chunk(chunk)
                chunk_emb = embed(chunk.content).tolist()
                archive_chunk(chunk, embedding=chunk_emb)

        # 6. Retrieve relevant archived chunks
        active_ids = {c.chunk_id for c in all_chunks if not c.archived}
        retrieved = retrieve_relevant(subgoal, existing_chunk_ids=active_ids)
        retrieved_messages = build_retrieval_messages(retrieved)

        # 7. Assemble final context
        final_messages, assemble_stats = assemble(all_chunks, retrieved_messages, token_budget)

        t_elapsed = (time.time() - t_start) * 1000

        stats = {
            "turn": turn_index,
            "latency_ms": round(t_elapsed, 2),
            "original_message_count": len(messages),
            "analysis": analysis,
            "policy": policy_summary,
            "retrieved_count": len(retrieved),
            "assemble": assemble_stats,
            "chunk_scores": [
                {"id": c.chunk_id[:8], "role": c.role, "score": round(c.relevance_score, 3),
                 "decision": c.policy_decision, "protected": c.is_protected}
                for c in all_chunks
            ],
        }

        self._log_turn(stats)
        return final_messages, stats

    def _log_turn(self, stats: dict) -> None:
        self.turn_logs.append(stats)
        log_path = self.log_dir / "firewall_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    def get_dashboard_data(self) -> list[dict]:
        """Return all logged turn data for dashboard display."""
        return self.turn_logs
