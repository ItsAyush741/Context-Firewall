"""
Context Rot Firewall — Metrics Collector

Computes the six benchmark metrics from a completed agent run result dict:

  1. task_success          — bool: did the agent pass the scenario's success_fn?
  2. critical_info_retained — bool: did protected chunks survive? (from firewall log)
  3. total_input_tokens    — int: sum of input tokens across all turns
  4. compression_ratio     — float: compressed tokens / original tokens (0.0 if no firewall)
  5. avg_latency_ms        — float: average wall-clock time per turn
  6. retrieval_accuracy    — float: fraction of retrieved chunks with cosine distance ≤ 0.3

Each function accepts a result dict as returned by agent.runner.run_agent().
"""
from __future__ import annotations

from typing import Any


def metric_task_success(result: dict, task) -> bool:
    """Whether the agent passed the task-specific success criterion."""
    return bool(task.success_fn(result))


def metric_critical_info_retained(result: dict) -> bool:
    """
    Check if all protected chunks were kept (never archived) across all turns.
    Returns True if no protected chunk was given an 'archive' or 'compress' decision.
    Returns True trivially in baseline mode (no firewall log).
    """
    fw_log = result.get("firewall_log", [])
    if not fw_log:
        return True  # Baseline — no management; can't measure

    for turn in fw_log:
        for cs in turn.get("chunk_scores", []):
            if cs.get("protected") and cs.get("decision") in ("archive", "compress"):
                return False
    return True


def metric_total_input_tokens(result: dict) -> int:
    """Total tokens fed to the LLM as input across all turns."""
    return result.get("total_input_tokens", 0)


def metric_compression_ratio(result: dict) -> float:
    """
    Compression ratio = (total managed tokens sent) / (total raw tokens if unmanaged).

    Computed from the firewall log: for each turn, compare final_tokens (what was
    actually sent) to original_message_count × ~100 tokens (rough estimate of
    what the baseline would have sent).

    Returns 1.0 (no compression) when running in baseline mode.
    """
    fw_log = result.get("firewall_log", [])
    if not fw_log:
        return 1.0

    managed_total = sum(t.get("assemble", {}).get("final_tokens", 0) for t in fw_log)
    # Estimate unmanaged: original_message_count * avg tokens per message (≈ 80)
    raw_total = sum(t.get("original_message_count", 0) * 80 for t in fw_log)

    if raw_total == 0:
        return 1.0
    return round(managed_total / raw_total, 4)


def metric_avg_latency_ms(result: dict) -> float:
    """
    Average wall-clock time per turn (LLM call only in baseline; LLM + firewall in managed mode).
    Uses llm_latency_ms from the transcript for both modes.
    """
    transcript = result.get("transcript", [])
    if not transcript:
        return 0.0
    latencies = [t.get("llm_latency_ms", 0) for t in transcript]
    return round(sum(latencies) / len(latencies), 2)


def metric_retrieval_accuracy(result: dict, distance_threshold: float = 0.3) -> float:
    """
    Fraction of retrieved chunks that were semantically relevant (cosine distance ≤ threshold).

    ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
    We consider distance ≤ 0.3 as "accurate retrieval".
    Returns 1.0 if no retrieval was attempted (baseline or no archives).
    """
    fw_log = result.get("firewall_log", [])
    if not fw_log:
        return 1.0

    total_retrieved = sum(t.get("retrieved_count", 0) for t in fw_log)
    if total_retrieved == 0:
        return 1.0  # No retrieval — N/A, treat as perfect

    # We don't have per-chunk distance in the firewall log at this level.
    # Approximate: if retrieved_count > 0 for a turn, that's a retrieval event.
    # For a more accurate metric, the interceptor would need to log distances.
    # For now we return a sentinel value of -1.0 to indicate "not measurable here"
    # unless the user adds distance logging.
    return -1.0  # Placeholder — enhance by logging distances in interceptor


def collect_all_metrics(result: dict, task) -> dict:
    """
    Collect all six metrics for a single run and return as a flat dict.

    Args:
        result: Return value from agent.runner.run_agent()
        task: EvalTask instance defining success_fn

    Returns:
        Dict with keys matching the 6 metrics.
    """
    return {
        "task_name": task.name,
        "scenario_type": task.scenario_type,
        "mode": result.get("mode", "unknown"),
        "turns": result.get("turns", 0),
        "task_success": metric_task_success(result, task),
        "critical_info_retained": metric_critical_info_retained(result),
        "total_input_tokens": metric_total_input_tokens(result),
        "compression_ratio": metric_compression_ratio(result),
        "avg_latency_ms": metric_avg_latency_ms(result),
        "retrieval_accuracy": metric_retrieval_accuracy(result),
    }
