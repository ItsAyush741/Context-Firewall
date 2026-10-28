"""
Context Rot Firewall — Benchmark Runner

Runs all 4 configurations × 3 tasks and compares metrics.

Configurations:
  A — baseline:          Raw agent, no context management
  B — simple_compaction: Truncate oldest chunks at 75% budget (no intelligence)
  C — fixed_relevance:   Embedding scores, fixed thresholds, no budget awareness
  D — adaptive:          Full firewall (task-aware + budget-aware + retrieval)

Usage:
    python -m evaluation.benchmark
    python -m evaluation.benchmark --tasks forgotten_requirement
    python -m evaluation.benchmark --configs baseline adaptive
    python -m evaluation.benchmark --output results/benchmark_results.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.runner import run_agent
from evaluation.tasks import ALL_TASKS, EvalTask
from evaluation.metrics import collect_all_metrics
from firewall.config_loader import cfg

# ─── Configuration definitions ────────────────────────────────────────────────

# Config A: Baseline — runner's firewall is disabled
CONFIG_BASELINE = {
    "name": "baseline",
    "label": "A — Baseline (no management)",
    "mode": "baseline",
    "firewall_overrides": {},
}

# Config B: Simple compaction — firewall enabled, but thresholds set so that
# everything is "kept" until budget pressure kicks in, then oldest are dropped.
# We simulate this by setting very low thresholds (almost nothing is archived
# proactively) and high pressure ratio.
CONFIG_SIMPLE_COMPACTION = {
    "name": "simple_compaction",
    "label": "B — Simple Compaction (truncate oldest at 75% budget)",
    "mode": "firewall",
    "firewall_overrides": {
        # High threshold = nothing relevant unless exactly matching sub-goal
        # Low threshold below 0 = nothing proactively archived
        "relevance.high_threshold": 0.99,
        "relevance.low_threshold": -1.0,   # Nothing archived proactively
        "context.budget_pressure_ratio": 0.75,  # Start dropping at 75%
        "relevance.llm_judge_enabled": False,
    },
}

# Config C: Fixed relevance — proper embedding scoring, but no budget awareness
CONFIG_FIXED_RELEVANCE = {
    "name": "fixed_relevance",
    "label": "C — Fixed Relevance (embedding scores, no budget awareness)",
    "mode": "firewall",
    "firewall_overrides": {
        "context.budget_pressure_ratio": 1.0,  # Never trigger budget override
        "relevance.llm_judge_enabled": False,   # Tier-1 only
    },
}

# Config D: Adaptive — full system
CONFIG_ADAPTIVE = {
    "name": "adaptive",
    "label": "D — Adaptive Firewall (full system)",
    "mode": "firewall",
    "firewall_overrides": {},  # Use config.yaml defaults
}

ALL_CONFIGS = [
    CONFIG_BASELINE,
    CONFIG_SIMPLE_COMPACTION,
    CONFIG_FIXED_RELEVANCE,
    CONFIG_ADAPTIVE,
]


# ─── Config patching helpers ───────────────────────────────────────────────────

def _apply_overrides(overrides: dict) -> dict:
    """
    Temporarily patch the global cfg dict with override values.
    Override keys use dotted paths e.g. "relevance.high_threshold".
    Returns a snapshot of the original values to restore later.
    """
    snapshot = {}
    for dotted_key, new_val in overrides.items():
        parts = dotted_key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        leaf = parts[-1]
        snapshot[dotted_key] = d[leaf]
        d[leaf] = new_val
    return snapshot


def _restore_overrides(snapshot: dict) -> None:
    """Restore cfg to the state before _apply_overrides was called."""
    for dotted_key, old_val in snapshot.items():
        parts = dotted_key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = old_val


# ─── Single run ───────────────────────────────────────────────────────────────

def run_single(task: EvalTask, config: dict, log_dir: str) -> dict:
    """
    Run one (task, config) combination.

    Injects the preamble messages into the agent's initial history so it
    starts with the appropriate prior context, then continues with the task.
    """
    print(f"\n  ▶  Task: {task.name}  |  Config: {config['label']}")

    # Clone preamble so we don't mutate the task definition
    preamble = copy.deepcopy(task.preamble_messages)

    snapshot = _apply_overrides(config.get("firewall_overrides", {}))
    try:
        result = run_agent(
            task=task.description,
            mode=config["mode"],
            max_turns=task.max_turns,
            log_dir=log_dir,
            initial_messages=preamble,  # extra kwarg — see runner.py note below
        )
    except TypeError:
        # Runner doesn't support initial_messages yet — fall back to plain run
        result = run_agent(
            task=task.description,
            mode=config["mode"],
            max_turns=task.max_turns,
            log_dir=log_dir,
        )
    finally:
        _restore_overrides(snapshot)

    metrics = collect_all_metrics(result, task)
    metrics["config_name"] = config["name"]
    metrics["config_label"] = config["label"]
    return metrics


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(
    task_filter: list[str] | None = None,
    config_filter: list[str] | None = None,
    output_path: str | None = None,
    log_dir: str = "./logs/benchmark",
) -> list[dict]:
    """
    Run the full benchmark suite.

    Args:
        task_filter: If provided, only run tasks whose scenario_type matches.
        config_filter: If provided, only run configs whose name matches.
        output_path: Path to write JSON results. Defaults to logs/benchmark_results.json.
        log_dir: Directory for agent telemetry logs.

    Returns:
        List of metric dicts (one per task × config combination).
    """
    tasks = ALL_TASKS
    configs = ALL_CONFIGS

    if task_filter:
        tasks = [t for t in tasks if t.scenario_type in task_filter]
    if config_filter:
        configs = [c for c in configs if c["name"] in config_filter]

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Context Rot Firewall — Benchmark")
    print(f"  Tasks: {[t.name for t in tasks]}")
    print(f"  Configs: {[c['name'] for c in configs]}")
    print(f"{'='*60}\n")

    all_results: list[dict] = []
    total = len(tasks) * len(configs)
    done = 0

    for task in tasks:
        for config in configs:
            done += 1
            print(f"[{done}/{total}]", end="")
            t0 = time.time()
            try:
                metrics = run_single(task, config, log_dir)
                metrics["run_wall_time_s"] = round(time.time() - t0, 1)
                metrics["error"] = None
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                metrics = {
                    "task_name": task.name,
                    "config_name": config["name"],
                    "error": str(e),
                }
            all_results.append(metrics)

    # Save results
    out = output_path or f"{log_dir}/benchmark_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {out}")
    print(f"{'='*60}\n")

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[dict]) -> None:
    """Print a human-readable comparison table to stdout."""
    print(f"\n{'─'*80}")
    print(f"{'Task':<28} {'Config':<22} {'Success':>7} {'Tokens':>8} {'Ratio':>7} {'Lat(ms)':>9}")
    print(f"{'─'*80}")
    for r in results:
        if r.get("error"):
            print(f"  ERROR in {r.get('task_name')} / {r.get('config_name')}: {r['error']}")
            continue
        print(
            f"{r.get('task_name',''):<28} "
            f"{r.get('config_name',''):<22} "
            f"{'✓' if r.get('task_success') else '✗':>7} "
            f"{r.get('total_input_tokens', 0):>8,} "
            f"{r.get('compression_ratio', 0):>7.3f} "
            f"{r.get('avg_latency_ms', 0):>9.1f}"
        )
    print(f"{'─'*80}\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Rot Firewall — Benchmark Runner")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["forgotten_requirement", "recover_discovery", "multi_file_coding"],
        help="Scenario types to run (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["baseline", "simple_compaction", "fixed_relevance", "adaptive"],
        help="Configurations to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/benchmark",
        help="Directory for agent telemetry logs",
    )
    args = parser.parse_args()

    results = run_benchmark(
        task_filter=args.tasks,
        config_filter=args.configs,
        output_path=args.output,
        log_dir=args.log_dir,
    )
    print(f"Done. {len(results)} run(s) completed.")
