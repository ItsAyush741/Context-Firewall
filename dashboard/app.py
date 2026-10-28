"""
Context Rot Firewall — Enhanced Streamlit Dashboard (Phase 6)

Features:
  - Live context health tab: token budget bar, relevance histogram,
    decision log, sub-goal display, latency trend
  - Benchmark results tab: multi-metric comparison table, success-rate
    bar chart, token-usage comparison, compression-ratio comparison,
    per-task heatmap
  - Demo / simulation tab: generate synthetic run data to showcase
    the dashboard without needing Ollama running
  - System info sidebar with config at a glance

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import streamlit as st
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from firewall.config_loader import cfg

# ─── Paths ────────────────────────────────────────────────────────────────────

LOG_DIR   = Path(cfg["storage"]["log_dir"])
LOG_FILE  = LOG_DIR / "firewall_log.jsonl"
BENCH_DIR = LOG_DIR / "benchmark"
BENCH_FILE = BENCH_DIR / "benchmark_results.json"

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Context Rot Firewall",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp { background: #0d1117; }

  /* ── Header ── */
  .page-header {
    background: linear-gradient(135deg, #0f1117 0%, #161b27 50%, #0f1420 100%);
    border: 1px solid #21293a;
    border-radius: 16px;
    padding: 1.6rem 2rem 1.2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .page-header::before {
    content: "";
    position: absolute;
    top: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(124,158,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .page-header-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c9eff 0%, #a78bfa 50%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .page-header-sub {
    color: #6b7a99;
    font-size: 0.9rem;
    margin-top: 0.2rem;
  }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 6px;
    vertical-align: middle;
  }
  .badge-green  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
  .badge-yellow { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
  .badge-red    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
  .badge-blue   { background: rgba(124,158,255,0.15); color: #7c9eff; border: 1px solid rgba(124,158,255,0.3); }
  .badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }

  /* ── Metric cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
  .kpi-card {
    background: linear-gradient(145deg, #161b27, #1a2035);
    border: 1px solid #21293a;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    transition: border-color 0.25s;
  }
  .kpi-card:hover { border-color: #7c9eff55; }
  .kpi-label { font-size: 0.7rem; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem; }
  .kpi-value { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; line-height: 1; }
  .kpi-sub   { font-size: 0.72rem; color: #8892a4; margin-top: 0.25rem; }

  /* ── Budget bar ── */
  .budget-bar-wrap {
    background: #161b27;
    border: 1px solid #21293a;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
  }
  .budget-bar-track {
    background: #1e2639;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin: 0.45rem 0;
  }
  .budget-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
  }

  /* ── Decision log ── */
  .chunk-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.8rem;
    background: #11161e;
  }
  .chunk-row:hover { background: #161d2a; }
  .chunk-id   { color: #6b7a99; font-family: monospace; min-width: 80px; }
  .d-keep     { color: #4ade80; font-weight: 600; }
  .d-compress { color: #fbbf24; font-weight: 600; }
  .d-archive  { color: #f87171; font-weight: 600; }

  /* ── Section headings ── */
  .section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #7c9eff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.1rem 0 0.5rem 0;
  }

  /* ── Heatmap cell ── */
  .heatmap-cell {
    display: inline-block;
    width: 100%;
    text-align: center;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 0;
  }

  /* ── Demo banner ── */
  .demo-banner {
    background: linear-gradient(90deg, rgba(167,139,250,0.1), rgba(232,121,249,0.1));
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #c4b5fd;
    margin-bottom: 1rem;
  }

  /* ── Info pill under charts ── */
  .info-pill {
    display: inline-block;
    background: rgba(124,158,255,0.1);
    border: 1px solid rgba(124,158,255,0.2);
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #7c9eff;
    margin-right: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_live_log() -> list[dict]:
    """Load the live firewall JSONL log."""
    if not LOG_FILE.exists():
        return []
    turns = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    turns.append(json.loads(line))
                except Exception:
                    pass
    return turns


def load_benchmark() -> list[dict]:
    """Load benchmark results JSON."""
    if not BENCH_FILE.exists():
        return []
    try:
        with open(BENCH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ─── Synthetic demo data ───────────────────────────────────────────────────────

def _make_demo_turns(n_turns: int = 18, seed: int = 42) -> list[dict]:
    """Generate synthetic firewall log turns for dashboard demo."""
    rnd = random.Random(seed)
    turns = []
    budget = cfg["context"]["token_budget"]
    decisions_pool = ["keep", "keep", "keep", "compress", "compress", "archive"]
    roles = ["user", "assistant", "tool"]

    for i in range(1, n_turns + 1):
        n_chunks = rnd.randint(4, 12)
        chunk_scores = []
        keep = compress = archive = 0
        for c in range(n_chunks):
            score = rnd.random()
            if score >= 0.75:
                decision = "keep"
                keep += 1
            elif score >= 0.40:
                decision = "compress"
                compress += 1
            else:
                decision = "archive"
                archive += 1
            chunk_scores.append({
                "id": f"chunk-{i:02d}-{c:02d}",
                "score": round(score, 4),
                "decision": decision,
                "role": rnd.choice(roles),
                "protected": c == 0,  # first chunk always protected
            })

        # Token usage grows then stabilises
        raw_tokens = min(budget, int(180 * i + rnd.randint(-80, 80)))
        managed = max(300, int(raw_tokens * (0.4 + rnd.random() * 0.35)))
        managed = min(managed, budget)
        util_pct = round(managed / budget * 100, 1)

        turns.append({
            "turn": i,
            "original_message_count": n_chunks,
            "chunk_scores": chunk_scores,
            "policy": {"keep": keep, "compress": compress, "archive": archive},
            "assemble": {
                "final_tokens": managed,
                "token_budget": budget,
                "utilization_pct": util_pct,
            },
            "analysis": {
                "current_subgoal": rnd.choice([
                    "Edit tests/test_utils.py to fix import paths",
                    "Analyse CSV file and plot statistics",
                    "Refactor authentication middleware",
                    "Fix failing unit tests in module A",
                    "Write docstrings for the data pipeline functions",
                ]),
            },
            "latency_ms": rnd.randint(60, 340),
            "retrieved_count": rnd.choice([0, 0, 0, 1, 2]),
        })
    return turns


def _make_demo_benchmark() -> list[dict]:
    """Generate synthetic benchmark results for the four configurations."""
    rnd = random.Random(7)
    configs = [
        {"name": "baseline",          "label": "A — Baseline",             "success_base": 0.38, "tok_mult": 1.0,  "comp": 1.0,  "lat": 210},
        {"name": "simple_compaction", "label": "B — Simple Compaction",    "success_base": 0.52, "tok_mult": 0.82, "comp": 0.82, "lat": 230},
        {"name": "fixed_relevance",   "label": "C — Fixed Relevance",      "success_base": 0.63, "tok_mult": 0.68, "comp": 0.68, "lat": 260},
        {"name": "adaptive",          "label": "D — Adaptive Firewall",    "success_base": 0.85, "tok_mult": 0.51, "comp": 0.51, "lat": 310},
    ]
    tasks = [
        ("forgotten_requirement", "Forgotten Requirement"),
        ("recover_discovery",     "Recover Discovery"),
        ("multi_file_coding",     "Multi-file Coding"),
    ]
    rows = []
    for task_type, task_name in tasks:
        for c in configs:
            success = min(1.0, c["success_base"] + rnd.uniform(-0.08, 0.08))
            tokens = int(cfg["context"]["token_budget"] * c["tok_mult"] * rnd.uniform(0.85, 1.15) * 12)
            comp = round(c["comp"] * rnd.uniform(0.9, 1.1), 4)
            lat = round(c["lat"] * rnd.uniform(0.85, 1.15), 1)
            rows.append({
                "task_name": task_name,
                "scenario_type": task_type,
                "config_name": c["name"],
                "config_label": c["label"],
                "task_success": success >= 0.5,
                "critical_info_retained": c["name"] == "adaptive" or success >= 0.6,
                "total_input_tokens": tokens,
                "compression_ratio": comp,
                "avg_latency_ms": lat,
                "retrieval_accuracy": 1.0 if c["name"] in ("fixed_relevance", "adaptive") else -1.0,
                "turns": rnd.randint(18, 45),
                "error": None,
            })
    return rows


# ─── Header ───────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="page-header">
      <p class="page-header-title">🔥 Context Rot Firewall</p>
      <p class="page-header-sub">
        Model-agnostic middleware that keeps AI agent context relevant, compressed, and recoverable.
        &nbsp;
        <span class="badge badge-green">Phase 6 Dashboard</span>
        <span class="badge badge-blue">Streamlit</span>
        <span class="badge badge-purple">ChromaDB</span>
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Live context tab ─────────────────────────────────────────────────────────

def _budget_color(pct: float) -> str:
    if pct < 60:   return "#4ade80"
    if pct < 85:   return "#fbbf24"
    return "#f87171"


def render_live_kpis(turns: list[dict]):
    if not turns:
        st.info("⏳ No firewall data yet. Run the agent with `--mode firewall` to populate this panel, or switch to the **🎭 Demo** tab.")
        return

    latest = turns[-1]
    asm = latest.get("assemble", {})
    pol = latest.get("policy", {})
    tokens = asm.get("final_tokens", 0)
    budget = asm.get("token_budget", cfg["context"]["token_budget"])
    util   = asm.get("utilization_pct", 0)
    color  = _budget_color(util)

    # KPI Cards via HTML
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Current Turn</div>
        <div class="kpi-value">{latest.get("turn","—")}</div>
        <div class="kpi-sub">of {len(turns)} total turns</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Tokens Used</div>
        <div class="kpi-value">{tokens:,}</div>
        <div class="kpi-sub">{util:.1f}% of {budget:,} budget</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Chunks Kept</div>
        <div class="kpi-value" style="color:#4ade80">{pol.get("keep", 0)}</div>
        <div class="kpi-sub">verbatim in context</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Compressed</div>
        <div class="kpi-value" style="color:#fbbf24">{pol.get("compress", 0)}</div>
        <div class="kpi-sub">summarised + archived</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Archived</div>
        <div class="kpi-value" style="color:#f87171">{pol.get("archive", 0)}</div>
        <div class="kpi-sub">in ChromaDB cold store</div>
      </div>
    </div>

    <div class="budget-bar-wrap">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.72rem;color:#6b7a99;text-transform:uppercase;letter-spacing:0.07em;">Token Budget Usage</span>
        <span style="font-size:0.75rem;color:{color};font-weight:600;">{util:.1f}%</span>
      </div>
      <div class="budget-bar-track">
        <div class="budget-bar-fill" style="width:{min(util,100):.1f}%;background:{color};"></div>
      </div>
      <div style="font-size:0.7rem;color:#6b7a99;">{tokens:,} / {budget:,} tokens</div>
    </div>
    """, unsafe_allow_html=True)


def render_token_trend(turns: list[dict]):
    if len(turns) < 2:
        return
    try:
        import pandas as pd
        st.markdown('<div class="section-title">Token Usage Over Turns</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Turn": [t["turn"] for t in turns],
            "Managed Context": [t.get("assemble", {}).get("final_tokens", 0) for t in turns],
            "Raw Message Count × 80": [t.get("original_message_count", 0) * 80 for t in turns],
        }).set_index("Turn")
        st.line_chart(df, color=["#7c9eff", "#f87171"])
        st.markdown('<span class="info-pill">Blue = tokens actually sent to LLM</span><span class="info-pill">Red = estimated without firewall</span>', unsafe_allow_html=True)
    except ImportError:
        st.warning("Install pandas for charts.")


def render_relevance_histogram(turns: list[dict]):
    if not turns:
        return
    rows = []
    for t in turns:
        for cs in t.get("chunk_scores", []):
            rows.append(cs["score"])
    if not rows:
        return
    try:
        import pandas as pd
        import collections
        st.markdown('<div class="section-title">Relevance Score Distribution</div>', unsafe_allow_html=True)
        buckets: dict[float, int] = collections.Counter()
        for s in rows:
            b = round(s * 10) / 10
            buckets[b] += 1
        df = pd.DataFrame({"Score": sorted(buckets), "Chunks": [buckets[k] for k in sorted(buckets)]}).set_index("Score")
        st.bar_chart(df, color="#a78bfa")
        hi = cfg["relevance"]["high_threshold"]
        lo = cfg["relevance"]["low_threshold"]
        st.markdown(f'<span class="info-pill">KEEP ≥ {hi}</span><span class="info-pill">ARCHIVE &lt; {lo}</span><span class="info-pill">COMPRESS in between</span>', unsafe_allow_html=True)
    except ImportError:
        pass


def render_latency_chart(turns: list[dict]):
    if len(turns) < 2:
        return
    try:
        import pandas as pd
        st.markdown('<div class="section-title">Firewall Overhead per Turn (ms)</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Turn": [t["turn"] for t in turns],
            "Latency (ms)": [t.get("latency_ms", 0) for t in turns],
        }).set_index("Turn")
        st.area_chart(df, color="#e879f9")
    except ImportError:
        pass


def render_decision_log(turns: list[dict]):
    if not turns:
        return
    latest = turns[-1]
    st.markdown('<div class="section-title">Chunk Decisions — Latest Turn</div>', unsafe_allow_html=True)

    icon_map  = {"keep": "🟢", "compress": "🟡", "archive": "🔴"}
    class_map = {"keep": "d-keep", "compress": "d-compress", "archive": "d-archive"}

    html_rows = ""
    for cs in latest.get("chunk_scores", []):
        d        = cs["decision"]
        icon     = icon_map.get(d, "⚪")
        cls      = class_map.get(d, "")
        lock     = " 🔒" if cs["protected"] else ""
        html_rows += f"""
        <div class="chunk-row">
          <span class="chunk-id">{cs['id']}</span>
          <span>{icon}</span>
          <span class="{cls}">{d}</span>{lock}
          <span style="color:#6b7a99;font-size:0.75rem;">role: {cs['role']}</span>
          <span style="margin-left:auto;color:#8892a4;font-size:0.75rem;">score {cs['score']:.3f}</span>
        </div>"""
    st.markdown(html_rows, unsafe_allow_html=True)


def render_subgoal(turns: list[dict]):
    if not turns:
        return
    sg = turns[-1].get("analysis", {}).get("current_subgoal", "")
    if sg:
        st.markdown('<div class="section-title">Detected Sub-Goal</div>', unsafe_allow_html=True)
        st.info(f"🎯 {sg[:420]}")


# ─── Benchmark tab ────────────────────────────────────────────────────────────

def render_benchmark(data: list[dict], demo: bool = False):
    if not data:
        st.info("No benchmark results yet.\n\nRun:\n```\npython -m evaluation.benchmark\n```")
        return

    if demo:
        st.markdown('<div class="demo-banner">🎭 Showing <strong>synthetic demo data</strong> — run <code>python -m evaluation.benchmark</code> for real results.</div>', unsafe_allow_html=True)

    try:
        import pandas as pd

        df = pd.DataFrame(data)
        if "error" in df.columns:
            df = df[df["error"].isna()].copy()
        if df.empty:
            st.warning("All benchmark runs errored.")
            return

        # ── Summary table ────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Full Results Table</div>', unsafe_allow_html=True)
        show_cols = ["task_name", "config_name", "task_success", "total_input_tokens",
                     "compression_ratio", "avg_latency_ms", "turns"]
        rename = {
            "task_name": "Task", "config_name": "Config",
            "task_success": "✅ Success", "total_input_tokens": "Input Tokens",
            "compression_ratio": "Compress Ratio", "avg_latency_ms": "Avg Latency (ms)",
            "turns": "Turns",
        }
        existing = [c for c in show_cols if c in df.columns]
        styled = df[existing].rename(columns=rename)
        st.dataframe(styled, use_container_width=True, height=260)

        st.divider()

        col_a, col_b = st.columns(2)

        # ── Success rate by config ───────────────────────────────────────────
        with col_a:
            st.markdown('<div class="section-title">Task Success Rate by Config</div>', unsafe_allow_html=True)
            if "task_success" in df.columns:
                sr = (
                    df.groupby("config_name")["task_success"]
                    .apply(lambda x: (x == True).mean())  # noqa
                    .rename("Success Rate")
                    .reset_index()
                    .set_index("config_name")
                )
                st.bar_chart(sr, color="#4ade80")
                st.caption("Higher is better. Adaptive firewall retains critical context across 40+ turns.")

        # ── Avg tokens by config ─────────────────────────────────────────────
        with col_b:
            st.markdown('<div class="section-title">Avg Input Tokens by Config</div>', unsafe_allow_html=True)
            if "total_input_tokens" in df.columns:
                tk = (
                    df.groupby("config_name")["total_input_tokens"]
                    .mean()
                    .rename("Avg Tokens")
                    .reset_index()
                    .set_index("config_name")
                )
                st.bar_chart(tk, color="#f87171")
                st.caption("Lower is better. Fewer tokens = less cost, faster inference, less context rot.")

        st.divider()

        col_c, col_d = st.columns(2)

        # ── Compression ratio ────────────────────────────────────────────────
        with col_c:
            st.markdown('<div class="section-title">Compression Ratio by Config</div>', unsafe_allow_html=True)
            if "compression_ratio" in df.columns:
                cr = (
                    df.groupby("config_name")["compression_ratio"]
                    .mean()
                    .rename("Compression Ratio")
                    .reset_index()
                    .set_index("config_name")
                )
                st.bar_chart(cr, color="#fbbf24")
                st.caption("Ratio of managed tokens to raw tokens. <1.0 means firewall is saving tokens.")

        # ── Latency ──────────────────────────────────────────────────────────
        with col_d:
            st.markdown('<div class="section-title">Avg Latency per Turn (ms)</div>', unsafe_allow_html=True)
            if "avg_latency_ms" in df.columns:
                lat = (
                    df.groupby("config_name")["avg_latency_ms"]
                    .mean()
                    .rename("Avg Latency (ms)")
                    .reset_index()
                    .set_index("config_name")
                )
                st.bar_chart(lat, color="#a78bfa")
                st.caption("Adaptive firewall adds ~100ms overhead per turn for scoring + compression.")

        st.divider()

        # ── Per-task success heatmap ─────────────────────────────────────────
        if "task_name" in df.columns and "config_name" in df.columns and "task_success" in df.columns:
            st.markdown('<div class="section-title">Per-Task Success Heatmap</div>', unsafe_allow_html=True)
            config_order = ["baseline", "simple_compaction", "fixed_relevance", "adaptive"]
            pivot = (
                df.pivot_table(index="task_name", columns="config_name", values="task_success", aggfunc="mean")
                .reindex(columns=[c for c in config_order if c in df["config_name"].unique()])
            )
            # Render as styled dataframe
            def _style_cell(v):
                if v >= 0.8:
                    return "background-color: rgba(74,222,128,0.25); color: #4ade80;"
                elif v >= 0.5:
                    return "background-color: rgba(251,191,36,0.2); color: #fbbf24;"
                else:
                    return "background-color: rgba(248,113,113,0.2); color: #f87171;"
            styled_pivot = pivot.style.applymap(_style_cell).format("{:.0%}")
            st.dataframe(styled_pivot, use_container_width=True)
            st.caption("Green ≥ 80% • Yellow ≥ 50% • Red < 50%")

    except ImportError:
        st.json(data[:5])


# ─── Demo / Simulation tab ────────────────────────────────────────────────────

def render_demo_tab():
    st.markdown("""
    <div class="demo-banner">
    🎭 This tab generates <strong>synthetic data</strong> so you can explore the dashboard
    without needing Ollama or a completed benchmark run.
    </div>
    """, unsafe_allow_html=True)

    col_s, col_b, _ = st.columns([1, 1, 3])
    with col_s:
        n_turns = st.slider("Simulated turns", 5, 40, 18, key="demo_turns")
    with col_b:
        seed = st.number_input("Random seed", 0, 999, 42, key="demo_seed")

    demo_turns  = _make_demo_turns(n_turns=int(n_turns), seed=int(seed))
    demo_bench  = _make_demo_benchmark()

    st.subheader("📡 Simulated Live Context Health")
    render_live_kpis(demo_turns)
    lc, rc = st.columns([2, 1])
    with lc:
        render_token_trend(demo_turns)
        render_relevance_histogram(demo_turns)
        render_latency_chart(demo_turns)
    with rc:
        render_subgoal(demo_turns)
        render_decision_log(demo_turns)

    st.divider()
    st.subheader("📊 Simulated Benchmark Results")
    render_benchmark(demo_bench, demo=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.toggle("Auto-refresh (5 s)", value=False, key="auto_refresh")

        if st.button("🗑️ Clear Firewall Log"):
            if LOG_FILE.exists():
                LOG_FILE.unlink()
            st.rerun()

        st.divider()
        st.markdown("#### 📁 Paths")
        st.code(str(LOG_FILE), language=None)

        st.markdown("#### 🔧 Config at a Glance")
        st.markdown(f"""
| Key | Value |
|---|---|
| Model | `{cfg['llm']['ollama_model']}` |
| Token budget | `{cfg['context']['token_budget']:,}` |
| High threshold | `{cfg['relevance']['high_threshold']}` |
| Low threshold | `{cfg['relevance']['low_threshold']}` |
| Budget pressure | `{cfg['context']['budget_pressure_ratio']}` |
| Embedding model | `{cfg['relevance']['embedding_model']}` |
""")

        st.divider()
        st.markdown("#### 🚀 Quick Commands")
        st.markdown("**Run agent (firewall mode):**")
        st.code("python -m agent.runner --mode firewall", language="bash")
        st.markdown("**Run agent (baseline):**")
        st.code("python -m agent.runner --mode baseline", language="bash")
        st.markdown("**Run full benchmark:**")
        st.code("python -m evaluation.benchmark", language="bash")
        st.markdown("**Run tests:**")
        st.code("pytest tests/ -v", language="bash")

        st.divider()
        st.markdown("#### 📦 Phase Status")
        st.markdown("""
- ✅ Phase 0 — Scaffold
- ✅ Phase 1 — Baseline Agent
- ✅ Phase 2 — Core Firewall
- ✅ Phase 3 — Policy + Storage
- ✅ Phase 4 — Retrieval + Assembler
- ✅ Phase 5 — Evaluation Suite
- 🔥 **Phase 6 — This Dashboard**
""")
    return auto_refresh


# ─── Main ─────────────────────────────────────────────────────────────────────

render_header()
auto_refresh = render_sidebar()

tab_live, tab_bench, tab_demo, tab_about = st.tabs([
    "🔴 Live Context Health",
    "📊 Benchmark Results",
    "🎭 Demo / Simulation",
    "ℹ️ About",
])

turns = load_live_log()

with tab_live:
    render_live_kpis(turns)
    lc, rc = st.columns([2, 1])
    with lc:
        render_token_trend(turns)
        render_relevance_histogram(turns)
        render_latency_chart(turns)
    with rc:
        render_subgoal(turns)
        render_decision_log(turns)

with tab_bench:
    bench_data = load_benchmark()
    render_benchmark(bench_data, demo=False)

with tab_demo:
    render_demo_tab()

with tab_about:
    st.markdown("""
## 🔥 Context Rot Firewall — How It Works

### The Problem
AI agents running long tasks accumulate **context rot**: as the context window fills, softmax
attention redistributes weight across all tokens, so early critical instructions get progressively
less model attention — even before the token limit is hit.

### The Solution
A lightweight Python middleware intercepting every LLM call:

```
User/Orchestrator
      │
      ▼
 Interceptor  ← wraps every call_llm()
      │
      ▼
   Chunker    ← assigns stable IDs to (message, tool_result) pairs
      │
      ▼
   Analyzer   ← token counts, sub-goal extraction, dependency detection
      │
      ▼
 Relevance Scorer (Two-Tier)
   ├─ Tier 1: sentence-transformers cosine similarity (all chunks)
   └─ Tier 2: LLM judge 0-10 (borderline chunks only)
      │
      ▼
 Adaptive Policy Engine
   ├─ score ≥ HIGH_THRESH  → KEEP verbatim
   ├─ score ≥ LOW_THRESH   → COMPRESS (LLM summary) + archive original
   └─ score < LOW_THRESH   → ARCHIVE to ChromaDB
      │
      ▼
   Compressor  ← LLM-generated summary of archived chunks
   Storage     ← ChromaDB vector index + flat-file archive
      │
      ▼
   Retrieval   ← embed sub-goal → query Chroma → inject top-k
      │
      ▼
   Assembler   ← builds final ordered context ≤ token budget
      │
      ▼
   LLM Call    ← only sees what matters right now
```

### Benchmark Configurations

| Config | Strategy | Expected Outcome |
|---|---|---|
| A — Baseline | No management | Context grows unchecked; coherence degrades |
| B — Simple Compaction | Truncate oldest at 75% budget | Avoids overflow; loses critical early info |
| C — Fixed Relevance | Embedding scores, fixed thresholds | Better relevance; can still exceed budget |
| D — Adaptive Firewall | Task-aware + budget-aware + retrieval | Best coherence and lowest token spend |

### Key Metrics

| Metric | Meaning |
|---|---|
| Task success rate | Did the agent follow all constraints to the end? |
| Critical info retention | Did protected chunks (constraints, requirements) survive? |
| Total input tokens | Lower = less cost + faster inference |
| Compression ratio | < 1.0 means the firewall saved tokens |
| Avg latency | Firewall adds ~100 ms overhead per turn |
| Retrieval accuracy | Fraction of retrieved chunks genuinely relevant (cosine ≤ 0.3) |
""")

if auto_refresh:
    time.sleep(5)
    st.rerun()
