"""
Context Rot Firewall — Streamlit Dashboard

Displays live context health from the firewall log file.
Run with:  streamlit run dashboard/app.py
"""
import json
import time
from pathlib import Path
import streamlit as st
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from firewall.config_loader import cfg

LOG_FILE = Path(cfg["storage"]["log_dir"]) / "firewall_log.jsonl"

st.set_page_config(
    page_title="Context Rot Firewall Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0f1117; }
  .metric-card {
    background: linear-gradient(135deg, #1e2130, #252b3b);
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #7c9eff; }
  .metric-label { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }
  .decision-keep   { color: #4ade80; font-weight: 600; }
  .decision-compress { color: #fbbf24; font-weight: 600; }
  .decision-archive { color: #f87171; font-weight: 600; }
  .chunk-row { border-bottom: 1px solid #1e2130; padding: 0.4rem 0; }
  .header-title { font-size: 1.8rem; font-weight: 700; background: linear-gradient(90deg, #7c9eff, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)


def load_log() -> list[dict]:
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


def render_header():
    st.markdown('<div class="header-title">🔥 Context Rot Firewall</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892a4;margin-top:-0.5rem;">Real-time context health for long-running AI agents</p>', unsafe_allow_html=True)
    st.divider()


def render_metrics_row(turns: list[dict]):
    if not turns:
        st.info("No firewall data yet. Run the agent with `--mode firewall` to start.")
        return

    latest = turns[-1]
    asm = latest.get("assemble", {})
    pol = latest.get("policy", {})

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Current Turn", latest.get("turn", "—"))
    with c2:
        tokens = asm.get("final_tokens", 0)
        budget = asm.get("token_budget", cfg["context"]["token_budget"])
        st.metric("Tokens Used", f"{tokens:,}", delta=f"{asm.get('utilization_pct', 0)}% of budget")
    with c3:
        st.metric("Chunks Kept", pol.get("keep", 0), delta=None)
    with c4:
        st.metric("Compressed", pol.get("compress", 0))
    with c5:
        st.metric("Archived", pol.get("archive", 0))

    # Token budget bar
    util = asm.get("utilization_pct", 0) / 100
    color = "#4ade80" if util < 0.6 else "#fbbf24" if util < 0.85 else "#f87171"
    st.markdown(f"""
    <div style="margin:0.5rem 0 1rem 0;">
      <div style="font-size:0.75rem;color:#8892a4;margin-bottom:4px;">TOKEN BUDGET USAGE</div>
      <div style="background:#1e2130;border-radius:8px;height:18px;overflow:hidden;">
        <div style="width:{min(util*100, 100):.1f}%;height:100%;background:{color};border-radius:8px;transition:width 0.5s;"></div>
      </div>
      <div style="font-size:0.7rem;color:#8892a4;margin-top:3px;">{asm.get('final_tokens',0):,} / {budget:,} tokens</div>
    </div>
    """, unsafe_allow_html=True)


def render_relevance_chart(turns: list[dict]):
    if not turns:
        return
    import json

    # Collect chunk score data across turns for a scatter plot
    rows = []
    for t in turns:
        for cs in t.get("chunk_scores", []):
            rows.append({
                "Turn": t["turn"],
                "Score": cs["score"],
                "Decision": cs["decision"],
                "Role": cs["role"],
                "Protected": cs["protected"],
            })

    if not rows:
        return

    st.subheader("Relevance Score Distribution")

    # Simple score histogram using streamlit native bar_chart
    import collections
    buckets = collections.Counter()
    for r in rows:
        bucket = round(r["Score"] * 10) / 10
        buckets[bucket] += 1

    chart_data = {
        "Score": sorted(buckets.keys()),
        "Count": [buckets[k] for k in sorted(buckets.keys())],
    }

    try:
        import pandas as pd
        df = pd.DataFrame({"Score": chart_data["Score"], "Count": chart_data["Count"]})
        df = df.set_index("Score")
        st.bar_chart(df, color="#7c9eff")
    except ImportError:
        st.write(chart_data)


def render_token_trend(turns: list[dict]):
    if len(turns) < 2:
        return

    st.subheader("Token Usage Over Turns")
    try:
        import pandas as pd
        data = {
            "Turn": [t["turn"] for t in turns],
            "Managed Tokens": [t.get("assemble", {}).get("final_tokens", 0) for t in turns],
            "Original Messages": [t.get("original_message_count", 0) for t in turns],
        }
        df = pd.DataFrame(data).set_index("Turn")
        st.line_chart(df, color=["#7c9eff", "#f87171"])
    except ImportError:
        st.write("Install pandas for charts: pip install pandas")


def render_decision_log(turns: list[dict]):
    if not turns:
        return

    st.subheader("Latest Turn — Chunk Decisions")
    latest = turns[-1]

    for cs in latest.get("chunk_scores", []):
        d = cs["decision"]
        color_map = {"keep": "🟢", "compress": "🟡", "archive": "🔴", "archived": "⚫"}
        icon = color_map.get(d, "⚪")
        protected_tag = " 🔒" if cs["protected"] else ""
        st.markdown(
            f"`{cs['id']}` {icon} **{d}**{protected_tag} — "
            f"role: `{cs['role']}` score: `{cs['score']:.3f}`"
        )


def render_subgoal(turns: list[dict]):
    if not turns:
        return
    latest = turns[-1]
    subgoal = latest.get("analysis", {}).get("current_subgoal", "")
    if subgoal:
        st.subheader("Current Sub-Goal Detected")
        st.info(subgoal[:400])


def render_latency(turns: list[dict]):
    if len(turns) < 2:
        return
    st.subheader("Firewall Latency per Turn (ms)")
    try:
        import pandas as pd
        data = {"Turn": [t["turn"] for t in turns], "Latency (ms)": [t.get("latency_ms", 0) for t in turns]}
        df = pd.DataFrame(data).set_index("Turn")
        st.area_chart(df, color="#a78bfa")
    except ImportError:
        pass


# ─── Benchmark Results helpers ────────────────────────────────────────────────

BENCHMARK_FILE = Path(cfg["storage"]["log_dir"]) / "benchmark" / "benchmark_results.json"


def load_benchmark() -> list[dict]:
    if not BENCHMARK_FILE.exists():
        return []
    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def render_benchmark(data: list[dict]):
    if not data:
        st.info(
            "No benchmark results yet. Run:\n"
            "```\npython -m evaluation.benchmark\n```"
        )
        return

    st.subheader("Benchmark Results — 4 Configs × 3 Tasks")

    try:
        import pandas as pd

        df = pd.DataFrame(data)
        if "error" in df.columns:
            df = df[df["error"].isna()]

        if df.empty:
            st.warning("All benchmark runs errored out.")
            return

        # Summary table
        display_cols = [
            "task_name", "config_name", "task_success",
            "total_input_tokens", "compression_ratio", "avg_latency_ms",
        ]
        existing_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[existing_cols].rename(columns={
                "task_name": "Task",
                "config_name": "Config",
                "task_success": "Success",
                "total_input_tokens": "Input Tokens",
                "compression_ratio": "Ratio",
                "avg_latency_ms": "Avg Latency (ms)",
            }),
            use_container_width=True,
        )

        # Success rate by config
        if "task_success" in df.columns and "config_name" in df.columns:
            st.subheader("Task Success Rate by Config")
            success_by_config = (
                df.groupby("config_name")["task_success"]
                .mean()
                .rename("Success Rate")
                .reset_index()
                .set_index("config_name")
            )
            st.bar_chart(success_by_config, color="#4ade80")

        # Token usage by config
        if "total_input_tokens" in df.columns and "config_name" in df.columns:
            st.subheader("Avg Input Tokens by Config")
            tokens_by_config = (
                df.groupby("config_name")["total_input_tokens"]
                .mean()
                .rename("Avg Input Tokens")
                .reset_index()
                .set_index("config_name")
            )
            st.bar_chart(tokens_by_config, color="#f87171")

    except ImportError:
        st.json(data)


# ─── Main App ─────────────────────────────────────────────────────────────────

render_header()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    if st.button("🗑️ Clear Firewall Log"):
        if LOG_FILE.exists():
            LOG_FILE.unlink()
        st.rerun()
    st.divider()
    st.markdown(f"**Log file:** `{LOG_FILE}`")
    st.markdown(f"**Model:** `{cfg['llm']['ollama_model']}`")
    st.markdown(f"**Budget:** `{cfg['context']['token_budget']} tokens`")
    st.markdown(f"**High threshold:** `{cfg['relevance']['high_threshold']}`")
    st.markdown(f"**Low threshold:** `{cfg['relevance']['low_threshold']}`")
    st.divider()
    st.markdown("**Run agent:**")
    st.code("python -m agent.runner --mode firewall", language="bash")
    st.markdown("**Run benchmark:**")
    st.code("python -m evaluation.benchmark", language="bash")
    st.markdown("**Run tests:**")
    st.code("pytest tests/ -v", language="bash")

# Tabbed layout
tab_live, tab_benchmark = st.tabs(["🔴 Live Context Health", "📊 Benchmark Results"])

turns = load_log()

with tab_live:
    render_metrics_row(turns)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_token_trend(turns)
        render_relevance_chart(turns)
        render_latency(turns)

    with col_right:
        render_subgoal(turns)
        render_decision_log(turns)

with tab_benchmark:
    benchmark_data = load_benchmark()
    render_benchmark(benchmark_data)

if auto_refresh:
    time.sleep(5)
    st.rerun()

