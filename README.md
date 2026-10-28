# 🔥 Context Rot Firewall

> A model-agnostic Python middleware library that prevents **context rot** in long-running AI agents.

---

## What is Context Rot?

As an AI agent works through a long task, its context window fills with tool outputs, file contents, old decisions, and stale intermediate results. This is **not** context overflow (hitting the token limit) — it happens well before the limit.

Because attention weights are softmax-normalised across the entire context, each token gets a proportionally **smaller share of attention** as context grows. Practitioners report visible coherence loss after roughly 20–30 turns.

---

## Solution — The Context Rot Firewall

A plug-and-play Python middleware that intercepts the context an AI agent sends to the LLM on every turn and actively decides what **stays**, what gets **compressed**, and what moves to **retrievable cold storage** — all based on relevance to the *current sub-goal*.

```
Context Interceptor → Chunker → Analyzer → Relevance Scorer → Policy Engine
                                                                    ↓
                                          Assembler ← Compressor + Storage/Retrieval
                                              ↓
                                           LLM Call
```

---

## Architecture

| # | Layer | File | Responsibility |
|---|---|---|---|
| 1 | Agent Loop | `agent/runner.py` | Plans task, calls tools, loops |
| 2 | Tools | `agent/tools.py` | File R/W, shell exec, search |
| 3 | Interceptor | `firewall/interceptor.py` | Wraps every `call_llm()` call |
| 4 | Chunker | `firewall/chunker.py` | Assigns stable IDs to (message, tool_result) pairs |
| 5 | Analyzer | `firewall/analyzer.py` | Token counts, sub-goal extraction, dependency detection |
| 6 | Relevance | `firewall/relevance.py` | Two-tier scorer: embedding → LLM for borderline chunks |
| 7 | Policy | `firewall/adaptive_policy.py` | Rule-based keep/compress/archive/retrieve + budget awareness |
| 8 | Compressor | `firewall/compressor.py` | LLM summarizes below-threshold chunks |
| 9 | Storage | `firewall/storage.py` | ChromaDB vector index + raw text archive |
| 10 | Retrieval | `firewall/retrieval.py` | Embeds sub-goal → queries Chroma → injects top-k back |
| 11 | Assembler | `firewall/assembler.py` | Builds final ordered context ≤ token budget |
| 12 | Dashboard | `dashboard/app.py` | Streamlit: token bar, relevance chart, eviction log, benchmark |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
ollama serve
ollama pull llama3.2:3b
```

### 3. Run the agent

```bash
# With Context Rot Firewall (recommended)
python -m agent.runner --task "Write a Python script that analyses a CSV" --mode firewall

# Baseline — no context management
python -m agent.runner --task "Write a Python script that analyses a CSV" --mode baseline
```

### 4. Open the live dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run the full benchmark suite
```bash
python -m evaluation.benchmark
```

### 6. Run unit tests
```bash
pytest tests/ -v
```

---

## Benchmark Configurations

| Config | Strategy |
|---|---|
| **A — Baseline** | Raw agent, no context management |
| **B — Simple compaction** | Truncate oldest chunks when > 75% budget |
| **C — Fixed relevance** | Embedding scores with fixed thresholds, no budget awareness |
| **D — Adaptive Firewall** | Full system: task-aware + budget-aware + retrieval |

### Test Scenarios

1. **Forgotten Requirement** — Insert a constraint early; 20 unrelated tool calls later, does the agent still honour it?
2. **Recover Earlier Discovery** — Critical info found at turn 2; 30+ intervening turns; does agent use it at turn 50?
3. **Multi-file Coding Task** — Edit 5 files, run tests, fix failures across 40+ turns. Measure correct edits / test pass rate.

---

## Configuration (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `token_budget` | `4096` | Max tokens sent to LLM per turn |
| `high_threshold` | `0.75` | Score above this → KEEP |
| `low_threshold` | `0.40` | Score below this → ARCHIVE |
| `budget_pressure_ratio` | `0.80` | Start downgrading KEEPs at this fraction of budget |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `ollama_model` | `llama3.2:3b` | Local LLM via Ollama |

---

## Project Structure

```
context-rot-firewall/
├── agent/              # Baseline agent loop + tools
│   ├── runner.py
│   └── tools.py
├── firewall/           # Core middleware modules
│   ├── interceptor.py
│   ├── chunker.py
│   ├── analyzer.py
│   ├── relevance.py
│   ├── adaptive_policy.py
│   ├── compressor.py
│   ├── storage.py
│   ├── retrieval.py
│   └── assembler.py
├── evaluation/         # Benchmark suite
│   ├── tasks.py
│   ├── benchmark.py
│   └── metrics.py
├── dashboard/          # Streamlit live dashboard
│   └── app.py
├── tests/              # pytest test suite
│   └── test_all.py
├── config.yaml
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Local LLM | Ollama (`llama3.2:3b` / `qwen2.5:7b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | `chromadb` (local, no server needed) |
| Token counting | `tiktoken` |
| Dashboard | `streamlit` |
| Testing | `pytest` |

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold & config | ✅ Done |
| 1 | Baseline agent loop | ✅ Done |
| 2 | Core Firewall (Interceptor → Relevance Scorer) | ✅ Done |
| 3 | Policy Engine + Compressor + Storage | ✅ Done |
| 4 | Retrieval + Assembler | ✅ Done |
| 5 | Evaluation benchmark suite | ✅ Done |
| 6 | Enhanced Streamlit Dashboard | 🚧 In Progress |

---

*Built as a research prototype to demonstrate within-session, turn-by-turn context triage for AI agents.*
