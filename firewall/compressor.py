"""
Context Rot Firewall — Compressor
Uses the local Ollama LLM to summarize a chunk into a short note.
The original chunk text is always archived to cold storage first.
"""
import httpx
from .chunker import Chunk
from .config_loader import cfg


def compress_chunk(chunk: Chunk) -> str:
    """
    Ask Ollama to summarize the chunk content into ≤3 sentences.
    Returns the summary string. Updates chunk.summary in place.
    Falls back to a truncated snippet on LLM error.
    """
    prompt = f"""Summarize the following context chunk into at most 3 concise sentences.
Preserve any key facts, file names, error messages, or decisions.
Do NOT add commentary. Output ONLY the summary.

CHUNK:
{chunk.content[:1200]}

SUMMARY:"""

    try:
        resp = httpx.post(
            f"{cfg['llm']['ollama_url']}/api/generate",
            json={
                "model": cfg["llm"]["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=cfg["llm"]["timeout_seconds"],
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        if summary:
            chunk.summary = summary
            return summary
    except Exception as e:
        print(f"[Compressor] LLM summarization failed: {e} — using excerpt")

    # Fallback: truncated excerpt
    fallback = chunk.content[:200].strip() + " [truncated]"
    chunk.summary = fallback
    return fallback
