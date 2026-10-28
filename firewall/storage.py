"""
Context Rot Firewall — Storage
Two-level storage:
  1. Flat files: archive/<chunk_id>.txt  — raw original text (source of truth)
  2. ChromaDB: vector index for semantic retrieval

Never deletes anything from flat files. ChromaDB is the index, not the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import chromadb

from .chunker import Chunk
from .config_loader import cfg

_COLLECTION_NAME = "context_rot_firewall"
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        persist_dir = cfg["storage"]["chroma_persist_dir"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def archive_chunk(chunk: Chunk, embedding: list[float] | None = None) -> None:
    """
    Archive a chunk to:
    - Flat file: archive/<chunk_id>.txt  (always)
    - ChromaDB: with embedding for retrieval (if embedding provided)

    Marks chunk.archived = True.
    """
    archive_dir = Path(cfg["storage"]["archive_dir"])
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Write full original content to flat file
    flat_path = archive_dir / f"{chunk.chunk_id}.txt"
    with open(flat_path, "w", encoding="utf-8") as f:
        f.write(chunk.content)

    # Write metadata sidecar
    meta_path = archive_dir / f"{chunk.chunk_id}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk.to_dict(), f, indent=2)

    # Add to ChromaDB
    coll = _get_collection()
    try:
        doc_meta = {
            "chunk_id": chunk.chunk_id,
            "role": chunk.role,
            "turn_index": chunk.turn_index,
            "summary": chunk.summary or "",
        }
        if embedding:
            coll.add(
                ids=[chunk.chunk_id],
                documents=[chunk.content],
                embeddings=[embedding],
                metadatas=[doc_meta],
            )
        else:
            coll.add(
                ids=[chunk.chunk_id],
                documents=[chunk.content],
                metadatas=[doc_meta],
            )
    except Exception as e:
        # ChromaDB may raise if chunk_id already exists
        if "already exists" not in str(e).lower():
            print(f"[Storage] ChromaDB add error for {chunk.chunk_id}: {e}")

    chunk.archived = True


def retrieve_chunks(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB by embedding similarity.

    Returns a list of dicts with keys: chunk_id, content, distance, metadata.
    Content is loaded from the flat file (ground truth).
    """
    coll = _get_collection()
    try:
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, coll.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[Storage] Retrieval error: {e}")
        return []

    if not results["ids"] or not results["ids"][0]:
        return []

    retrieved = []
    for i, chunk_id in enumerate(results["ids"][0]):
        flat_path = Path(cfg["storage"]["archive_dir"]) / f"{chunk_id}.txt"
        content = flat_path.read_text(encoding="utf-8") if flat_path.exists() else results["documents"][0][i]
        retrieved.append({
            "chunk_id": chunk_id,
            "content": content,
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return retrieved


def get_archive_count() -> int:
    """Return number of chunks in the vector store."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
