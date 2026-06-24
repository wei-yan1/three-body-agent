# RAG Chunks

This directory contains auditable JSONL chunks generated before vector indexing.

## Files

- `persona_chunks.jsonl`: structure-aware chunks from temporal persona JSONL
  files. Each stage is split into semantic chunks:
  `persona_core`, `knowledge_boundary`, `relationships`, `novel_enrichment`,
  and `rag_card`.
- `novel_chunks.jsonl`: chapter-aware chunks from
  `data/raw/three_body_characters/three_body.txt`.

## Rebuild

```powershell
uv run python scripts\build_indexes.py chunks
```

## Why Keep These

These files make the RAG pipeline inspectable. You can verify chunk content and
metadata before sending data to Chroma or another vector store.
