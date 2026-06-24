# Retrieval Indexes

Generated retrieval artifacts belong here.

The directory is ignored by Git except for this README, because vector indexes
and cache files can be large and reproducible.

Recommended index groups:

- `persona_profiles`: distilled stage cards from `data/processed/persona_profiles`.
- `novel_chunks`: audited chunks from
  `data/processed/rag_chunks/novel_chunks.jsonl`, originally derived from
  `data/raw/three_body_characters/three_body.txt`.
- `celebrity_reviews`: external commentary on characters.

Each index should preserve `character`, `timeline_stage`, `card_id` or
`source_id`, and `knowledge_type` metadata for filtering.

## Current Chroma Index

The current local Chroma index is stored in `data/indexes/chroma`.

Collections:

- `three_body_persona_profiles`: structure-aware chunks derived from persona
  JSONL files.
- `three_body_novel_chunks`: timeline-aware novel chunks imported from
  `data/processed/rag_chunks/novel_chunks.jsonl`.

Embedding provider:

- DashScope/Bailian via `langchain_community.embeddings.DashScopeEmbeddings`.
- Default coarse retrieval embedding model: `text-embedding-v2`.
- Cross-Encoder rerank model: `qwen3-vl-rerank`.
- The Luo Ji Agent uses multi-route recall, Min-Max weighted fusion, then
  `qwen3-vl-rerank` for final reranking.
- If your Bailian console exposes these models under custom deployment names, set
  `DASHSCOPE_DOCUMENT_EMBEDDING_MODEL` and `DASHSCOPE_QUERY_EMBEDDING_MODEL`.
- `DASHSCOPE_API_KEY` must be available in the environment or `.env`.

Rebuild commands:

```powershell
cd D:\AgentCode\two
uv run python scripts\build_indexes.py chunks
uv run python scripts\build_indexes.py novel-async-input
uv run python scripts\build_indexes.py novel-async-submit --input-url <OSS_OR_HTTP_URL> --wait
uv run python scripts\build_indexes.py novel-async-download --result-url <DASHSCOPE_RESULT_URL>
uv run python scripts\build_indexes.py novel-async-import
uv run python scripts\build_indexes.py novel-chroma
uv run python scripts\build_indexes.py chroma
uv run python scripts\build_indexes.py all
```

Use `novel-chroma` when you want to rebuild the novel vector collection from
`data/processed/rag_chunks/novel_chunks.jsonl` with the configured coarse
embedding model. The `novel-async-*` commands are kept for DashScope batch text
embedding models.

If your IDE tries to run `uv` from another project directory, set its project
root/interpreter working directory to `D:\AgentCode\two`. The commands above are
intended to run from this repository root.
