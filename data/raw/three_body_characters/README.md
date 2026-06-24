# Raw Three-Body Character Data

This directory stores raw Three-Body source material used before persona
distillation and indexing.

## Files

- `three_body.txt`: local novel text imported from `C:/Users/MR/Downloads/三体.txt`.

## Intended Use

- Distill character-specific stage anchors and event memories.
- Build quote or scene retrieval indexes later.
- Cross-check generated temporal persona JSONL files against source text.

Keep raw source files separate from processed persona profiles so downstream
loaders can decide whether they need full-text retrieval, distilled RAG cards, or
both.
