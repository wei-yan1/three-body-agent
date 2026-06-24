# Raw Celebrity Reviews

This directory is reserved for raw celebrity, critic, or thinker commentary on
Three-Body characters.

## Planned Shape

Suggested future file types:

- `*.jsonl`: one review/commentary item per line.
- `*.md`: manually curated long-form notes.
- `sources.json`: source URL, author, date, and reliability metadata.

## RAG Use

When commentary is indexed, include at least:

- `commentator`
- `character`
- `timeline_stage`
- `source`
- `knowledge_type`

This lets the agent retrieve famous-person commentary without crossing a
character's stage knowledge boundary.
