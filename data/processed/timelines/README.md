# Processed Timelines

This directory stores normalized timeline anchors shared by all temporal persona
profiles.

## Files

- `global_timeline.json`: canonical `T0-T6` alignment rules. Luo Ji's character
  arc is the primary axis, and other characters should only generate stages when
  they have an active dialogue premise.

## Retrieval Rule

Dialogue and RAG retrieval should identify `timeline_stage` before loading a
character profile or source chunk. This is the main guardrail against characters
knowing future events too early.
