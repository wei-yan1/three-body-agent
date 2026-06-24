# Processed Persona Profiles

This directory contains JSONL persona skills for temporal Three-Body dialogue
agents.

## Current Files

- `luoji_temporal_persona_skill.jsonl`: Luo Ji stages `T1-T6`.
- `yewenjie_temporal_persona_skill.jsonl`: Ye Wenjie stages `T0-T1`.
- `wangmiao_temporal_persona_skill.jsonl`: Wang Miao stage `T0`.
- `zhangbeihai_temporal_persona_skill.jsonl`: Zhang Beihai stages `T1`, `T2`,
  and `T4`.
- `manifest.json`: machine-readable inventory for ingestion scripts.

## JSONL Contract

- Line 1 is a `meta` record.
- Each following line is one `stage` record.
- Each stage has one `RAG知识卡片` object containing `card_id`, `content`, and
  `metadata`.
- `metadata.timeline_stage` should be used as the default RAG filter to prevent
  future-sight leakage.
