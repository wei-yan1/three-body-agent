"""Evidence provenance graph persistence for explainable StoryRole answers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.storage.postgres.client import postgres_connection
from app.storage.postgres.observability_schema import init_observability_schema


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EvidenceGraph:
    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.nodes: dict[str, int] = {}

    def add_node(
        self, node_type: str, source_id: str | None = None, *,
        content: str = "", metadata: dict[str, Any] | None = None,
    ) -> int:
        init_observability_schema()
        key = f"{node_type}:{source_id or _hash(content)}"
        if key in self.nodes:
            return self.nodes[key]
        with postgres_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO evidence_graph_nodes
                (trace_id,node_type,source_id,content_hash,content_preview,metadata)
                VALUES (%s,%s,%s,%s,%s,%s::jsonb) RETURNING id""",
                (self.trace_id, node_type, source_id, _hash(content) if content else None,
                 content[:1000], json.dumps(metadata or {}, ensure_ascii=False)),
            )
            node_id = int(cursor.fetchone()["id"])
        self.nodes[key] = node_id
        return node_id

    def add_edge(
        self, from_node: int, to_node: int, edge_type: str, *,
        weight: float | None = None, metadata: dict[str, Any] | None = None,
    ) -> None:
        init_observability_schema()
        with postgres_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO evidence_graph_edges
                (trace_id,from_node_id,to_node_id,edge_type,weight,metadata)
                VALUES (%s,%s,%s,%s,%s,%s::jsonb)""",
                (self.trace_id, from_node, to_node, edge_type, weight,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )

    def add_evidence(self, evidence: list[dict[str, Any]], *, target_node: int) -> list[int]:
        ids = []
        for item in evidence:
            source_id = str(item.get("chunk_id") or item.get("id") or "")
            node = self.add_node("novel_chunk", source_id, content=str(item.get("text") or item.get("content") or ""), metadata=item)
            self.add_edge(target_node, node, "supports", weight=float(item.get("score") or 0.0))
            ids.append(node)
        return ids
