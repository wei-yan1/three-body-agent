"""Durable tracing, model usage accounting, and provider-call helpers."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Iterator

from app.storage.postgres.client import postgres_connection


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _now() -> datetime:
    return datetime.now(UTC)


def _model_name(model: Any) -> str:
    return str(getattr(model, "model_name", None) or getattr(model, "model", None) or getattr(model, "model_id", None) or os.getenv("TEST_CHAT_MODEL", "unknown"))


def _provider_name(model: Any) -> str:
    return str(getattr(model, "model_provider", None) or getattr(model, "provider", None) or "openai-compatible")


def _agent_name(operation: str) -> str:
    """Map an internal operation to the user-facing Agent that performed it."""
    if operation.startswith("a2a:"):
        return operation[4:]
    return {
        "nuwa_profile": "nuwa-profiler",
        "period_analysis": "character-period-analysis",
        "storyrole_answer_generation": "character-conversation",
        "character_reasoning": "character-reasoning",
        "role_cognition": "role-cognition",
        "consistency_guard": "consistency-guard",
        "evidence_analysis": "evidence-analysis",
        "deep_question_planning": "deep-question-planner",
        "embedding": "embedding",
        "rerank": "reranker",
    }.get(operation, operation)


def model_usage_from_response(response: Any) -> dict[str, int | None]:
    """Normalize common LangChain/OpenAI/DashScope usage shapes."""
    metadata = getattr(response, "response_metadata", None) or {}
    additional = getattr(response, "additional_kwargs", None) or {}
    usage: Any = getattr(response, "usage_metadata", None) or metadata.get("token_usage") or metadata.get("usage") or metadata.get("usage_metadata") or additional.get("usage")
    if usage is None and isinstance(response, dict):
        response_metadata = response.get("response_metadata") or {}
        additional_kwargs = response.get("additional_kwargs") or {}
        usage = (response.get("usage") or response.get("usage_metadata")
                 or response_metadata.get("token_usage") or response_metadata.get("usage")
                 or additional_kwargs.get("usage"))
    usage = usage or {}
    if isinstance(usage, dict):
        usage = usage.get("token_usage") or usage.get("usage") or usage

    def pick(*names: str) -> int | None:
        for name in names:
            value: Any = usage
            for part in name.split("."):
                value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
                if value is None:
                    break
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    input_tokens = pick("input_tokens", "prompt_tokens")
    output_tokens = pick("output_tokens", "completion_tokens")
    cached_tokens = pick("cache_read_input_tokens", "cache_read_tokens", "cached_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens")
    total_tokens = pick("total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def model_cost(model: str, usage: dict[str, int | None]) -> float | None:
    """Calculate cost from MODEL_PRICING_JSON without hard-coding vendor prices."""
    raw = os.getenv("MODEL_PRICING_JSON", "")
    if not raw:
        return None
    try:
        pricing = json.loads(raw)
        item = pricing.get(model) or pricing.get("*")
        if not isinstance(item, dict):
            return None
        if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
            return None
        return round(
            (usage.get("input_tokens", 0) or 0) / 1_000_000 * float(item.get("input_per_million", 0))
            + (usage.get("output_tokens", 0) or 0) / 1_000_000 * float(item.get("output_per_million", 0)),
            8,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_type: str = "request"
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        try:
            with postgres_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO pipeline_runs (trace_id,run_type,status,metadata,started_at)
                    VALUES (%s,%s,'running',%s::jsonb,%s)
                    ON CONFLICT (trace_id) DO NOTHING""",
                    (self.trace_id, self.run_type, json.dumps(self.metadata, ensure_ascii=False), _now()),
                )
        except Exception:
            return

    def finish(self, *, status: str = "completed", error: str | None = None) -> None:
        try:
            with postgres_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """UPDATE pipeline_runs r SET status=%s,error=%s,ended_at=%s,
                    duration_ms=EXTRACT(EPOCH FROM (%s - r.started_at))*1000,
                    input_tokens=(SELECT COALESCE(SUM(input_tokens),0) FROM model_invocations WHERE trace_id=%s),
                    output_tokens=(SELECT COALESCE(SUM(output_tokens),0) FROM model_invocations WHERE trace_id=%s),
                    total_tokens=(SELECT COALESCE(SUM(total_tokens),0) FROM model_invocations WHERE trace_id=%s),
                    model_call_count=(SELECT COUNT(*) FROM model_invocations WHERE trace_id=%s),
                    usage_complete=COALESCE((SELECT BOOL_AND(total_tokens IS NOT NULL) FROM model_invocations WHERE trace_id=%s), FALSE)
                    WHERE r.trace_id=%s""",
                    (status, error, _now(), _now(), self.trace_id, self.trace_id, self.trace_id, self.trace_id, self.trace_id, self.trace_id),
                )
        except Exception:
            return

    @contextmanager
    def step(self, name: str, *, metadata: dict[str, Any] | None = None, parent_step_id: int | None = None) -> Iterator[int | None]:
        started = time.perf_counter()
        step_id: int | None = None
        try:
            try:
                with postgres_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO pipeline_steps
                        (trace_id,parent_step_id,step_name,status,metadata,started_at)
                        VALUES (%s,%s,%s,'running',%s::jsonb,%s) RETURNING id""",
                        (self.trace_id, parent_step_id, name, json.dumps(metadata or {}, ensure_ascii=False), _now()),
                    )
                    row = cur.fetchone()
                    step_id = int(row["id"] if isinstance(row, dict) else row[0])
            except Exception:
                step_id = None
            yield step_id
        except Exception as exc:
            self._finish_step(step_id, "failed", time.perf_counter() - started, str(exc))
            raise
        else:
            self._finish_step(step_id, "completed", time.perf_counter() - started, None)

    def _finish_step(self, step_id: int | None, status: str, elapsed: float, error: str | None) -> None:
        if step_id is None:
            return
        try:
            with postgres_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE pipeline_steps SET status=%s,error=%s,duration_ms=%s,ended_at=%s WHERE id=%s",
                    (status, error, round(elapsed * 1000, 3), _now(), step_id),
                )
        except Exception:
            return


def record_model_invocation(*, trace_id: str | None, operation: str, provider: str, model: str, started_at: datetime, duration_ms: float, usage: dict[str, int | None] | None = None, cost: float | None = None, cache_hit: bool = False, prompt_hash: str | None = None, status: str = "completed", error: str | None = None, metadata: dict[str, Any] | None = None, call_count: int = 1, remote_call_count: int | None = None, cache_hit_count: int | None = None, agent_id: str | None = None) -> None:
    usage = usage or {}
    remote_call_count = (0 if cache_hit else call_count) if remote_call_count is None else remote_call_count
    cache_hit_count = (call_count if cache_hit else 0) if cache_hit_count is None else cache_hit_count
    agent_id = agent_id or _agent_name(operation)
    try:
        with postgres_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_invocations
                (trace_id,owner_id,agent_id,operation,provider,model,status,duration_ms,input_tokens,output_tokens,cached_tokens,total_tokens,cost,cache_hit,prompt_hash,metadata,started_at,ended_at,error)
                VALUES (%s,(SELECT (metadata->>'owner_id')::bigint FROM pipeline_runs WHERE trace_id=%s),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)""",
                (trace_id, trace_id, agent_id, operation, provider, model, status, duration_ms, usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cached_tokens"), usage.get("total_tokens"), cost, cache_hit, prompt_hash, json.dumps(metadata or {}, ensure_ascii=False), started_at, _now(), error),
            )
            cur.execute(
                """INSERT INTO model_usage_daily
                (usage_date,provider,model,operation,call_count,remote_call_count,cache_hits,input_tokens,output_tokens,cached_tokens,total_tokens,total_cost,error_count)
                VALUES (CURRENT_DATE,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (usage_date,provider,model,operation) DO UPDATE SET
                call_count=model_usage_daily.call_count+EXCLUDED.call_count,
                remote_call_count=model_usage_daily.remote_call_count+EXCLUDED.remote_call_count,
                cache_hits=model_usage_daily.cache_hits+EXCLUDED.cache_hits,
                input_tokens=model_usage_daily.input_tokens+EXCLUDED.input_tokens,
                output_tokens=model_usage_daily.output_tokens+EXCLUDED.output_tokens,
                cached_tokens=model_usage_daily.cached_tokens+EXCLUDED.cached_tokens,
                total_tokens=model_usage_daily.total_tokens+EXCLUDED.total_tokens,
                total_cost=model_usage_daily.total_cost+EXCLUDED.total_cost,
                error_count=model_usage_daily.error_count+EXCLUDED.error_count""",
                (provider, model, operation, call_count, remote_call_count, cache_hit_count, usage.get("input_tokens") or 0, usage.get("output_tokens") or 0, usage.get("cached_tokens") or 0, usage.get("total_tokens") or 0, cost or 0, 1 if status == "failed" else 0),
            )
    except Exception:
        return


def observe_sync_call(call: Callable[[], Any], *, operation: str, trace_id: str | None = None, model: Any = None, prompt: str | None = None, provider: str | None = None) -> Any:
    started = _now(); clock = time.perf_counter()
    selected_model = _model_name(model) if model is not None else os.getenv("TEST_CHAT_MODEL", "unknown")
    selected_provider = provider or (_provider_name(model) if model is not None else "openai-compatible")
    try:
        result = call()
        usage = model_usage_from_response(result)
        record_model_invocation(trace_id=trace_id, operation=operation, provider=selected_provider, model=selected_model, started_at=started, duration_ms=(time.perf_counter() - clock) * 1000, usage=usage, cost=model_cost(selected_model, usage), prompt_hash=sha256_text(prompt) if prompt else None)
        return result
    except Exception as exc:
        record_model_invocation(trace_id=trace_id, operation=operation, provider=selected_provider, model=selected_model, started_at=started, duration_ms=(time.perf_counter() - clock) * 1000, status="failed", error=str(exc), prompt_hash=sha256_text(prompt) if prompt else None)
        raise


async def observe_async_call(call: Callable[[], Awaitable[Any]], *, operation: str, trace_id: str | None = None, model: Any = None, prompt: str | None = None, provider: str | None = None) -> Any:
    started = _now(); clock = time.perf_counter()
    selected_model = _model_name(model) if model is not None else os.getenv("STORYROLE_CHAT_MODEL", os.getenv("TEST_CHAT_MODEL", "unknown"))
    selected_provider = provider or (_provider_name(model) if model is not None else "openai-compatible")
    try:
        result = await call()
        usage = model_usage_from_response(result if not isinstance(result, dict) else result.get("messages", [result])[-1])
        record_model_invocation(trace_id=trace_id, operation=operation, provider=selected_provider, model=selected_model, started_at=started, duration_ms=(time.perf_counter() - clock) * 1000, usage=usage, cost=model_cost(selected_model, usage), prompt_hash=sha256_text(prompt) if prompt else None)
        return result
    except Exception as exc:
        record_model_invocation(trace_id=trace_id, operation=operation, provider=selected_provider, model=selected_model, started_at=started, duration_ms=(time.perf_counter() - clock) * 1000, status="failed", error=str(exc), prompt_hash=sha256_text(prompt) if prompt else None)
        raise


def usage_summary(filters: dict[str, str | None]) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if filters.get("owner_id"):
        clauses.append("owner_id=%s")
        params.append(int(filters["owner_id"]))
    for key, column in (("provider", "provider"), ("model", "model"), ("operation", "operation")):
        if filters.get(key):
            clauses.append(f"{column}=%s")
            params.append(filters[key])
    if filters.get("date_from"):
        clauses.append("started_at >= %s::date")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        clauses.append("started_at < (%s::date + INTERVAL '1 day')")
        params.append(filters["date_to"])
    where = " AND ".join(clauses)
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT provider, model, operation, COUNT(*) calls,
            COUNT(*) FILTER (WHERE NOT cache_hit) remote_calls,
            COUNT(*) FILTER (WHERE cache_hit) cache_hits,
            COALESCE(SUM(input_tokens), 0) input_tokens,
            COALESCE(SUM(output_tokens), 0) output_tokens,
            COALESCE(SUM(cached_tokens), 0) cached_tokens,
            COALESCE(SUM(total_tokens), 0) total_tokens,
            COALESCE(SUM(cost), 0) total_cost,
            COUNT(*) FILTER (WHERE status='failed') errors,
            AVG(duration_ms) avg_latency_ms,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) p50_latency_ms,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) p95_latency_ms
            FROM model_invocations WHERE {where}
            GROUP BY provider, model, operation ORDER BY total_cost DESC NULLS LAST""", params)
        rows = list(cur.fetchall())
    for row in rows:
        row["cache_hit_rate"] = (row["cache_hits"] / row["calls"]) if row["calls"] else 0.0
    return rows


def usage_dashboard(owner_id: int, *, days: int = 14) -> dict[str, Any]:
    """Return one bounded, frontend-friendly usage snapshot."""
    days = max(1, min(int(days), 90))
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) calls, COUNT(*) FILTER (WHERE status='failed') errors,
                COUNT(*) FILTER (WHERE cache_hit) cache_hits,
                COALESCE(SUM(input_tokens),0) input_tokens,
                COALESCE(SUM(output_tokens),0) output_tokens,
                COALESCE(SUM(cached_tokens),0) cached_tokens,
                COALESCE(SUM(total_tokens),0) total_tokens,
                COALESCE(SUM(cost),0) total_cost,
                COALESCE(AVG(duration_ms),0) avg_latency_ms,
                COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms),0) p50_latency_ms,
                COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms),0) p95_latency_ms
                FROM model_invocations WHERE owner_id=%s AND started_at >= CURRENT_DATE - (%s * INTERVAL '1 day')""",
            (owner_id, days),
        )
        totals = cur.fetchone() or {}
        cur.execute(
            """SELECT id, trace_id, COALESCE(agent_id, operation) agent_id, operation, provider, model, status, duration_ms,
                input_tokens, output_tokens, cached_tokens, total_tokens, cost, cache_hit,
                started_at, error FROM model_invocations
                WHERE owner_id=%s ORDER BY started_at DESC, id DESC LIMIT 50""",
            (owner_id,),
        )
        recent = list(cur.fetchall())
        cur.execute(
            """SELECT COALESCE(agent_id, operation) agent_id, COUNT(*) calls,
                COUNT(*) FILTER (WHERE status='failed') errors,
                COALESCE(SUM(total_tokens),0) total_tokens,
                COALESCE(SUM(cost),0) total_cost,
                COALESCE(AVG(duration_ms),0) avg_latency_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) p95_latency_ms
                FROM model_invocations
                WHERE owner_id=%s AND started_at >= CURRENT_DATE - (%s * INTERVAL '1 day')
                GROUP BY COALESCE(agent_id, operation) ORDER BY calls DESC""",
            (owner_id, days),
        )
        agents = list(cur.fetchall())
        cur.execute(
            """SELECT ps.trace_id, ps.step_name, ps.status, ps.duration_ms,
                ps.started_at, ps.ended_at, pr.run_type, ps.id, ps.parent_step_id
                FROM pipeline_steps ps JOIN pipeline_runs pr ON pr.trace_id=ps.trace_id
                WHERE (pr.metadata->>'owner_id')::bigint=%s
                  AND ps.started_at >= CURRENT_DATE - (%s * INTERVAL '1 day')
                ORDER BY ps.started_at DESC, ps.id DESC LIMIT 100""",
            (owner_id, days),
        )
        pipeline_steps = list(cur.fetchall())
        cur.execute(
            """SELECT step_name, COUNT(*) calls,
                COUNT(*) FILTER (WHERE ps.status='failed') errors,
                COALESCE(AVG(ps.duration_ms), 0) avg_duration_ms,
                COALESCE(MAX(ps.duration_ms), 0) max_duration_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY ps.duration_ms) p95_duration_ms
                FROM pipeline_steps ps JOIN pipeline_runs pr ON pr.trace_id=ps.trace_id
                WHERE (pr.metadata->>'owner_id')::bigint=%s
                  AND ps.started_at >= CURRENT_DATE - (%s * INTERVAL '1 day')
                GROUP BY step_name ORDER BY avg_duration_ms DESC""",
            (owner_id, days),
        )
        pipeline_stage_summary = list(cur.fetchall())
        cur.execute(
            """SELECT trace_id, run_type, status, duration_ms, started_at, ended_at,
                input_tokens, output_tokens, total_tokens, model_call_count, error
                FROM pipeline_runs
                WHERE (metadata->>'owner_id')::bigint=%s
                  AND started_at >= CURRENT_DATE - (%s * INTERVAL '1 day')
                ORDER BY started_at DESC LIMIT 30""",
            (owner_id, days),
        )
        pipeline_runs = list(cur.fetchall())
        cur.execute(
            """SELECT usage_date, SUM(call_count) calls, SUM(remote_call_count) remote_calls,
                SUM(cache_hits) cache_hits, SUM(total_tokens) total_tokens,
                SUM(total_cost) total_cost, SUM(error_count) errors
                FROM model_usage_daily
                WHERE usage_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
                GROUP BY usage_date ORDER BY usage_date""",
            (days,),
        )
        daily = list(cur.fetchall())
        cur.execute(
            """SELECT COUNT(*) entries, COALESCE(SUM(dimension),0) dimensions
                FROM embedding_cache""",
        )
        cache = cur.fetchone() or {}
    for row in [totals, *recent, *agents, *pipeline_steps, *pipeline_stage_summary, *pipeline_runs, *daily, cache]:
        if isinstance(row, dict):
            for key, value in list(row.items()):
                if hasattr(value, "isoformat"):
                    row[key] = value.isoformat()
    totals["cache_hit_rate"] = (float(totals.get("cache_hits", 0)) / int(totals.get("calls", 0))) if totals.get("calls") else 0.0
    return {
        "days": days,
        "totals": totals,
        "recent": recent,
        "agents": agents,
        "pipeline_steps": pipeline_steps,
        "pipeline_stage_summary": pipeline_stage_summary,
        "pipeline_runs": pipeline_runs,
        "daily": daily,
        "embedding_cache": cache,
    }


def trace_details(trace_id: str, *, owner_id: int | None = None) -> dict[str, Any] | None:
    with postgres_connection() as conn, conn.cursor() as cur:
        if owner_id is None:
            cur.execute("SELECT * FROM pipeline_runs WHERE trace_id=%s", (trace_id,))
        else:
            cur.execute("SELECT * FROM pipeline_runs WHERE trace_id=%s AND (metadata->>'owner_id')::bigint=%s", (trace_id, owner_id))
        run = cur.fetchone()
        if not run: return None
        cur.execute("SELECT * FROM pipeline_steps WHERE trace_id=%s ORDER BY started_at,id", (trace_id,))
        steps = list(cur.fetchall())
        cur.execute("SELECT * FROM model_invocations WHERE trace_id=%s ORDER BY started_at,id", (trace_id,))
        calls = list(cur.fetchall())
    return {"run": run, "steps": steps, "model_invocations": calls}


def trace_report(trace_id: str, *, owner_id: int | None = None) -> dict[str, Any] | None:
    """Return the exact agents, model calls, and stages for one API request."""
    details = trace_details(trace_id, owner_id=owner_id)
    if not details:
        return None
    run = details["run"] or {}
    calls = details["model_invocations"]
    agents: dict[str, dict[str, Any]] = {}
    for step in details["steps"]:
        step_name = str(step.get("step_name") or "")
        if not step_name.startswith("a2a:"):
            continue
        agent_id = step_name[4:]
        item = agents.setdefault(agent_id, {
            "agent_id": agent_id, "calls": 0, "total_tokens": 0,
            "input_tokens": 0, "output_tokens": 0, "duration_ms": 0.0,
            "model_duration_ms": 0.0, "errors": 0,
        })
        item["duration_ms"] += float(step.get("duration_ms") or 0)
        item["errors"] += int(step.get("status") == "failed")
    for call in calls:
        agent_id = str(call.get("agent_id") or call.get("operation") or "unknown")
        item = agents.setdefault(agent_id, {
            "agent_id": agent_id, "calls": 0, "total_tokens": 0,
            "input_tokens": 0, "output_tokens": 0, "duration_ms": 0.0,
            "model_duration_ms": 0.0,
            "errors": 0,
        })
        item["calls"] += 1
        item["total_tokens"] += int(call.get("total_tokens") or 0)
        item["input_tokens"] += int(call.get("input_tokens") or 0)
        item["output_tokens"] += int(call.get("output_tokens") or 0)
        item["model_duration_ms"] += float(call.get("duration_ms") or 0)
        if not any(str(step.get("step_name") or "") == f"a2a:{agent_id}" for step in details["steps"]):
            item["duration_ms"] += float(call.get("duration_ms") or 0)
        item["errors"] += int(call.get("status") == "failed")
    for item in agents.values():
        item["avg_latency_ms"] = item["duration_ms"] / item["calls"] if item["calls"] else 0
    summary = {
        "trace_id": trace_id,
        "run_type": run.get("run_type"),
        "endpoint": (run.get("metadata") or {}).get("endpoint"),
        "novel_id": (run.get("metadata") or {}).get("novel_id"),
        "character_id": (run.get("metadata") or {}).get("character_id"),
        "status": run.get("status"),
        "duration_ms": run.get("duration_ms"),
        "input_tokens": run.get("input_tokens"),
        "output_tokens": run.get("output_tokens"),
        "total_tokens": run.get("total_tokens"),
        "model_call_count": run.get("model_call_count", 0),
        "usage_complete": bool(run.get("usage_complete")),
    }
    return {
        "trace_id": trace_id,
        "summary": summary,
        "agents": list(agents.values()),
        "steps": details["steps"],
        "model_invocations": calls,
        "run": run,
    }
