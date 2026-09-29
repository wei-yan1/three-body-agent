"""Unified PostgreSQL schema for the test-stage application."""
from __future__ import annotations

from threading import Lock
from typing import Any

from app.storage.postgres.client import postgres_connection

_lock = Lock()
_initialized = False


def init_postgres_schema() -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        with postgres_connection() as connection, connection.cursor() as cursor:
            _create_core(cursor)
            _create_storyrole(cursor)
            _create_observability(cursor)
        _initialized = True


def _create_core(cur: Any) -> None:
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id BIGSERIAL PRIMARY KEY, username VARCHAR(64) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL, display_name VARCHAR(64), avatar_url VARCHAR(512),
        status VARCHAR(32) NOT NULL DEFAULT 'active', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), last_login_at TIMESTAMPTZ)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_threads (
        id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        username VARCHAR(64) NOT NULL, character_name VARCHAR(64) NOT NULL,
        timeline_stage VARCHAR(16) NOT NULL, mode VARCHAR(32) NOT NULL,
        thread_name VARCHAR(128) NOT NULL DEFAULT '默认线程', novel_id VARCHAR(96),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (user_id, novel_id, character_name, timeline_stage, mode, thread_name))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY, thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
        role VARCHAR(32) NOT NULL, content TEXT NOT NULL, metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
    cur.execute("ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS character_id BIGINT")
    cur.execute("ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS period_id BIGINT")
    cur.execute("ALTER TABLE chat_threads DROP CONSTRAINT IF EXISTS chat_threads_user_id_novel_id_character_name_timeline_stage_mode_thread_name_key")
    cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_threads_scope
        ON chat_threads (user_id, COALESCE(novel_id, ''), COALESCE(character_id, 0),
                         COALESCE(period_id, 0), mode, thread_name)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS episodic_memories (
        id BIGSERIAL PRIMARY KEY, memory_id VARCHAR(96) NOT NULL UNIQUE,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, username VARCHAR(64) NOT NULL,
        character_name VARCHAR(64) NOT NULL, timeline_stage VARCHAR(16) NOT NULL, mode VARCHAR(32) NOT NULL,
        thread_name VARCHAR(128) NOT NULL, thread_id BIGINT REFERENCES chat_threads(id) ON DELETE CASCADE,
        memory_type VARCHAR(32) NOT NULL DEFAULT 'episodic', content TEXT NOT NULL, summary TEXT,
        importance DOUBLE PRECISION NOT NULL DEFAULT 0.5, source_turn_range VARCHAR(64), metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), last_used_at TIMESTAMPTZ)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS app_settings (
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        setting_key VARCHAR(128) NOT NULL, setting_value JSONB NOT NULL,
        is_secret BOOLEAN NOT NULL DEFAULT FALSE, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, setting_key))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_context_checkpoints (
        thread_id BIGINT PRIMARY KEY REFERENCES chat_threads(id) ON DELETE CASCADE,
        covered_message_id BIGINT REFERENCES chat_messages(id) ON DELETE SET NULL,
        summary TEXT NOT NULL,
        degraded BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_episodic_scope ON episodic_memories(user_id, character_name, timeline_stage, mode, thread_name)")


def _create_storyrole(cur: Any) -> None:
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_novels (
        id BIGSERIAL PRIMARY KEY, novel_id VARCHAR(96) NOT NULL,
        owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'processing',
        source_kind VARCHAR(24) NOT NULL DEFAULT 'imported', chunks_path TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (novel_id, owner_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_characters (
        id BIGSERIAL PRIMARY KEY, novel_id VARCHAR(96) NOT NULL,
        owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        canonical_name VARCHAR(255) NOT NULL, aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
        mention_count INTEGER NOT NULL DEFAULT 0, first_chapter VARCHAR(255), last_chapter VARCHAR(255),
        evidence_summary TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (novel_id, owner_id, canonical_name), UNIQUE (novel_id, owner_id, id),
        FOREIGN KEY (novel_id, owner_id) REFERENCES storyrole_novels(novel_id, owner_id) ON DELETE CASCADE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_character_periods (
        id BIGSERIAL PRIMARY KEY, character_id BIGINT NOT NULL REFERENCES storyrole_characters(id) ON DELETE CASCADE,
        novel_id VARCHAR(96) NOT NULL, owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL, chapter_start VARCHAR(255), chapter_end VARCHAR(255),
        keywords JSONB NOT NULL DEFAULT '[]'::jsonb, evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        rationale TEXT NOT NULL DEFAULT '', confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        status VARCHAR(32) NOT NULL DEFAULT 'confirmed', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE (character_id, name),
        FOREIGN KEY (novel_id, owner_id, character_id) REFERENCES storyrole_characters(novel_id, owner_id, id) ON DELETE CASCADE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_period_analyses (
        id BIGSERIAL PRIMARY KEY, character_id BIGINT NOT NULL REFERENCES storyrole_characters(id) ON DELETE CASCADE,
        novel_id VARCHAR(96) NOT NULL, owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        decision VARCHAR(32) NOT NULL, confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        result JSONB NOT NULL DEFAULT '{}'::jsonb, status VARCHAR(32) NOT NULL DEFAULT 'ready',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        FOREIGN KEY (novel_id, owner_id, character_id) REFERENCES storyrole_characters(novel_id, owner_id, id) ON DELETE CASCADE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_persona_profiles (
        id BIGSERIAL PRIMARY KEY, character_id BIGINT NOT NULL REFERENCES storyrole_characters(id) ON DELETE CASCADE,
        period_id BIGINT REFERENCES storyrole_character_periods(id) ON DELETE SET NULL,
        novel_id VARCHAR(96) NOT NULL, owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        version INTEGER NOT NULL, profile JSONB NOT NULL, model VARCHAR(128) NOT NULL DEFAULT 'heuristic',
        status VARCHAR(32) NOT NULL DEFAULT 'ready', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (character_id, version))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_persona_evidence (
        id BIGSERIAL PRIMARY KEY, profile_id BIGINT NOT NULL REFERENCES storyrole_persona_profiles(id) ON DELETE CASCADE,
        chapter VARCHAR(255), chunk_id VARCHAR(255), claim TEXT NOT NULL, evidence_text TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_relationship_states (
        id BIGSERIAL PRIMARY KEY, novel_id VARCHAR(96) NOT NULL,
        character_id BIGINT NOT NULL REFERENCES storyrole_characters(id) ON DELETE CASCADE,
        owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, target_type VARCHAR(32) NOT NULL,
        target_key VARCHAR(128) NOT NULL, target_name VARCHAR(255) NOT NULL,
        trust DOUBLE PRECISION NOT NULL DEFAULT 0.5, intimacy DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        tension DOUBLE PRECISION NOT NULL DEFAULT 0.0, dependency DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        state JSONB NOT NULL DEFAULT '{}'::jsonb, evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE (character_id, owner_id, target_type, target_key),
        FOREIGN KEY (novel_id, owner_id) REFERENCES storyrole_novels(novel_id, owner_id) ON DELETE CASCADE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS storyrole_memory_candidates (
        id BIGSERIAL PRIMARY KEY, novel_id VARCHAR(96) NOT NULL,
        character_id BIGINT NOT NULL REFERENCES storyrole_characters(id) ON DELETE CASCADE,
        owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, thread_id BIGINT REFERENCES chat_threads(id) ON DELETE SET NULL,
        thread_name VARCHAR(128) NOT NULL DEFAULT '线程1', memory_type VARCHAR(64) NOT NULL,
        content TEXT NOT NULL, summary TEXT NOT NULL, importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
        status VARCHAR(16) NOT NULL DEFAULT 'pending', reason TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), reviewed_at TIMESTAMPTZ,
        FOREIGN KEY (novel_id, owner_id) REFERENCES storyrole_novels(novel_id, owner_id) ON DELETE CASCADE)""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_storyrole_characters_novel ON storyrole_characters(novel_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_storyrole_profiles_character ON storyrole_persona_profiles(character_id, version DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_storyrole_memory_candidates_owner ON storyrole_memory_candidates(owner_id, status, created_at DESC)")


def _create_observability(cur: Any) -> None:
    cur.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
        trace_id VARCHAR(64) PRIMARY KEY, run_type VARCHAR(32) NOT NULL, status VARCHAR(24) NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb, error TEXT, started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ, duration_ms DOUBLE PRECISION, input_tokens BIGINT, output_tokens BIGINT,
        total_tokens BIGINT, model_call_count INTEGER NOT NULL DEFAULT 0, usage_complete BOOLEAN NOT NULL DEFAULT FALSE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pipeline_steps (
        id BIGSERIAL PRIMARY KEY, trace_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(trace_id) ON DELETE CASCADE,
        parent_step_id BIGINT REFERENCES pipeline_steps(id) ON DELETE SET NULL, step_name VARCHAR(96) NOT NULL,
        status VARCHAR(24) NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, error TEXT,
        duration_ms DOUBLE PRECISION, started_at TIMESTAMPTZ NOT NULL, ended_at TIMESTAMPTZ)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS model_invocations (
        id BIGSERIAL PRIMARY KEY, trace_id VARCHAR(64), owner_id BIGINT, agent_id VARCHAR(96), operation VARCHAR(64) NOT NULL,
        provider VARCHAR(64) NOT NULL, model VARCHAR(128) NOT NULL, status VARCHAR(24) NOT NULL,
        duration_ms DOUBLE PRECISION, input_tokens BIGINT, output_tokens BIGINT, cached_tokens BIGINT,
        total_tokens BIGINT, cost NUMERIC(18,8), cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
        prompt_hash VARCHAR(64), metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        started_at TIMESTAMPTZ NOT NULL, ended_at TIMESTAMPTZ, error TEXT)""")
    cur.execute("ALTER TABLE model_invocations ADD COLUMN IF NOT EXISTS agent_id VARCHAR(96)")
    cur.execute("""UPDATE model_invocations SET agent_id = CASE operation
        WHEN 'nuwa_profile' THEN 'nuwa-profiler'
        WHEN 'period_analysis' THEN 'character-period-analysis'
        WHEN 'storyrole_answer_generation' THEN 'character-conversation'
        WHEN 'character_reasoning' THEN 'character-reasoning'
        WHEN 'role_cognition' THEN 'role-cognition'
        WHEN 'consistency_guard' THEN 'consistency-guard'
        WHEN 'evidence_analysis' THEN 'evidence-analysis'
        WHEN 'deep_question_planning' THEN 'deep-question-planner'
        WHEN 'embedding' THEN 'embedding'
        WHEN 'rerank' THEN 'reranker'
        WHEN 'a2a:nuwa-profiler' THEN 'nuwa-profiler'
        WHEN 'a2a:character-period-analysis' THEN 'character-period-analysis'
        ELSE operation END
        WHERE agent_id IS NULL""")
    cur.execute("""CREATE TABLE IF NOT EXISTS model_usage_daily (
        usage_date DATE NOT NULL, provider VARCHAR(64) NOT NULL, model VARCHAR(128) NOT NULL,
        operation VARCHAR(64) NOT NULL, call_count BIGINT NOT NULL DEFAULT 0, remote_call_count BIGINT NOT NULL DEFAULT 0,
        cache_hits BIGINT NOT NULL DEFAULT 0, input_tokens BIGINT NOT NULL DEFAULT 0, output_tokens BIGINT NOT NULL DEFAULT 0,
        cached_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0, total_cost NUMERIC(18,8) NOT NULL DEFAULT 0,
        error_count BIGINT NOT NULL DEFAULT 0, PRIMARY KEY (usage_date, provider, model, operation))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS embedding_cache (
        provider VARCHAR(64) NOT NULL, model VARCHAR(128) NOT NULL, model_version VARCHAR(128) NOT NULL DEFAULT '',
        text_hash VARCHAR(64) NOT NULL, dimension INTEGER NOT NULL, vector JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (provider, model, model_version, text_hash, dimension))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS evidence_graph_nodes (
        id BIGSERIAL PRIMARY KEY, trace_id VARCHAR(64) NOT NULL, node_type VARCHAR(40) NOT NULL,
        source_id VARCHAR(160), content_hash VARCHAR(64), content_preview TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS evidence_graph_edges (
        id BIGSERIAL PRIMARY KEY, trace_id VARCHAR(64) NOT NULL,
        from_node_id BIGINT NOT NULL REFERENCES evidence_graph_nodes(id) ON DELETE CASCADE,
        to_node_id BIGINT NOT NULL REFERENCES evidence_graph_nodes(id) ON DELETE CASCADE,
        edge_type VARCHAR(40) NOT NULL, weight DOUBLE PRECISION, metadata JSONB NOT NULL DEFAULT '{}'::jsonb)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS eval_runs (
        eval_run_id VARCHAR(64) PRIMARY KEY, config_hash VARCHAR(64) NOT NULL, dataset_hash VARCHAR(64) NOT NULL,
        config JSONB NOT NULL, status VARCHAR(24) NOT NULL, metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
        started_at TIMESTAMPTZ NOT NULL, ended_at TIMESTAMPTZ, error TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS eval_cases (
        id BIGSERIAL PRIMARY KEY, eval_run_id VARCHAR(64) NOT NULL REFERENCES eval_runs(eval_run_id) ON DELETE CASCADE,
        case_id VARCHAR(128) NOT NULL, input JSONB NOT NULL, UNIQUE (eval_run_id, case_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id BIGSERIAL PRIMARY KEY, eval_run_id VARCHAR(64) NOT NULL REFERENCES eval_runs(eval_run_id) ON DELETE CASCADE,
        case_id VARCHAR(128) NOT NULL, trace_id VARCHAR(64), retrieval JSONB NOT NULL DEFAULT '{}'::jsonb,
        answer_quality JSONB NOT NULL DEFAULT '{}'::jsonb, cost JSONB NOT NULL DEFAULT '{}'::jsonb,
        latency JSONB NOT NULL DEFAULT '{}'::jsonb, passed BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE (eval_run_id, case_id))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_steps_trace ON pipeline_steps(trace_id, started_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_invocations_lookup ON model_invocations(provider, model, operation, started_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_invocations_agent ON model_invocations(owner_id, agent_id, started_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evidence_graph_trace ON evidence_graph_nodes(trace_id, created_at)")
