"""MySQL schema initialization."""

from __future__ import annotations

import os

from app.storage.mysql.client import mysql_connection


def init_mysql_schema() -> None:
    database = os.getenv("MYSQL_DATABASE", "three_body_agent")
    with mysql_connection(database="") as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

    with mysql_connection(database=database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(64) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    display_name VARCHAR(64) NULL,
                    avatar_url VARCHAR(512) NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'active',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    last_login_at DATETIME NULL,
                    INDEX idx_users_username (username)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_threads (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    user_id BIGINT NOT NULL,
                    username VARCHAR(64) NOT NULL,
                    character_name VARCHAR(64) NOT NULL,
                    timeline_stage VARCHAR(16) NOT NULL,
                    mode VARCHAR(32) NOT NULL,
                    thread_name VARCHAR(128) NOT NULL DEFAULT '默认线程',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_chat_thread (
                        user_id,
                        character_name,
                        timeline_stage,
                        mode,
                        thread_name
                    ),
                    INDEX idx_chat_threads_user (user_id),
                    INDEX idx_chat_threads_character_stage (character_name, timeline_stage),
                    CONSTRAINT fk_chat_threads_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    thread_id BIGINT NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    metadata JSON NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_chat_messages_thread_created (thread_id, created_at),
                    CONSTRAINT fk_chat_messages_thread
                        FOREIGN KEY (thread_id) REFERENCES chat_threads(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    memory_id VARCHAR(96) NOT NULL UNIQUE,
                    user_id BIGINT NOT NULL,
                    username VARCHAR(64) NOT NULL,
                    character_name VARCHAR(64) NOT NULL,
                    timeline_stage VARCHAR(16) NOT NULL,
                    mode VARCHAR(32) NOT NULL,
                    thread_name VARCHAR(128) NOT NULL,
                    thread_id BIGINT NULL,
                    memory_type VARCHAR(32) NOT NULL DEFAULT 'episodic',
                    content MEDIUMTEXT NOT NULL,
                    summary TEXT NULL,
                    importance FLOAT NOT NULL DEFAULT 0.5,
                    source_turn_range VARCHAR(64) NULL,
                    metadata JSON NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    last_used_at DATETIME NULL,
                    INDEX idx_episodic_scope (
                        user_id,
                        character_name,
                        timeline_stage,
                        mode,
                        thread_name
                    ),
                    INDEX idx_episodic_thread (thread_id),
                    INDEX idx_episodic_importance (importance),
                    CONSTRAINT fk_episodic_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE,
                    CONSTRAINT fk_episodic_thread
                        FOREIGN KEY (thread_id) REFERENCES chat_threads(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
