from __future__ import annotations

import sqlite3
from datetime import datetime
from uuid import uuid4

from .config import DATABASE_PATH

#Esse arquivo é responsável por gerenciar o banco de dados SQLite que armazena as conversas e mensagens do assistente. Ele fornece funções para criar, ler, atualizar e deletar conversas e mensagens, garantindo que os dados sejam persistidos de forma eficiente e organizada.
def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
            "ON messages(conversation_id, id)"
        )


def create_conversation(title: str = "Nova conversa") -> dict:
    conversation_id = str(uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title[:120], now, now),
        )

    return {
        "id": conversation_id,
        "title": title[:120],
        "created_at": now,
        "updated_at": now,
    }


def ensure_conversation(conversation_id: str | None) -> str:
    if conversation_id:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row:
                return conversation_id

    return create_conversation()["id"]


def add_message(conversation_id: str, role: str, content: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

        if role == "user":
            count = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ? AND role = 'user'",
                (conversation_id,),
            ).fetchone()["total"]
            if count == 1:
                title = " ".join(content.strip().split())[:70] or "Nova conversa"
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title, conversation_id),
                )


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict]:
    with _connect() as conn:
        if limit:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM (
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

    return [dict(row) for row in rows]


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
