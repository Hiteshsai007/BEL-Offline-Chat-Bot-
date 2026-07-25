"""
SQLite-backed conversation session store for the BEL Offline AI Assistant.

Provides persistent multi-turn chat history management per session ID.
Thread-safe, loopback/offline-only, cross-platform (Windows & Linux).
Automatically prunes old messages per session exceeding MAX_STORED_MESSAGES.
"""
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional

from app.logger import get_logger
from app.settings import MAX_HISTORY_TURNS, MAX_STORED_MESSAGES, SESSION_DB_PATH

log = get_logger(__name__)

_lock = threading.Lock()
_store_instance: Optional["SessionStore"] = None


class SessionStore:
    """Manages session messages in a local SQLite database."""

    def __init__(self, db_path: Path = SESSION_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id);"
                )
        finally:
            conn.close()
        log.info("Session database initialized at %s", self.db_path)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        max_stored: int = MAX_STORED_MESSAGES,
    ) -> None:
        """Add a single message turn and prune older records beyond max_stored."""
        if not session_id or not content.strip():
            return

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content.strip()),
                )
                # Auto-prune messages for this session exceeding max_stored
                conn.execute(
                    """
                    DELETE FROM messages
                    WHERE session_id = ?
                      AND id NOT IN (
                          SELECT id FROM messages
                          WHERE session_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      );
                    """,
                    (session_id, session_id, max_stored),
                )
        finally:
            conn.close()

    def add_turn(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        max_stored: int = MAX_STORED_MESSAGES,
    ) -> None:
        """Convenience method to add user question and assistant answer in one transaction."""
        if not session_id:
            return
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
                    (session_id, user_question.strip()),
                )
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
                    (session_id, assistant_answer.strip()),
                )
                conn.execute(
                    """
                    DELETE FROM messages
                    WHERE session_id = ?
                      AND id NOT IN (
                          SELECT id FROM messages
                          WHERE session_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      );
                    """,
                    (session_id, session_id, max_stored),
                )
        finally:
            conn.close()

    def get_history(
        self,
        session_id: str,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> List[Dict[str, str]]:
        """
        Retrieve the latest N turns (max_turns * 2 messages) for a session ID
        in chronological order.
        """
        if not session_id:
            return []

        limit = max_turns * 2
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC;
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
        finally:
            conn.close()

    def clear_session(self, session_id: str) -> None:
        """Purge all stored messages for a specific session ID."""
        if not session_id:
            return
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        finally:
            conn.close()
        log.info("Cleared history for session: %s", session_id)


def get_session_store() -> SessionStore:
    """Return singleton SessionStore instance."""
    global _store_instance
    if _store_instance is None:
        with _lock:
            if _store_instance is None:
                _store_instance = SessionStore()
    return _store_instance
