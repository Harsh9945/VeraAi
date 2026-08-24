"""
SQLite-backed state store for the challenge bot.

Implements the same API as the original in-memory state:
  - contexts: keyed by (scope, context_id) -> {"version": int, "payload": dict}
    Idempotent on (context_id, version) per the testing brief §2.1.
  - conversations: keyed by conversation_id -> ConversationState
    Tracks turn history + a few derived flags (auto-reply streak, intent signal)
    used by the composer / validator / auto-reply detector.

Provides process-level isolation + file-based persistence via SQLite.
"""

import sqlite3
import json
import os
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ContextEntry:
    version: int
    payload: dict[str, Any]


@dataclass
class Turn:
    from_role: Literal["merchant", "customer", "vera"]
    message: str
    ts: str


class PersistentTurnsList:
    def __init__(self, store, conversation_id: str):
        self.store = store
        self.conversation_id = conversation_id

    def append(self, turn: Turn):
        self.store._add_turn(self.conversation_id, turn)

    def __iter__(self):
        return iter(self.store._get_turns(self.conversation_id))

    def __len__(self):
        return len(self.store._get_turns(self.conversation_id))

    def __getitem__(self, index):
        return self.store._get_turns(self.conversation_id)[index]


class PersistentSentBodiesList:
    def __init__(self, store, conversation_id: str):
        self.store = store
        self.conversation_id = conversation_id

    def append(self, body: str):
        self.store._add_sent_body(self.conversation_id, body)

    def __iter__(self):
        return iter(self.store._get_sent_bodies(self.conversation_id))

    def __len__(self):
        return len(self.store._get_sent_bodies(self.conversation_id))

    def __contains__(self, item):
        return item in self.store._get_sent_bodies(self.conversation_id)


class ConversationState:
    def __init__(self, store, conversation_id: str):
        self.store = store
        self.conversation_id = conversation_id
        self.turns = PersistentTurnsList(store, conversation_id)
        self.sent_bodies = PersistentSentBodiesList(store, conversation_id)

    @property
    def merchant_id(self) -> str | None:
        return self.store._get_conv_field(self.conversation_id, "merchant_id")

    @merchant_id.setter
    def merchant_id(self, val: str | None):
        self.store._set_conv_field(self.conversation_id, "merchant_id", val)

    @property
    def customer_id(self) -> str | None:
        return self.store._get_conv_field(self.conversation_id, "customer_id")

    @customer_id.setter
    def customer_id(self, val: str | None):
        self.store._set_conv_field(self.conversation_id, "customer_id", val)

    @property
    def trigger_id(self) -> str | None:
        return self.store._get_conv_field(self.conversation_id, "trigger_id")

    @trigger_id.setter
    def trigger_id(self, val: str | None):
        self.store._set_conv_field(self.conversation_id, "trigger_id", val)

    @property
    def unanswered_nudges(self) -> int:
        return self.store._get_conv_field(self.conversation_id, "unanswered_nudges") or 0

    @unanswered_nudges.setter
    def unanswered_nudges(self, val: int):
        self.store._set_conv_field(self.conversation_id, "unanswered_nudges", val)

    @property
    def ended(self) -> bool:
        return bool(self.store._get_conv_field(self.conversation_id, "ended"))

    @ended.setter
    def ended(self, val: bool):
        self.store._set_conv_field(self.conversation_id, "ended", 1 if val else 0)

    def last_n_merchant_messages(self, n: int) -> list[str]:
        msgs = [t.message for t in self.turns if t.from_role in ("merchant", "customer")]
        return msgs[-n:]


class ConversationsDict:
    def __init__(self, store):
        self.store = store

    def get(self, conversation_id: str) -> ConversationState | None:
        if self.store._conv_exists(conversation_id):
            return ConversationState(self.store, conversation_id)
        return None

    def __getitem__(self, conversation_id: str) -> ConversationState:
        if self.store._conv_exists(conversation_id):
            return ConversationState(self.store, conversation_id)
        raise KeyError(conversation_id)

    def __setitem__(self, conversation_id: str, conv: ConversationState):
        pass

    def values(self) -> list[ConversationState]:
        return self.store._all_conversations()


class Store:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get("DATABASE_PATH", "state.db")
        self.db_path = db_path
        self._init_db()
        self.conversations = ConversationsDict(self)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                scope TEXT,
                context_id TEXT,
                version INTEGER,
                payload TEXT,
                PRIMARY KEY (scope, context_id)
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                merchant_id TEXT,
                customer_id TEXT,
                trigger_id TEXT,
                unanswered_nudges INTEGER DEFAULT 0,
                ended INTEGER DEFAULT 0
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                conversation_id TEXT,
                from_role TEXT,
                message TEXT,
                ts TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_bodies (
                conversation_id TEXT,
                body TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            );
            """)
            conn.commit()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    # ---- context ----

    def push_context(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[bool, int | None]:
        """Returns (accepted, current_version_if_rejected)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM contexts WHERE scope = ? AND context_id = ?",
                (scope, context_id)
            )
            row = cursor.fetchone()
            if row and row[0] >= version:
                return False, row[0]

            cursor.execute(
                "INSERT OR REPLACE INTO contexts (scope, context_id, version, payload) VALUES (?, ?, ?, ?)",
                (scope, context_id, version, json.dumps(payload))
            )
            conn.commit()
            return True, None

    def get_context(self, scope: str, context_id: str | None) -> dict | None:
        if not context_id:
            return None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload FROM contexts WHERE scope = ? AND context_id = ?",
                (scope, context_id)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def counts(self) -> dict[str, int]:
        res = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scope, COUNT(*) FROM contexts GROUP BY scope")
            for row in cursor.fetchall():
                if row[0] in res:
                    res[row[0]] = row[1]
        return res


    def all_of_scope(self, scope: str) -> dict[str, dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT context_id, payload FROM contexts WHERE scope = ?", (scope,))
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    # ---- conversations ----

    def _conv_exists(self, conversation_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
            return cursor.fetchone() is not None

    def _all_conversations(self) -> list[ConversationState]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT conversation_id FROM conversations")
            ids = [row[0] for row in cursor.fetchall()]
            return [ConversationState(self, cid) for cid in ids]

    def get_or_create_conversation(self, conversation_id: str, merchant_id: str | None = None,
                                    customer_id: str | None = None, trigger_id: str | None = None) -> ConversationState:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO conversations (conversation_id, merchant_id, customer_id, trigger_id, unanswered_nudges, ended) VALUES (?, ?, ?, ?, 0, 0)",
                    (conversation_id, merchant_id, customer_id, trigger_id)
                )
                conn.commit()
        return ConversationState(self, conversation_id)

    def teardown(self) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM contexts")
            conn.execute("DELETE FROM conversations")
            conn.execute("DELETE FROM turns")
            conn.execute("DELETE FROM sent_bodies")
            conn.commit()

    # ---- internal helpers for ConversationState ----

    def _get_conv_field(self, conversation_id: str, field_name: str) -> Any:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {field_name} FROM conversations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _set_conv_field(self, conversation_id: str, field_name: str, value: Any):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE conversations SET {field_name} = ? WHERE conversation_id = ?", (value, conversation_id))
            conn.commit()

    def _get_turns(self, conversation_id: str) -> list[Turn]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT from_role, message, ts FROM turns WHERE conversation_id = ? ORDER BY rowid ASC",
                (conversation_id,)
            )
            return [Turn(from_role=row[0], message=row[1], ts=row[2]) for row in cursor.fetchall()]

    def _add_turn(self, conversation_id: str, turn: Turn):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO turns (conversation_id, from_role, message, ts) VALUES (?, ?, ?, ?)",
                (conversation_id, turn.from_role, turn.message, turn.ts)
            )
            conn.commit()

    def _get_sent_bodies(self, conversation_id: str) -> list[str]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT body FROM sent_bodies WHERE conversation_id = ?", (conversation_id,))
            return [row[0] for row in cursor.fetchall()]

    def _add_sent_body(self, conversation_id: str, body: str):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sent_bodies (conversation_id, body) VALUES (?, ?)", (conversation_id, body))
            conn.commit()


store = Store()
