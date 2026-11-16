#!/usr/bin/env python3
"""Persistent agent→title assignments.

An assignment is "this agent always has these titles in scope". It is the
agent-facing counterpart of the corpus: the corpus says what exists, an
assignment says who gets it without searching first.

Storage is one SQLite file next to the plugin (``plugin.env`` →
``KNOWLEDGE_DB_PATH``, default ``<plugin>/knowledge_base.db``). SQLite because
the data is tiny, relational (agent → title set) and must survive a process
restart — the in-memory dict this replaces did not, so an assignment made by
one run was invisible to the next.

Schema
------

``assignment(id, agent_id, kind, value, assigned_by, created_at)``
    ``kind`` is ``book``, ``category`` or ``tag``. ``(agent_id, kind, value)``
    is unique, so assigning twice is a no-op instead of a duplicate row.

The table is created on first use; nothing has to be migrated by hand.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent

#: Kinds an assignment can name.
KINDS = ("book", "category", "tag")


def db_path() -> Path:
    """Location of the assignment database (``KNOWLEDGE_DB_PATH`` overrides)."""
    configured = os.getenv("KNOWLEDGE_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    # Keep it beside the plugin config, not in the repo root.
    return PLUGIN_DIR / "knowledge_base.db"


def _connect() -> Optional[sqlite3.Connection]:
    """Open the database, creating the schema on first use."""
    try:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignment (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id    TEXT NOT NULL,
                kind        TEXT NOT NULL,
                value       TEXT NOT NULL,
                assigned_by TEXT NOT NULL DEFAULT 'operator',
                created_at  TEXT NOT NULL,
                UNIQUE (agent_id, kind, value)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assignment_agent ON assignment(agent_id)"
        )
        conn.commit()
        return conn
    except Exception as exc:  # noqa: BLE001 - assignments are optional
        logger.warning("[KNOWLEDGE_BASE] assignment DB unavailable: %s", exc)
        return None


def assign(
    agent_id: str,
    books: Iterable[str] = (),
    categories: Iterable[str] = (),
    tags: Iterable[str] = (),
    assigned_by: str = "operator",
) -> Dict[str, Any]:
    """Add assignments for ``agent_id`` (idempotent).

    Returns the newly added entries plus the agent's full assignment list.
    """
    agent_id = str(agent_id or "").strip()
    if not agent_id:
        return {"error": "agent_id is required", "assigned": []}

    wanted: List[tuple] = []
    for kind, values in (
        ("book", books),
        ("category", categories),
        ("tag", tags),
    ):
        for value in values or ():
            text = str(value or "").strip()
            if text:
                wanted.append((kind, text))

    conn = _connect()
    if conn is None:
        return {"error": "assignment database unavailable", "assigned": []}

    created: List[str] = []
    now = datetime.now(timezone.utc).isoformat()
    try:
        for kind, value in wanted:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO assignment "
                "(agent_id, kind, value, assigned_by, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent_id, kind, value, assigned_by, now),
            )
            if cursor.rowcount:
                created.append(f"{kind}:{value}")
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] assign failed: %s", exc)
        return {"error": str(exc), "assigned": []}
    finally:
        conn.close()

    return {
        "agent_id": agent_id,
        "assigned": created,
        "all": list_for_agent(agent_id),
        "assigned_by": assigned_by,
    }


def list_for_agent(agent_id: str) -> List[str]:
    """Assignments of one agent as ``["book:x", "category:y", …]``."""
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT kind, value FROM assignment WHERE agent_id = ? "
            "ORDER BY kind, value",
            (str(agent_id),),
        ).fetchall()
        return [f"{kind}:{value}" for kind, value in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] list failed: %s", exc)
        return []
    finally:
        conn.close()


def books_for_agent(agent_id: str) -> List[str]:
    """Book ids assigned to ``agent_id`` (the common case)."""
    return [
        entry.split(":", 1)[1]
        for entry in list_for_agent(agent_id)
        if entry.startswith("book:")
    ]


def list_all() -> Dict[str, List[str]]:
    """Every agent with its assignments."""
    conn = _connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT agent_id, kind, value FROM assignment ORDER BY agent_id, kind, value"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] list_all failed: %s", exc)
        return {}
    finally:
        conn.close()

    out: Dict[str, List[str]] = {}
    for agent_id, kind, value in rows:
        out.setdefault(agent_id, []).append(f"{kind}:{value}")
    return out


def unassign(agent_id: str, kind: str, value: str) -> bool:
    """Remove one assignment (``True`` when a row was deleted)."""
    conn = _connect()
    if conn is None:
        return False
    try:
        cursor = conn.execute(
            "DELETE FROM assignment WHERE agent_id = ? AND kind = ? AND value = ?",
            (str(agent_id), str(kind), str(value)),
        )
        conn.commit()
        return bool(cursor.rowcount)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] unassign failed: %s", exc)
        return False
    finally:
        conn.close()


__all__ = [
    "KINDS",
    "assign",
    "books_for_agent",
    "db_path",
    "list_all",
    "list_for_agent",
    "unassign",
]
