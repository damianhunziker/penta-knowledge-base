#!/usr/bin/env python3
"""Persistent agent→title assignments, scoped by where they apply.

An assignment is "this agent always has these titles in scope". It is the
agent-facing counterpart of the corpus: the corpus says what exists, an
assignment says who gets it without searching first.

Scope
-----
An assignment is only meaningful *where* it applies. Giving a book to an
agent at project level must not silently give it to that agent in every other
project of the customer. So every row carries a scope, mirroring the
description-language ladder:

``deployment``
    ``custom`` — every customer of this deployment ("generally available").
``client``
    every project of one customer (``customer`` set).
``project``
    one project (``customer`` + ``project`` set).

Legacy rows (written before scoping existed) carry no scope. They are read as
**``legacy``**: global for backwards compatibility, but marked as such so the
audit can report them instead of pretending they were deliberate.

Reading is ladder-aware (:func:`titles_for`): a run inside customer A/project P
sees ``deployment`` ∪ ``client:A`` ∪ ``project:A/P``. Narrow never leaks wide —
a project-level assignment is invisible outside that project.

Storage
-------
One SQLite file, resolved by :mod:`langgraph_crew_knowledge_base.state_paths`
(a path inside the repo is read-only in the container —
``attempt to write a readonly database``). The schema is migrated in place on
first use, so an existing database keeps its rows.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langgraph_crew_knowledge_base.state_paths import resolve_db_path

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent

#: Kinds an assignment can name.
KINDS = ("book", "category", "tag")

#: Where an assignment applies, widest → narrowest.
SCOPES = ("deployment", "client", "project")

#: Scope stored on rows that predate scoping. Read like ``deployment`` so an
#: upgrade never drops a title, but reported separately so it is visible.
LEGACY_SCOPE = "legacy"


def db_path() -> Path:
    """Location of the assignment database (``KNOWLEDGE_DB_PATH`` overrides)."""
    return resolve_db_path(os.getenv("KNOWLEDGE_DB_PATH", "").strip() or None)


def _columns(conn: sqlite3.Connection) -> set:
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(assignment)")}
    except sqlite3.Error:
        return set()


def _connect() -> Optional[sqlite3.Connection]:
    """Open the database, creating and migrating the schema on first use."""
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
                created_at  TEXT NOT NULL
            )
            """
        )
        # Scoped columns. Added in place: an existing database keeps its rows
        # and simply reports them as `legacy`.
        existing = _columns(conn)
        for column, ddl in (
            ("scope", "ALTER TABLE assignment ADD COLUMN scope TEXT NOT NULL DEFAULT 'legacy'"),
            ("customer", "ALTER TABLE assignment ADD COLUMN customer TEXT NOT NULL DEFAULT ''"),
            ("project", "ALTER TABLE assignment ADD COLUMN project TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in existing:
                try:
                    conn.execute(ddl)
                except sqlite3.Error as exc:  # a concurrent migration
                    logger.debug("[KNOWLEDGE_BASE] migration %s: %s", column, exc)

        # SQLite cannot drop the inline constraint, so the old one is simply
        # wider (a value assigned at two scopes in one agent collapses to one
        # row) — acceptable, and the reason the scope is part of the row.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assignment_scope "
            "ON assignment(customer, project)"
        )
        # The effective uniqueness a fresh database gets: one row per
        # (agent, kind, value, scope ids). A database created before scoping
        # existed still carries the wider inline UNIQUE, so this index is
        # "best effort" — `INSERT ... ON CONFLICT DO NOTHING` needs it.
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_assignment_unique "
                "ON assignment(agent_id, kind, value, scope, customer, project)"
            )
        except sqlite3.Error as exc:
            logger.debug("[KNOWLEDGE_BASE] scoped unique index skipped: %s", exc)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assignment_agent ON assignment(agent_id)"
        )
        conn.commit()
        return conn
    except Exception as exc:  # noqa: BLE001 - assignments are optional
        logger.warning(
            "[KNOWLEDGE_BASE] assignment DB unavailable at %s: %s", db_path(), exc
        )
        return None


def current_scope() -> Dict[str, str]:
    """Customer/project of the active run (``{}`` values when unknown)."""
    customer = project = ""
    try:
        from app.core import access_scope

        scope = access_scope.current_scope()
        customer = str(getattr(scope, "customer", "") or "")
        project = str(getattr(scope, "project", "") or "")
    except Exception:  # noqa: BLE001 - scope is contextual, never fatal
        pass
    if not customer:
        customer = os.getenv("CUSTOMER_ID", "") or os.getenv("ACTIVE_CUSTOMER", "") or ""
    return {"customer": customer, "project": project}


def _normalise_scope(scope: str, customer: str, project: str) -> Dict[str, str]:
    """Validate a scope and fill in the identifiers it implies."""
    name = str(scope or "").strip().lower()
    if name not in SCOPES:
        return {"error": f"scope must be one of {list(SCOPES)} (got {scope!r})"}

    live = current_scope()
    customer = str(customer or "").strip() or live.get("customer", "")
    project = str(project or "").strip()
    if name == "project":
        project = project or live.get("project", "")
        if not customer or not project:
            return {
                "error": (
                    "a project-scoped assignment needs a customer and a project "
                    "(none active — call namespace___set_namespace first, or pass "
                    "customer/project)"
                )
            }
    if name == "client" and not customer:
        return {"error": "a client-scoped assignment needs a customer (none active)"}
    if name == "deployment":
        # Deployment = everyone; identifiers would be noise and could be read
        # as a narrower scope by a later reader.
        customer = project = ""
    if name == "client":
        project = ""
    return {"scope": name, "customer": customer, "project": project}


def assign(
    agent_id: str,
    books: Iterable[str] = (),
    categories: Iterable[str] = (),
    tags: Iterable[str] = (),
    assigned_by: str = "operator",
    scope: str = "deployment",
    customer: str = "",
    project: str = "",
) -> Dict[str, Any]:
    """Add assignments for ``agent_id`` at ``scope`` (idempotent).

    Returns the newly added entries plus the agent's assignments in that scope.
    """
    agent_id = str(agent_id or "").strip()
    if not agent_id:
        return {"error": "agent_id is required", "assigned": []}

    target = _normalise_scope(scope, customer, project)
    if target.get("error"):
        return {"error": target["error"], "assigned": []}

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
        return {
            "error": (
                f"assignment database unavailable at {db_path()} "
                "(not writable in this deployment)"
            ),
            "assigned": [],
        }

    created: List[str] = []
    now = datetime.now(timezone.utc).isoformat()
    try:
        for kind, value in wanted:
            exists = conn.execute(
                "SELECT 1 FROM assignment WHERE agent_id = ? AND kind = ? "
                "AND value = ? AND scope = ? AND customer = ? AND project = ?",
                (
                    agent_id,
                    kind,
                    value,
                    target["scope"],
                    target["customer"],
                    target["project"],
                ),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO assignment "
                "(agent_id, kind, value, assigned_by, created_at, scope, customer, project) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    kind,
                    value,
                    assigned_by,
                    now,
                    target["scope"],
                    target["customer"],
                    target["project"],
                ),
            )
            created.append(f"{kind}:{value}")
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] assign failed: %s", exc)
        return {"error": str(exc), "assigned": [], "database": str(db_path())}
    finally:
        conn.close()

    return {
        "agent_id": agent_id,
        "assigned": created,
        "all": list_for_agent(agent_id),
        "assigned_by": assigned_by,
        "scope": target,
        "database": str(db_path()),
    }


def list_for_agent(
    agent_id: str,
    *,
    scope: Optional[str] = None,
    customer: str = "",
    project: str = "",
) -> List[str]:
    """Assignments of one agent as ``["book:x", "category:y", …]``.

    Our default is the ladder-aware view (:func:`titles_for`): deployment ∪
    client ∪ project of the active scope. ``scope="deployment"`` lists exactly
    one rung; ``scope="*"`` lists every row (audit).
    """
    if scope is None:
        return titles_for(
            agent_id, customer=customer, project=project, kinds=KINDS
        )

    conn = _connect()
    if conn is None:
        return []
    where, params = _scope_where(scope, customer, project)
    try:
        rows = conn.execute(
            "SELECT kind, value FROM assignment WHERE agent_id = ? "
            + (f"AND {where} " if where else "")
            + "ORDER BY kind, value",
            (str(agent_id), *params),
        ).fetchall()
        return [f"{kind}:{value}" for kind, value in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] list failed: %s", exc)
        return []
    finally:
        conn.close()


def _scope_where(scope: str, customer: str, project: str) -> tuple:
    """SQL predicate + params for one scope selector."""
    name = str(scope or "").strip().lower()
    live = current_scope()
    customer = str(customer or "").strip() or live.get("customer", "")
    project = str(project or "").strip() or live.get("project", "")

    if name in ("*", "all"):
        return "", ()
    if name == "legacy":
        return "scope = ?", (LEGACY_SCOPE,)
    if name == "deployment":
        # Legacy rows were written when there was no scope at all: they were
        # deployment-wide in intent and are read as such.
        return "scope IN ('deployment', ?)", (LEGACY_SCOPE,)
    if name == "client":
        if not customer:
            return "0 = 1", ()
        return "scope = 'client' AND customer = ?", (customer,)
    if name == "project":
        if not customer or not project:
            return "0 = 1", ()
        return "scope = 'project' AND customer = ? AND project = ?", (customer, project)
    # Unknown selector: nothing rather than everything.
    return "0 = 1", ()


def titles_for(
    agent_id: str,
    *,
    customer: str = "",
    project: str = "",
    kinds: Sequence[str] = ("book",),
    include_legacy: bool = True,
) -> List[str]:
    """Titles of ``agent_id`` visible from the active (customer, project).

    The ladder is cumulative — a run inside a project sees what was assigned
    for the deployment, for its customer, and for that project. A narrower
    scope never leaks upward.
    """
    agent_id = str(agent_id or "").strip()
    if not agent_id:
        return []
    live = current_scope()
    customer = str(customer or "").strip() or live.get("customer", "")
    project = str(project or "").strip() or live.get("project", "")

    clauses = [
        "scope = 'deployment'",
        "scope = 'client' AND customer = ?",
        "scope = 'project' AND customer = ? AND project = ?",
    ]
    params: List[Any] = [customer, customer, project]
    if include_legacy:
        clauses.insert(0, "scope = ?")
        params.insert(0, LEGACY_SCOPE)

    kind_list = list(kinds or ("book",))
    placeholders = ",".join("?" for _ in kind_list)

    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT DISTINCT value FROM assignment WHERE agent_id = ? "
            f"AND kind IN ({placeholders}) AND (" + " OR ".join(clauses) + ") "
            "ORDER BY value",
            (agent_id, *kind_list, *params),
        ).fetchall()
        return [str(row[0]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] ladder read failed: %s", exc)
        return []
    finally:
        conn.close()


def books_for_agent(
    agent_id: str,
    *,
    customer: str = "",
    project: str = "",
) -> List[str]:
    """Book ids visible to ``agent_id`` from the active scope (ladder)."""
    return titles_for(agent_id, customer=customer, project=project, kinds=("book",))


def list_all(scope: Optional[str] = None) -> Dict[str, List[str]]:
    """Every agent with its assignments (scoped when ``scope`` is given)."""
    conn = _connect()
    if conn is None:
        return {}
    where, params = ("", ()) if scope is None else _scope_where(scope, "", "")
    try:
        rows = conn.execute(
            "SELECT agent_id, kind, value FROM assignment "
            + (f"WHERE {where} " if where else "")
            + "ORDER BY agent_id, kind, value",
            params,
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


def grants() -> List[Dict[str, Any]]:
    """Every assignment row with its scope — the access overview.

    This is the table the operator and the agents read and write: what is
    available, to which agent, and **where** it applies.
    """
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT agent_id, kind, value, scope, customer, project, assigned_by, created_at "
            "FROM assignment "
            "ORDER BY CASE scope "
            "  WHEN 'legacy' THEN 0 WHEN 'deployment' THEN 1 "
            "  WHEN 'client' THEN 2 ELSE 3 END, customer, project, agent_id, kind, value"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] grants failed: %s", exc)
        return []
    finally:
        conn.close()

    return [
        {
            "agent_id": agent_id,
            "kind": kind,
            "value": value,
            "scope": scope,
            "customer": customer or None,
            "project": project or None,
            "assigned_by": assigned_by,
            "created_at": created_at,
            "database": str(db_path()),
        }
        for (
            agent_id,
            kind,
            value,
            scope,
            customer,
            project,
            assigned_by,
            created_at,
        ) in rows
    ]


def unassign(
    agent_id: str,
    kind: str,
    value: str,
    *,
    scope: str = "deployment",
    customer: str = "",
    project: str = "",
) -> bool:
    """Remove one assignment (``True`` when a row was deleted)."""
    conn = _connect()
    if conn is None:
        return False
    where, params = _scope_where(scope, customer, project)
    if not where:
        return False
    try:
        cursor = conn.execute(
            "DELETE FROM assignment WHERE agent_id = ? AND kind = ? AND value = ? "
            f"AND ({where})",
            (str(agent_id), str(kind), str(value), *params),
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
    "LEGACY_SCOPE",
    "SCOPES",
    "assign",
    "books_for_agent",
    "current_scope",
    "db_path",
    "grants",
    "list_all",
    "list_for_agent",
    "titles_for",
    "unassign",
]
