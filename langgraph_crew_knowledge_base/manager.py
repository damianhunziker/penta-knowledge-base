#!/usr/bin/env python3
"""Read-side model behind the knowledge-base manager.

The manager shows three things and lets an operator change them:

``Bases``      what exists in the corpus (a *base* is a corpus: a set of books)
``Bindings``   which agent may read which base, at which scope
``Namespaces`` how those bindings differ per customer and per project

The store (Qdrant payload + the SQLite grant table) already carries everything
needed; this module only *reads* it and folds it into the shapes the manager and
its tests want. Writing goes through
:mod:`langgraph_crew_knowledge_base.assignments` (the same code the tools use),
so a change made in the manager is the change an agent sees — never a second
code path that can drift.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus: bases, books, chunk quality
# ---------------------------------------------------------------------------


def _payload_books(points: Iterable[Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Fold chunk payloads into one row per ``(corpus, book_id)``."""
    books: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for point in points:
        payload = getattr(point, "payload", None) or {}
        corpus = str(payload.get("corpus") or "unspecified")
        book_id = str(payload.get("book_id") or "")
        if not book_id:
            continue
        key = (corpus, book_id)
        row = books.get(key)
        if row is None:
            row = books[key] = {
                "corpus": corpus,
                "book_id": book_id,
                "title": payload.get("title") or book_id,
                "authors": list(payload.get("authors") or []),
                "year": payload.get("year"),
                "tier": payload.get("tier") or "unspecified",
                "chunks": 0,
                "real_chunks": 0,
                "chars": 0,
                "source": payload.get("source_ref") or "",
                # The stored original (knowledge-raw:…) when the book was
                # ingested after the raw store existed; "" otherwise.
                "raw_ref": payload.get("raw_ref") or "",
            }
        text = payload.get("text") or ""
        row["chunks"] += 1
        row["chars"] += len(text)
        # A chunk is "real text" when it is neither the document's YAML
        # frontmatter nor a one-line bullet. Those two are what a hand-written
        # summary produces, and they are useless to a reader.
        if not text.lstrip().startswith("---") and len(text) >= 800:
            row["real_chunks"] += 1
        if not row["source"]:
            row["source"] = payload.get("source_ref") or ""
    for row in books.values():
        row["avg_chars"] = row["chars"] // max(1, row["chunks"])
        row["is_real"] = row["real_chunks"] >= max(1, row["chunks"] // 2)
    return books


def _scroll_points() -> List[Any]:
    """All chunk points of the collection (``[]`` when unreachable).

    The one seam the manager reads through: tests replace it, the real path
    probes Qdrant. Kept separate from :func:`corpus_rows` so the folding logic
    is testable without a vector database.
    """
    try:
        from langgraph_crew_knowledge_base.tools import KB_COLLECTION, _qdrant

        client = _qdrant()
        if client is None or not client.collection_exists(KB_COLLECTION):
            return []
        points, _ = client.scroll(KB_COLLECTION, limit=5000, with_payload=True)
        return list(points)
    except Exception as exc:  # noqa: BLE001 - the manager must still open
        logger.warning("[KB_MANAGER] corpus read failed: %s", exc)
        return []


def corpus_rows() -> List[Dict[str, Any]]:
    """Every book in the corpus, with a quality verdict (``[]`` when empty)."""
    return sorted(
        _payload_books(_scroll_points()).values(),
        key=lambda r: (r["corpus"], r["title"].lower()),
    )


def bases(books: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One row per base (= corpus), with its books and totals."""
    books = corpus_rows() if books is None else books
    grouped: Dict[str, Dict[str, Any]] = {}
    for book in books:
        base = grouped.setdefault(
            book["corpus"],
            {
                "base": book["corpus"],
                "books": [],
                "chunks": 0,
                "real_chunks": 0,
                "real_books": 0,
            },
        )
        base["books"].append(book)
        base["chunks"] += book["chunks"]
        base["real_chunks"] += book["real_chunks"]
        base["real_books"] += 1 if book["is_real"] else 0
    return sorted(grouped.values(), key=lambda b: b["base"].lower())


def find(query: str, rows: Iterable[Dict[str, Any]], fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Case-insensitive substring match across ``fields`` (empty query = all)."""
    term = (query or "").strip().lower()
    if not term:
        return list(rows)
    found = []
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = " ".join(str(v) for v in value)
            if term in str(value).lower():
                found.append(row)
                break
    return found


# ---------------------------------------------------------------------------
# Bindings and namespaces
# ---------------------------------------------------------------------------


def grants() -> List[Dict[str, Any]]:
    """Every grant row (agent, value, scope, customer, project)."""
    from langgraph_crew_knowledge_base.assignments import grants as _grants

    return _grants()


def binding_rows(
    books: Optional[List[Dict[str, Any]]] = None,
    grants_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One row per grant, resolved to what it points at.

    A grant names a ``book`` id (the common case) or a ``category``. Both are
    resolved here so the manager can show a title instead of an id — and mark a
    grant whose title no longer exists in the corpus (`missing`), which is what
    a deleted summary leaves behind.
    """
    books = corpus_rows() if books is None else books
    rows = grants() if grants_rows is None else grants_rows

    by_book = {b["book_id"]: b for b in books}
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for book in books:
        for category in book.get("categories") or []:
            by_category[str(category)].append(book)

    out: List[Dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind") or "")
        value = str(row.get("value") or "")
        entry = {
            "agent_id": row.get("agent_id") or "",
            "kind": kind,
            "value": value,
            "scope": row.get("scope") or "",
            "customer": row.get("customer") or "",
            "project": row.get("project") or "",
            "assigned_by": row.get("assigned_by") or "",
            "missing": False,
            "title": value,
            "base": "",
        }
        if kind == "book":
            book = by_book.get(value)
            if book is None:
                entry["missing"] = True
            else:
                entry["title"] = book["title"]
                entry["base"] = book["corpus"]
        elif kind == "category":
            hits = by_category.get(value) or []
            entry["title"] = f"{value} ({len(hits)} book(s))"
            entry["missing"] = not hits
        out.append(entry)

    return sorted(
        out,
        key=lambda r: (
            _scope_rank(r["scope"]),
            r["customer"],
            r["project"],
            r["agent_id"],
            r["title"].lower(),
        ),
    )


def _scope_rank(scope: str) -> int:
    order = {"legacy": 0, "deployment": 1, "client": 2, "project": 3}
    return order.get(scope, 4)


def namespaces(bindings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every (customer, project) a grant varies, with what it changes.

    This is the "how does it differ per namespace" view: a row per namespace
    that has at least one project-scoped grant, counting the grants made for it
    and the grants its customer made for all its projects. A client-wide grant
    on its own is a row with an empty project (it belongs to no single project).
    Deployment grants reach everything and are reported once, separately.
    """
    scoped = [b for b in bindings if b["scope"] in ("client", "project")]

    # Namespaces that actually exist as a project.
    projects = {(b["customer"], b["project"]) for b in scoped if b["scope"] == "project"}
    # A client-only grant becomes its own "(whole client)" row, but only when
    # that customer has no project rows to fold into.
    customers_with_projects = {c for c, _ in projects}
    for binding in scoped:
        if binding["scope"] == "client" and binding["customer"] not in customers_with_projects:
            projects.add((binding["customer"], ""))

    out: List[Dict[str, Any]] = []
    for customer, project in sorted(projects):
        own = [
            b for b in scoped
            if b["customer"] == customer
            and (
                b["scope"] == "client"
                or (project and b["scope"] == "project" and b["project"] == project)
            )
        ]
        out.append(
            {
                "customer": customer,
                "project": project,
                "label": f"{customer}/{project}" if project else f"{customer} (whole client)",
                "grants": len(own),
                "project_specific": sum(1 for b in own if b["scope"] == "project"),
                "client_wide": sum(1 for b in own if b["scope"] == "client"),
                "agents": sorted({b["agent_id"] for b in own}),
            }
        )
    return out


def effective_for(
    agent_id: str,
    customer: str,
    project: str,
    bindings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Grants that reach ``agent_id`` in ``customer/project`` (the ladder)."""
    out = []
    for binding in bindings:
        if binding["agent_id"] != agent_id:
            continue
        scope = binding["scope"]
        if scope in ("deployment", "legacy"):
            out.append(binding)
        elif scope == "client" and binding["customer"] == customer:
            out.append(binding)
        elif scope == "project" and binding["customer"] == customer and binding["project"] == project:
            out.append(binding)
    return out


__all__ = [
    "bases",
    "binding_rows",
    "corpus_rows",
    "effective_for",
    "find",
    "grants",
    "namespaces",
]
