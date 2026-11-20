#!/usr/bin/env python3
"""The knowledge ladder: manifests are the base, the store is the extension.

Two sources feed what an agent may read, and they were never shown together:

``manifest``   ``x-penta.knowledge_base`` in ``agents/<name>.md`` at a spec
               level (core → custom → plugin → client → project). Declared in
               description language, versioned with the repo, inherited the
               same way every other agent setting is.
``store``      the ``assignment`` table. An *override layer*: it extends or
               narrows what the manifests say without touching a file, so an
               operator can change a binding at runtime.

The runtime has always merged the two (``node.py`` unions the manifest titles
with ``books_for_agent()``). The manager showed only the store, so a grant that
came from a manifest looked like it did not exist — and the same book then read
as "assigned twice" once somebody added it to the store.

This module resolves both into one answer per agent and says, per entry, where
it came from. Nothing here writes: writing lives in :mod:`assignments` (store)
and the manifest writer (files), which the editor calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Spec levels, least specific first. The runtime ladder is
#: ``core → custom → plugin → client → project`` and the most specific wins.
LEVELS: Tuple[str, ...] = ("core", "custom", "plugin", "client", "project")

#: How a level is named in a grant. ``custom`` is the deployment level: it
#: applies to every customer, which is why the store calls it ``deployment``.
STORE_SCOPE_FOR_LEVEL: Dict[str, str] = {
    "core": "deployment",
    "custom": "deployment",
    "plugin": "deployment",
    "client": "client",
    "project": "project",
}

#: Reverse direction, for showing a store row under a level.
LEVEL_FOR_STORE_SCOPE: Dict[str, str] = {
    "deployment": "custom",
    "client": "client",
    "project": "project",
    "legacy": "custom",
}

LEVEL_HELP: Dict[str, str] = {
    "core": "What PENTA ships (repo root). Read-only in a container.",
    "custom": "This deployment: custom/. Every customer and project.",
    "plugin": "A plugin's own description content (plugins/<name>/).",
    "client": "One customer: <client-vault>/agents/.",
    "project": "One project: <project>/agents/. Narrowest rung.",
}


def spec_roots_by_level() -> Dict[str, List[str]]:
    """Where each level lives on disk (``{}`` when the spec layer is absent)."""
    out: Dict[str, List[str]] = {}
    try:
        from app.core.spec.loader import spec_roots
    except Exception as exc:  # noqa: BLE001 - the editor must work without it
        logger.warning("[KNOWLEDGE_BASE] spec loader unavailable: %s", exc)
        return out
    try:
        for entry in spec_roots():
            out.setdefault(entry.level, []).append(str(entry.path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] spec roots failed: %s", exc)
    return out


def live_books() -> set:
    """Book ids that exist in the corpus right now (``set()`` if unreadable)."""
    try:
        from langgraph_crew_knowledge_base.tools import _live_catalog

        catalog = _live_catalog()
        return {
            book_id
            for corpus_data in (catalog.get("corpora") or {}).values()
            for book_id in (corpus_data.get("books") or {})
        }
    except Exception as exc:  # noqa: BLE001 - an empty set must not read as "all stale"
        logger.warning("[KNOWLEDGE_BASE] live catalog unavailable: %s", exc)
        return set()


def _manifest_entry(agent_id: str) -> Dict[str, Any]:
    """The ``x-penta.knowledge_base`` block and the level it was read from."""
    names = [agent_id]
    if agent_id.endswith("_agent"):
        names.append(agent_id[: -len("_agent")])
    else:
        names.append(f"{agent_id}_agent")

    try:
        from app.core.spec.loader import build_catalog
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] manifest lookup unavailable: %s", exc)
        return {}

    try:
        catalog = build_catalog()
        agent_entry = None
        for name in names:
            agent_entry = catalog.by_kind("agent").get(name)
            if agent_entry is not None:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KNOWLEDGE_BASE] catalog build failed: %s", exc)
        return {}

    if agent_entry is None:
        return {}

    manifest = getattr(agent_entry, "manifest", None)
    config = getattr(getattr(manifest, "x_penta", None), "knowledge_base", None)
    if config is None:
        return {
            "agent_id": agent_id,
            "level": getattr(agent_entry, "level", ""),
            "path": getattr(agent_entry, "path", ""),
            "titles": [],
            "categories": [],
            "tags": [],
            "chunks": None,
            "enabled": True,
            "declared": False,
        }
    return {
        "agent_id": agent_id,
        "level": getattr(agent_entry, "level", ""),
        "path": getattr(agent_entry, "path", ""),
        "titles": list(getattr(config, "titles", []) or []),
        "categories": list(getattr(config, "categories", []) or []),
        "tags": list(getattr(config, "tags", []) or []),
        "chunks": getattr(config, "chunks", None),
        "enabled": bool(getattr(config, "enabled", True)),
        "declared": True,
    }


def _store_entries(agent_id: str) -> List[Dict[str, Any]]:
    """Store rows of one agent, tagged with the level they belong to."""
    from langgraph_crew_knowledge_base.assignments import grants

    out: List[Dict[str, Any]] = []
    for row in grants():
        if str(row.get("agent_id") or "") != agent_id:
            continue
        out.append(
            {
                "kind": row.get("kind"),
                "value": row.get("value"),
                "scope": row.get("scope"),
                "level": LEVEL_FOR_STORE_SCOPE.get(str(row.get("scope")), "custom"),
                "customer": row.get("customer"),
                "project": row.get("project"),
                "assigned_by": row.get("assigned_by"),
                "created_at": row.get("created_at"),
            }
        )
    return out


def resolve_agent(
    agent_id: str,
    *,
    customer: str = "",
    project: str = "",
) -> Dict[str, Any]:
    """One agent: what it may read, and where each part of that came from.

    ``source`` per title is ``manifest`` (declared in a file), ``store``
    (granted at runtime) or ``both`` — the last one is the interesting case:
    the two sources agree, which is safe but worth knowing about.
    """
    from langgraph_crew_knowledge_base.assignments import books_for_agent

    manifest = _manifest_entry(agent_id)
    manifest_titles = list(manifest.get("titles") or [])

    store = _store_entries(agent_id)
    store_titles = [e["value"] for e in store if e.get("kind") == "book"]

    try:
        effective = books_for_agent(agent_id, customer=customer, project=project)
    except Exception as exc:  # noqa: BLE001 - report, never crash the view
        logger.warning(
            "[KNOWLEDGE_BASE] ladder read failed for %s: %s", agent_id, exc
        )
        effective = []

    # Effective = what a run of this agent actually sees. Store titles resolved
    # through the ladder are folded in; manifest titles are unconditional.
    sources: Dict[str, str] = {}
    for title in manifest_titles:
        sources[title] = "manifest"
    for title in effective:
        sources[title] = "both" if title in sources else "store"
    for title in store_titles:
        sources.setdefault(title, "store")

    titles = sorted(sources)
    live = live_books()
    stale = [t for t in titles if live and t not in live]
    return {
        "agent_id": agent_id,
        "manifest": manifest,
        "manifest_level": manifest.get("level") or "",
        "manifest_path": manifest.get("path") or "",
        "manifest_declared": bool(manifest.get("declared")),
        "store": store,
        "effective": effective,
        "titles": [
            {
                "book_id": t,
                "source": sources[t],
                "live": (t in live) if live else None,
            }
            for t in titles
        ],
        "stale": stale,
        "enabled": manifest.get("enabled", True),
        "chunks": manifest.get("chunks"),
        "categories": sorted(
            set(manifest.get("categories") or [])
            | {e["value"] for e in store if e.get("kind") == "category"}
        ),
        "tags": sorted(
            set(manifest.get("tags") or [])
            | {e["value"] for e in store if e.get("kind") == "tag"}
        ),
    }


def partition_by_level(agent_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """One agent's grants grouped by level — the ladder, one rung per row.

    Manifest entries live at the level of the manifest that declared them;
    store entries at the level their scope maps to.
    """
    resolved = resolve_agent(agent_id)
    out: Dict[str, List[Dict[str, Any]]] = {level: [] for level in LEVELS}

    manifest = resolved["manifest"]
    level = resolved["manifest_level"] or "custom"
    for title in manifest.get("titles") or []:
        out.setdefault(level, []).append({"kind": "book", "value": title, "source": "manifest"})
    for category in manifest.get("categories") or []:
        out.setdefault(level, []).append(
            {"kind": "category", "value": category, "source": "manifest"}
        )
    for tag in manifest.get("tags") or []:
        out.setdefault(level, []).append({"kind": "tag", "value": tag, "source": "manifest"})

    for entry in resolved["store"]:
        rung = entry.get("level") or "custom"
        out.setdefault(rung, []).append(
            {
                "kind": entry.get("kind"),
                "value": entry.get("value"),
                "source": "store",
                "scope": entry.get("scope"),
                "customer": entry.get("customer"),
                "project": entry.get("project"),
                "assigned_by": entry.get("assigned_by"),
            }
        )
    return out


def matrix(
    agent_ids: Iterable[str],
    *,
    customer: str = "",
    project: str = "",
) -> Dict[str, Any]:
    """Agents × levels × bases — the full picture the editor shows.

    ``cells[level][agent]`` lists the books configured at that rung, so an
    operator can see at one glance which agent is given which base where.
    """
    levels: Dict[str, Dict[str, Dict[str, Any]]] = {
        level: {} for level in LEVELS
    }
    agents: List[str] = []
    for agent_id in agent_ids:
        agent_id = str(agent_id or "").strip()
        if not agent_id or agent_id in agents:
            continue
        agents.append(agent_id)
        resolved = resolve_agent(agent_id, customer=customer, project=project)
        rungs = partition_by_level(agent_id)
        for level, entries in rungs.items():
            levels.setdefault(level, {})[agent_id] = {
                "entries": entries,
                "manifest_path": resolved["manifest_path"]
                if resolved["manifest_level"] == level
                else "",
            }
    return {
        "levels": list(LEVELS),
        "level_help": LEVEL_HELP,
        "agents": sorted(agents),
        "cells": levels,
        "active_scope": {"customer": customer, "project": project},
    }


__all__ = [
    "LEVELS",
    "LEVEL_HELP",
    "LEVEL_FOR_STORE_SCOPE",
    "STORE_SCOPE_FOR_LEVEL",
    "live_books",
    "matrix",
    "partition_by_level",
    "resolve_agent",
    "spec_roots_by_level",
]
