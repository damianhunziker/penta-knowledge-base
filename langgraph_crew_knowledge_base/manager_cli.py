#!/usr/bin/env python3
"""``/knowledge`` — the manager as a chat-CLI command.

The interactive script (``scripts/knowledge_manager.py``) is the full tool. From
inside a chat there is no separate terminal, so the same views are reachable as
one command. The renderers and the write path are the *script's* — this module
only parses the sub-command and returns the message the CLI prints.

    /knowledge                     overview (ladder, bases, bindings, namespaces)
    /knowledge levels              every level and what it configures
    /knowledge levels <level>      one level in full (agents, skills, workflows, policies)
    /knowledge matrix              which agent gets which base, per level
    /knowledge agent <id>          one agent: every level, source per binding
    /knowledge agent <id> prune    drop bindings to books no longer in the corpus
    /knowledge raw [book_id]       which books have a stored original (and which lost it)
    /knowledge raw backfill        archive sources still on disk + tag their chunks
    /knowledge bases [query]       the bases and their books
    /knowledge bindings [query]    who may read what, at which scope
    /knowledge namespaces          where the bindings differ
    /knowledge effective <agent> <customer>/<project>
    /knowledge assign <agent> <book_id> [scope] [customer] [project]
    /knowledge unassign <agent> <book_id> [scope] [customer] [project]

A wider-than-project assignment (client, deployment) is refused here: the chat
cannot ask for the authorisation note that a write beyond the current project
requires. Use the script, or ``knowledge_base___grant_access`` with
``authorized_by``/``authorization_note``.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any, Dict

#: Scopes the command may write without an authorisation note.
_LOCAL_SCOPES = frozenset({"project"})


def _manager_module():
    """The interactive script's module (one implementation, two front-ends)."""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import knowledge_manager  # noqa: PLC0415 - path set above

    return knowledge_manager


def _capture(fn, *args, **kwargs) -> str:
    """Run a renderer and return what it printed (ANSI stripped)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args, **kwargs)
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    return text.strip("\n")


def _overview(manager) -> str:
    from langgraph_crew_knowledge_base import level_editor, ladder

    levels = ladder.spec_roots_by_level()
    configured = [name for name in ladder.LEVELS if levels.get(name)]
    lines = [
        f"Ladder: {' → '.join(configured) or '(spec layer unavailable)'} "
        f"· /knowledge levels for detail · /knowledge agent <id>",
        "",
        f"Knowledge bases: {len(manager.bases)} · books: {len(manager.books)} "
        f"· bindings: {len(manager.bindings)}",
        f"store: {manager.db_path}",
        "",
    ]
    lines.append(_capture(_manager_module().show_bases, manager))
    lines.append("")
    lines.append(_capture(_manager_module().show_bindings, manager))
    if manager.namespaces:
        lines.append("")
        lines.append(_capture(_manager_module().show_namespaces, manager))
    return "\n".join(lines)


def knowledge_cli(args: str = "") -> Dict[str, Any]:
    """Entry point for ``/knowledge`` (handler contract: args -> dict)."""
    km = _manager_module()
    try:
        manager = km.Manager()
    except Exception as exc:  # noqa: BLE001 - a command must report, not crash
        return {"success": False, "message": f"knowledge manager unavailable: {exc}"}

    parts = (args or "").strip().split()
    if not parts:
        return {"success": True, "message": _overview(manager)}

    cmd = parts[0].lower()
    rest = parts[1:]

    try:
        if cmd in ("levels", "level", "ladder"):
            from langgraph_crew_knowledge_base import level_editor

            if rest:
                return {"success": True, "message": _capture(level_editor.render_level, rest[0])}
            return {"success": True, "message": _capture(level_editor.render_levels)}
        if cmd == "matrix":
            from langgraph_crew_knowledge_base import level_editor

            return {"success": True, "message": _capture(level_editor.render_matrix)}
        if cmd in ("agent", "agents"):
            from langgraph_crew_knowledge_base import level_editor

            if not rest:
                return {
                    "success": False,
                    "message": "usage: /knowledge agent <agent_id> | "
                               "/knowledge agent <agent_id> prune",
                }
            agent_id = rest[0]
            if not _known_agent(agent_id):
                return {
                    "success": False,
                    "message": (
                        f"unknown agent '{agent_id}' — neither a manifest nor a grant "
                        f"mentions it. `/knowledge matrix` lists every agent that has "
                        f"a binding."
                    ),
                }
            if len(rest) > 1 and rest[1].lower() == "prune":
                result = level_editor.prune_agent(agent_id)
                if result.get("error"):
                    return {"success": False, "message": result["error"]}
                removed = result.get("removed") or []
                if not removed:
                    return {"success": True, "message": result.get("message", "nothing to remove")}
                lines = [f"removed {len(removed)} stale binding(s) from {agent_id}:"]
                lines += [f"  ✗ {b}" for b in removed]
                if result.get("errors"):
                    lines += ["", "could not remove:"] + [f"  ! {e}" for e in result["errors"]]
                return {"success": True, "message": "\n".join(lines)}
            return {"success": True, "message": _capture(level_editor.render_agent, agent_id)}
        if cmd == "raw":
            from langgraph_crew_knowledge_base.raw_store import backfill, raw_root

            if not rest:
                return {"success": True, "message": _raw_status()}
            if rest[0].lower() == "backfill":
                report = backfill(rest[1] if len(rest) > 1 else "")
                lines = [
                    f"stored: {raw_root()}",
                    f"archived {report.get('archived', 0)} · tagged {report.get('tagged', 0)} chunks "
                    f"· missing {report.get('missing', 0)}",
                ]
                for name, entry in (report.get("books") or {}).items():
                    if entry.get("ok"):
                        lines.append(f"  ✓ {name}")
                    else:
                        lines.append(f"  ✗ {name}: {entry.get('reason')}")
                return {"success": report.get("ok", False), "message": "\n".join(lines)}
            return {"success": True, "message": _capture(_raw_one, rest[0])}
        if cmd in ("bases", "books", "list"):
            return {"success": True, "message": _capture(km.show_bases, manager, " ".join(rest))}
        if cmd in ("bindings", "grants"):
            return {"success": True, "message": _capture(km.show_bindings, manager, " ".join(rest))}
        if cmd == "namespaces":
            return {"success": True, "message": _capture(km.show_namespaces, manager)}
        if cmd == "search":
            if not rest:
                return {"success": False, "message": "usage: /knowledge search <query>"}
            return {"success": True, "message": _capture(km.show_search, manager, " ".join(rest))}
        if cmd == "effective":
            if not rest:
                return {"success": False, "message": "usage: /knowledge effective <agent> <customer>/<project>"}
            agent = rest[0]
            namespace = rest[1] if len(rest) > 1 else ""
            customer, _, project = namespace.partition("/")
            rows = manager.effective(agent, customer, project)
            if not rows:
                return {
                    "success": True,
                    "message": f"{agent} has no bound base in {namespace or '(no namespace)'} "
                               "— it searches the corpus per task.",
                }
            lines = [f"{agent} in {namespace or '(no namespace)'}:"]
            for row in rows:
                lines.append(f"  {row['scope']:<12} {row['title']}  ({row['value']})")
            return {"success": True, "message": "\n".join(lines)}

        if cmd in ("assign", "unassign"):
            return _write(km, manager, cmd, rest)

        return {
            "success": False,
            "message": (
                f"unknown sub-command '{cmd}'. Use: bases | bindings | namespaces | "
                "effective | search | assign | unassign"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"{type(exc).__name__}: {exc}"}


def _write(km, manager, cmd: str, rest) -> Dict[str, Any]:
    """``assign`` / ``unassign``: agent, book, optional scope + namespace."""
    if len(rest) < 2:
        return {
            "success": False,
            "message": f"usage: /knowledge {cmd} <agent> <book_id> [scope] [customer] [project]",
        }
    agent, book = rest[0], rest[1]
    scope = rest[2] if len(rest) > 2 else "project"
    customer = rest[3] if len(rest) > 3 else ""
    project = rest[4] if len(rest) > 4 else ""

    if cmd == "assign" and scope not in _LOCAL_SCOPES:
        return {
            "success": False,
            "message": (
                f"'{scope}' reaches beyond the current project and needs an "
                "authorisation note. Use the manager script "
                "(python plugins/knowledge-base/scripts/knowledge_manager.py) or "
                "knowledge_base___grant_access with authorized_by + authorization_note."
            ),
        }

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if cmd == "assign":
            ok = km.do_assign(manager, agent, book, scope, customer, project)
        else:
            ok = km.do_unassign(manager, agent, book, scope, customer, project)
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue()).strip()
    return {"success": bool(ok), "message": text or "(no output)"}


__all__ = ["knowledge_cli"]


def _raw_status() -> str:
    """Which books still have their original, and which lost it."""
    import json

    from langgraph_crew_knowledge_base.raw_store import raw_root

    from langgraph_crew_knowledge_base.tools import _live_catalog

    store = raw_root()
    catalog = _live_catalog()
    lines = [f"RAW STORE: {store}", ""]
    have, missing = [], []
    for corpus, data in (catalog.get("corpora") or {}).items():
        for book_id, info in (data.get("books") or {}).items():
            ref = (info or {}).get("raw_ref") or ""
            (have if ref else missing).append((book_id, ref))
    lines.append(f"  {len(have)} book(s) with a stored original")
    for book_id, ref in sorted(have)[:20]:
        lines.append(f"    ✓ {book_id}")
    if len(have) > 20:
        lines.append(f"    … {len(have) - 20} more")
    lines.append("")
    lines.append(f"  {len(missing)} book(s) without one (chunks exist, citations cannot link)")
    for book_id, _ in sorted(missing)[:20]:
        lines.append(f"    ✗ {book_id}")
    if len(missing) > 20:
        lines.append(f"    … {len(missing) - 20} more")
    lines.append("")
    lines.append("  `/knowledge raw backfill` archives sources still on disk")
    return "\n".join(lines)


def _raw_one(book_id: str) -> str:
    from langgraph_crew_knowledge_base.tools import _find_book

    found = _find_book(book_id)
    if not found:
        return f"book not found: {book_id}"
    ref = found.get("raw_ref") or ""
    if not ref:
        return (
            f"{book_id}: no stored original\n"
            f"  source_ref: {found.get('source_ref') or '(none)'}\n"
            f"  → `/knowledge raw backfill {book_id}` tries to archive it"
        )
    from langgraph_crew_knowledge_base.raw_store import link_for

    link = link_for(ref)
    status = "available" if link.get("available") else f"unavailable ({link.get('reason')})"
    return f"{book_id}: {ref}\n  {status} · {link.get('bytes', 0)} bytes"


def _known_agent(agent_id: str) -> bool:
    """True when a manifest or a grant mentions ``agent_id`` (either spelling)."""
    from langgraph_crew_knowledge_base.assignments import grants
    from langgraph_crew_knowledge_base.ladder import _manifest_entry

    wanted = {agent_id, agent_id.removesuffix("_agent"), f"{agent_id}_agent"}
    try:
        if {str(row.get("agent_id") or "") for row in grants()} & wanted:
            return True
    except Exception as exc:  # noqa: BLE001 - fall through to the manifest check
        logger.warning("[KNOWLEDGE_BASE] grant lookup failed: %s", exc)
    return bool(_manifest_entry(agent_id))
