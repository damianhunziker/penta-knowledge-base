#!/usr/bin/env python3
"""``/knowledge`` — the manager as a chat-CLI command.

The interactive script (``scripts/knowledge_manager.py``) is the full tool. From
inside a chat there is no separate terminal, so the same views are reachable as
one command. The renderers and the write path are the *script's* — this module
only parses the sub-command and returns the message the CLI prints.

    /knowledge                     overview (bases, bindings, namespaces)
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
    lines = [
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
