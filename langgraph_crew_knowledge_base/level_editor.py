#!/usr/bin/env python3
"""The level editor behind ``/knowledge`` — see the whole ladder, edit one rung.

What the operator asked for: see, laid out by LEVEL, what is configured for
which agent — knowledge bases, instructions, tools, policies — then pick a
level and change things there without leaving the chat.

The rendering is Rich (the same console ``/agents`` and ``/tools`` use); there
is no web service, so nothing new listens on a port.

Design rules this follows:

* **One truth.** Every number comes from the loaders the runtime uses —
  ``app.core.spec.loader`` for manifests, :mod:`ladder` for the merged
  knowledge view, :mod:`assignments` for the store. The editor reads; it does
  not keep its own idea of what is configured.
* **Say where it came from.** A title shows as ``file`` (manifest),
  ``store`` (runtime grant) or ``both``. Two sources disagreeing silently is
  how the old view lost half the picture.
* **Writes go through the existing paths** — ``assignments.assign`` /
  ``unassign`` for the store, the manifest writer for files. No third writer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from langgraph_crew_knowledge_base import ladder

logger = logging.getLogger(__name__)

LEVEL_ORDER = ladder.LEVELS


def _console() -> Console:
    return Console(highlight=False, soft_wrap=False)


# ── what is configured at a level ────────────────────────────────────────────


def _spec_entries(level: str, root: str) -> Dict[str, List[str]]:
    """Agents, skills, workflows and policies declared under one spec root."""
    import os

    out: Dict[str, List[str]] = {"agents": [], "skills": [], "workflows": [], "policies": []}
    for kind in ("agents", "skills", "workflows", "policies"):
        directory = os.path.join(root, kind)
        if not os.path.isdir(directory):
            continue
        try:
            names = sorted(
                os.path.splitext(name)[0]
                for name in os.listdir(directory)
                if name.endswith((".md", ".yaml", ".yml"))
            )
        except OSError as exc:
            logger.warning("[KNOWLEDGE_BASE] cannot list %s: %s", directory, exc)
            continue
        out[kind] = names
    return out


def level_inventory() -> List[Dict[str, Any]]:
    """Every level with its roots and what each root declares."""
    roots = ladder.spec_roots_by_level()
    inventory: List[Dict[str, Any]] = []
    for level in LEVEL_ORDER:
        paths = roots.get(level) or []
        entries: Dict[str, List[str]] = {
            "agents": [], "skills": [], "workflows": [], "policies": []
        }
        for root in paths:
            found = _spec_entries(level, root)
            for kind, names in found.items():
                for name in names:
                    if name not in entries[kind]:
                        entries[kind].append(name)
        inventory.append(
            {
                "level": level,
                "help": ladder.LEVEL_HELP.get(level, ""),
                "roots": paths,
                "entries": {k: sorted(v) for k, v in entries.items()},
            }
        )
    return inventory


def _agent_base_count(agent_id: str) -> int:
    resolved = ladder.resolve_agent(agent_id)
    return len(resolved["titles"])


# ── rendering ────────────────────────────────────────────────────────────────


def render_levels(console: Optional[Console] = None) -> str:
    """The stacked ladder: each level with its roots and what it configures."""
    console = console or _console()
    inventory = level_inventory()

    console.print()
    console.print("[bold cyan]CONFIGURATION LADDER[/bold cyan]  "
                  "[dim]most specific wins: core → custom → plugin → client → project[/dim]")
    console.print("[dim]" + "─" * 78 + "[/dim]")

    for entry in inventory:
        level = entry["level"]
        entries = entry["entries"]
        roots = entry["roots"]
        console.print()
        header = f"[bold green]{level.upper():<8}[/bold green] [dim]{entry['help']}[/dim]"
        console.print(header)
        if not roots:
            console.print("  [yellow]not present on this host[/yellow]")
            continue
        for root in roots[:3]:
            console.print(f"  [dim]root:[/dim] {root}")
        if len(roots) > 3:
            console.print(f"  [dim]… {len(roots) - 3} more root(s)[/dim]")

        for kind in ("agents", "skills", "workflows", "policies"):
            names = entries.get(kind) or []
            if not names:
                continue
            shown = ", ".join(names[:6])
            more = f" [dim]+{len(names) - 6}[/dim]" if len(names) > 6 else ""
            console.print(f"  [bold]{kind:<10}[/bold] [dim]{len(names):>2}[/dim]  {shown}{more}")

    console.print()
    console.print("[dim]`/knowledge levels <level>` for the full list · "
                  "`/knowledge agent <id>` for one agent's effective view[/dim]")
    return "levels"


def render_level(level: str, console: Optional[Console] = None) -> str:
    """One rung in full: what it configures, per kind, with paths."""
    console = console or _console()
    level = str(level or "").strip().lower()
    if level not in LEVEL_ORDER:
        return f"unknown level '{level}'. Use one of: {', '.join(LEVEL_ORDER)}"

    inventory = {e["level"]: e for e in level_inventory()}[level]
    console.print()
    console.print(f"[bold cyan]LEVEL {level.upper()}[/bold cyan]  "
                  f"[dim]{inventory['help']}[/dim]")
    console.print("[dim]" + "─" * 78 + "[/dim]")
    if not inventory["roots"]:
        console.print("[yellow]not present on this host[/yellow]")
        return "level"
    for root in inventory["roots"]:
        console.print(f"  [dim]root:[/dim] {root}")
    for kind in ("agents", "skills", "workflows", "policies"):
        names = inventory["entries"].get(kind) or []
        console.print()
        console.print(f"  [bold]{kind}[/bold] [dim]({len(names)})[/dim]")
        if not names:
            console.print("    [dim]none[/dim]")
            continue
        for name in names:
            if kind == "agents":
                console.print(f"    · {name}   [dim]{_agent_base_count(name)} base(s)[/dim]")
            else:
                console.print(f"    · {name}")
    return "level"


def render_agent(agent_id: str, console: Optional[Console] = None) -> str:
    """One agent across every level — the answer to 'what can it read, where from'."""
    console = console or _console()
    resolved = ladder.resolve_agent(agent_id)
    rungs = ladder.partition_by_level(agent_id)

    console.print()
    console.print(f"[bold cyan]{agent_id}[/bold cyan]")
    if resolved["manifest_path"]:
        console.print(f"  [dim]manifest:[/dim] {resolved['manifest_path']} "
                      f"[dim]({resolved['manifest_level']})[/dim]")
    else:
        console.print("  [dim]manifest:[/dim] [yellow]none found — no x-penta knowledge binding[/yellow]")
    console.print(f"  [dim]enabled:[/dim] {'yes' if resolved['enabled'] else 'no'}"
                  f"  [dim]chunks:[/dim] {resolved['chunks'] or 'default'}")

    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("level", style="green", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("value")
    table.add_column("source", style="dim", no_wrap=True)
    table.add_column("in corpus", no_wrap=True)
    stale_ids = set(resolved.get("stale") or [])
    rows = 0
    for level in LEVEL_ORDER:
        for entry in rungs.get(level) or []:
            value = str(entry.get("value"))
            if entry.get("kind") == "book":
                mark = "[red]missing[/red]" if value in stale_ids else "[green]yes[/green]"
            else:
                mark = "[dim]—[/dim]"
            table.add_row(
                level,
                str(entry.get("kind")),
                value,
                str(entry.get("source")),
                mark,
            )
            rows += 1
    if rows:
        console.print()
        console.print(table)
    else:
        console.print("  [yellow]nothing configured at any level — it searches the corpus per task[/yellow]")

    console.print()
    console.print(f"  [bold]effective now:[/bold] {len(resolved['effective'])} title(s)")
    stale = resolved.get("stale") or []
    if stale:
        # An agent bound to a deleted book searches for nothing and says
        # nothing. 12 of 30 grants were in this state when the editor was
        # written, so it is not a corner case.
        console.print(
            f"  [red bold]{len(stale)} binding(s) point at books that are no "
            f"longer in the corpus[/red bold] — this agent cannot read them. "
            f"`/knowledge agent {agent_id} prune` removes them."
        )
    return "agent"


def prune_agent(agent_id: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Remove this agent's bindings to books the corpus no longer holds."""
    resolved = ladder.resolve_agent(agent_id)
    live = ladder.live_books()
    if not live:
        return {
            "error": (
                "the corpus could not be read, so nothing can be called stale — "
                "refusing to delete bindings on a failed read"
            ),
            "removed": [],
        }
    stale = [t["book_id"] for t in resolved["titles"] if t["book_id"] not in live]
    if not stale:
        return {"removed": [], "message": "no stale binding"}
    if dry_run:
        return {"removed": [], "would_remove": stale, "dry_run": True}

    from langgraph_crew_knowledge_base.assignments import unassign

    removed: List[str] = []
    errors: List[str] = []
    for book_id in stale:
        for entry in resolved["store"]:
            if entry.get("kind") != "book" or entry.get("value") != book_id:
                continue
            result = unassign(
                agent_id,
                books=[book_id],
                scope=entry.get("scope") or "deployment",
                customer=entry.get("customer") or "",
                project=entry.get("project") or "",
            )
            if result.get("error"):
                errors.append(f"{book_id}: {result['error']}")
            else:
                removed.append(book_id)
    return {"removed": removed, "errors": errors}


def render_matrix(console: Optional[Console] = None) -> str:
    """Agents × levels — which agent is given which base, at which rung."""
    console = console or _console()
    from langgraph_crew_knowledge_base.assignments import grants

    agents = sorted({str(row.get("agent_id") or "") for row in grants()} - {""})
    data = ladder.matrix(agents)

    table = Table(title="Knowledge bases per level and agent", box=None,
                  show_header=True, header_style="bold")
    table.add_column("agent", no_wrap=True)
    for level in LEVEL_ORDER:
        table.add_column(level, justify="right")
    for agent_id in data["agents"]:
        counts = []
        for level in LEVEL_ORDER:
            cell = (data["cells"].get(level) or {}).get(agent_id) or {}
            books = [e for e in cell.get("entries") or [] if e.get("kind") == "book"]
            counts.append(str(len(books)) if books else "[dim]·[/dim]")
        table.add_row(agent_id, *counts)
    console.print()
    console.print(table)
    console.print("[dim]number = books configured at that level · "
                  "`/knowledge agent <id>` for the entries[/dim]")
    return "matrix"


__all__ = [
    "level_inventory",
    "render_agent",
    "render_level",
    "render_levels",
    "render_matrix",
]
