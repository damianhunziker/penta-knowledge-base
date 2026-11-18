#!/usr/bin/env python3
"""Where the knowledge base may write its own state.

The plugin lives inside the repo, and inside the container the repo is a
**read-only bind** (``..:/workspace:ro``). A default of
``<plugin>/knowledge_base.db`` therefore worked on the host and failed in
production with the least helpful message SQLite can produce::

    attempt to write a readonly database

That message names neither the file nor the reason, so one assignment attempt
disappeared into a filesystem question. This module is the single answer to
"where may this plugin write", used by the assignment store and the corpus
manifest alike, and it **verifies writability by writing** instead of trusting
``os.access`` (a read-only bind mount can report writable for the owner bits).

Resolution order (first writable wins):

1. ``KNOWLEDGE_DB_PATH`` / the configured path, when its directory is writable
2. ``$PENTA_STATE_DIR`` (an operator-set state directory)
3. ``<repo>/db`` — a ``rw`` bind in the compose stack (single source of truth
   for repo-level SQLite files)
4. ``~/.penta/knowledge-base`` — last resort, never inside the repo

Host and container therefore agree on the file, and a deployment that keeps
state elsewhere only has to set ``PENTA_STATE_DIR``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent

#: Fallback file name used in every candidate directory.
DB_FILENAME = "knowledge_base.db"


def _repo_root() -> Optional[Path]:
    try:
        from app.core.spec.loader import _repo_root as _spec_repo_root

        return Path(_spec_repo_root())
    except Exception:  # noqa: BLE001 - not imported in some test contexts
        pass
    # ``plugins/knowledge-base/langgraph_crew_knowledge_base/`` → repo
    candidate = PLUGIN_DIR.parent.parent
    return candidate if (candidate / "app").is_dir() else None


def writable_dir(directory: Path) -> bool:
    """True when a file can actually be created in ``directory``.

    A real write is the only reliable test: ``os.access`` answers from the
    inode's permission bits, not from the mount options, so it says "yes" on a
    read-only bind mount and the next ``open()`` fails.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = None
    try:
        fd, probe = tempfile.mkstemp(prefix=".kb-probe-", dir=str(directory))
        os.close(fd)
        return True
    except OSError:
        return False
    finally:
        if probe:
            try:
                os.unlink(probe)
            except OSError:
                pass


def candidate_dirs(configured: Optional[str] = None) -> List[Path]:
    """Directories to try, most specific first."""
    out: List[Path] = []

    def _add(value: str) -> None:
        text = (value or "").strip()
        if not text:
            return
        path = Path(text).expanduser()
        if path not in out:
            out.append(path)

    if configured:
        path = Path(str(configured)).expanduser()
        # A configured path may be the file itself; its directory is what we
        # have to be able to write in.
        _add(str(path.parent if path.suffix else path))
    _add(os.getenv("PENTA_STATE_DIR", ""))
    repo = _repo_root()
    if repo is not None:
        _add(str(repo / "db"))
    _add(str(PLUGIN_DIR))
    _add(str(Path.home() / ".penta" / "knowledge-base"))
    return out


def resolve_db_path(configured: Optional[str] = None, *, filename: str = DB_FILENAME) -> Path:
    """Return the first writable database path, or the last candidate.

    Returning a path even when nothing is writable keeps this function total:
    the caller fails on ``connect`` with the filename in the message instead of
    raising during import.
    """
    # An explicitly configured FILE path that is already writable is honoured
    # as-is (an operator who pointed at a specific file meant that file).
    text = (configured or "").strip()
    if text:
        explicit = Path(text).expanduser()
        if explicit.suffix and writable_dir(explicit.parent):
            return explicit

    for directory in candidate_dirs(configured):
        if writable_dir(directory):
            return directory / filename

    repo = _repo_root()
    fallback = (repo / "db" / filename) if repo else (PLUGIN_DIR / filename)
    logger.warning(
        "[KNOWLEDGE_BASE] no writable state directory found (tried %s) — "
        "falling back to %s; writes will fail until one is writable",
        ", ".join(str(d) for d in candidate_dirs(configured)),
        fallback,
    )
    return fallback


__all__ = [
    "DB_FILENAME",
    "candidate_dirs",
    "resolve_db_path",
    "writable_dir",
]
