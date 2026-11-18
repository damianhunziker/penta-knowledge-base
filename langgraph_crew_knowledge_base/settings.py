#!/usr/bin/env python3
"""Environment-driven settings for the knowledge base plugin.

The plugin owns its own configuration file, exactly like the web-discovery
plugin owns ``plugin.env``:

    plugins/knowledge-base/plugin.env.example   (versioned template)
    plugins/knowledge-base/plugin.env           (local override, gitignored)

Precedence: **process environment > plugin.env > plugin.env.example > default**.
A malformed value logs a warning and falls back to the default — a typo in the
config file must never break a run.

Settings
--------

``KNOWLEDGE_INJECT_AFTER_TOOLS``
    Inject the next knowledge block after this many tool calls. Default ``13``.
    ``0`` disables the mid-run injection entirely (cold injection at task
    start still happens).

``KNOWLEDGE_INJECT_CHUNKS``
    How many chunks one mid-run injection carries. Default ``3``.

``KNOWLEDGE_TASK_CHUNKS``
    How many chunks are injected when a task starts (cold injection).
    Default ``3``. A task may override this per task with ``knowledge_chunks``.

``KNOWLEDGE_INJECT_MAX_PER_RUN``
    Safety cap: how often the mid-run hook may fire for one agent run.
    Default ``6``. Guards against a long tool loop drowning the context.

``KNOWLEDGE_INJECT_MIN_SCORE``
    Minimum retrieval score for a chunk to be injected. Default ``0.35``.

``KNOWLEDGE_INJECT_BUDGET_TOKENS``
    Token ceiling for ONE injection block. Default ``900``.

``DISABLE_KNOWLEDGE_DISCOVERY``
    ``true``/``1``/``yes`` switches the whole knowledge discovery off.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

#: ``<plugin>/plugin.env`` lives two levels above this module.
PLUGIN_DIR = Path(__file__).resolve().parent.parent

#: Versioned template (always present, holds the documented defaults).
ENV_EXAMPLE = PLUGIN_DIR / "plugin.env.example"

#: Local override, gitignored.
ENV_FILE = PLUGIN_DIR / "plugin.env"


#: Defaults are part of the contract — the docs and the tests pin them.
DEFAULT_INJECT_AFTER_TOOLS = 13
DEFAULT_INJECT_CHUNKS = 3
DEFAULT_TASK_CHUNKS = 3
DEFAULT_MAX_PER_RUN = 6
DEFAULT_MIN_SCORE = 0.35
DEFAULT_BUDGET_TOKENS = 900


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a ``.env``-style file (no interpolation, no exec)."""
    values: Dict[str, str] = {}
    try:
        if not path.exists():
            return values
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    except OSError as exc:  # noqa: BLE001 - config must never break a run
        logger.warning("[KNOWLEDGE_BASE] could not read %s: %s", path, exc)
    return values


def _file_values() -> Dict[str, str]:
    """Merged values from the two plugin files (example first, local wins)."""
    merged = _parse_env_file(ENV_EXAMPLE)
    merged.update(_parse_env_file(ENV_FILE))
    return merged


def _raw(name: str, default: str = "") -> str:
    """Resolve one value: process env > plugin.env > plugin.env.example."""
    env_value = os.environ.get(name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    return _file_values().get(name, default)


def _int(name: str, default: int) -> int:
    """Read an int, falling back on anything unusable.

    A typo in ``plugin.env`` must never break a run, so a non-numeric value
    logs a warning and yields the default.
    """
    raw = _raw(name)
    if raw == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%r is not an integer, using %s", name, raw, default
        )
        return default
    if value < 0:
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%s is negative, using %s", name, value, default
        )
        return default
    return value


def _float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw == "":
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%r is not a number, using %s", name, raw, default
        )
        return default
    if value < 0:
        return default
    return value


def _bool(name: str, default: bool = False) -> bool:
    raw = _raw(name)
    if raw == "":
        return default
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class KnowledgeSettings:
    """Resolved settings for one run (read fresh on every call)."""

    inject_after_tools: int = DEFAULT_INJECT_AFTER_TOOLS
    inject_chunks: int = DEFAULT_INJECT_CHUNKS
    task_chunks: int = DEFAULT_TASK_CHUNKS
    max_per_run: int = DEFAULT_MAX_PER_RUN
    min_score: float = DEFAULT_MIN_SCORE
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    disabled: bool = False
    enabled_by_default: bool = False

    @property
    def mid_run_enabled(self) -> bool:
        """Mid-run injection is on unless switched off or set to 0."""
        return not self.disabled and self.inject_after_tools > 0


def load_settings() -> KnowledgeSettings:
    """Read the settings (process env > plugin.env > example > default)."""
    return KnowledgeSettings(
        inject_after_tools=_int(
            "KNOWLEDGE_INJECT_AFTER_TOOLS", DEFAULT_INJECT_AFTER_TOOLS
        ),
        inject_chunks=_int("KNOWLEDGE_INJECT_CHUNKS", DEFAULT_INJECT_CHUNKS),
        task_chunks=_int("KNOWLEDGE_TASK_CHUNKS", DEFAULT_TASK_CHUNKS),
        max_per_run=_int("KNOWLEDGE_INJECT_MAX_PER_RUN", DEFAULT_MAX_PER_RUN),
        min_score=_float("KNOWLEDGE_INJECT_MIN_SCORE", DEFAULT_MIN_SCORE),
        budget_tokens=_int(
            "KNOWLEDGE_INJECT_BUDGET_TOKENS", DEFAULT_BUDGET_TOKENS
        ),
        disabled=_bool("DISABLE_KNOWLEDGE_DISCOVERY", False),
        enabled_by_default=_bool("ENABLE_KNOWLEDGE_DISCOVERY_DEFAULT", False),
    )


def chunks_for_task(task: object, settings: KnowledgeSettings | None = None) -> int:
    """How many chunks to inject when ``task`` starts.

    Priority: the task's own ``knowledge_chunks`` → ``KNOWLEDGE_TASK_CHUNKS``.
    A task value of ``0`` means "no literature for this task".
    """
    settings = settings or load_settings()
    default = settings.task_chunks
    if not isinstance(task, dict):
        return default
    raw = task.get("knowledge_chunks")
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "DEFAULT_INJECT_AFTER_TOOLS",
    "DEFAULT_INJECT_CHUNKS",
    "DEFAULT_MAX_PER_RUN",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_TASK_CHUNKS",
    "ENV_EXAMPLE",
    "ENV_FILE",
    "KnowledgeSettings",
    "PLUGIN_DIR",
    "chunks_for_task",
    "load_settings",
]
