#!/usr/bin/env python3
"""Environment-driven settings for the knowledge base plugin.

All values come from the process environment, so an operator can tune the
behaviour without touching code. Every setting has a documented default and a
safe fallback: a malformed value never breaks a run, it just falls back.

Environment variables
---------------------

``KNOWLEDGE_INJECT_AFTER_TOOLS``
    Inject the next knowledge block after this many tool calls. Default ``13``.
    ``0`` disables the mid-run injection entirely (cold injection at task
    start still happens).

``KNOWLEDGE_INJECT_CHUNKS``
    How many chunks one injection carries. Default ``3``.

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

logger = logging.getLogger(__name__)

#: Defaults are part of the contract — the docs and the tests pin them.
DEFAULT_INJECT_AFTER_TOOLS = 13
DEFAULT_INJECT_CHUNKS = 3
DEFAULT_TASK_CHUNKS = 3
DEFAULT_MAX_PER_RUN = 6
DEFAULT_MIN_SCORE = 0.35
DEFAULT_BUDGET_TOKENS = 900


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an int from the environment, falling back on anything unusable.

    A typo in ``.env`` must never break a run, so a non-numeric value logs a
    warning and yields the default.
    """
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%r is not an integer, using %s", name, raw, default
        )
        return default
    if value < minimum:
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%s is below %s, using %s", name, value, minimum, default
        )
        return default
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "[KNOWLEDGE_BASE] %s=%r is not a number, using %s", name, raw, default
        )
        return default
    if value < minimum:
        return default
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
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

    @property
    def mid_run_enabled(self) -> bool:
        """Mid-run injection is on unless switched off or set to 0."""
        return not self.disabled and self.inject_after_tools > 0


def load_settings() -> KnowledgeSettings:
    """Read the settings from the environment (callers cache nothing)."""
    return KnowledgeSettings(
        inject_after_tools=_int_env(
            "KNOWLEDGE_INJECT_AFTER_TOOLS", DEFAULT_INJECT_AFTER_TOOLS
        ),
        inject_chunks=_int_env("KNOWLEDGE_INJECT_CHUNKS", DEFAULT_INJECT_CHUNKS),
        task_chunks=_int_env("KNOWLEDGE_TASK_CHUNKS", DEFAULT_TASK_CHUNKS),
        max_per_run=_int_env("KNOWLEDGE_INJECT_MAX_PER_RUN", DEFAULT_MAX_PER_RUN),
        min_score=_float_env("KNOWLEDGE_INJECT_MIN_SCORE", DEFAULT_MIN_SCORE),
        budget_tokens=_int_env(
            "KNOWLEDGE_INJECT_BUDGET_TOKENS", DEFAULT_BUDGET_TOKENS
        ),
        disabled=_bool_env("DISABLE_KNOWLEDGE_DISCOVERY", False),
    )


def chunks_for_task(task: object, settings: KnowledgeSettings | None = None) -> int:
    """How many chunks to inject when ``task`` starts.

    Priority: the task's own ``knowledge_chunks`` → ``KNOWLEDGE_TASK_CHUNKS``.
    A task value is clamped to the same minimum (0 means "none for this task").
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
    "KnowledgeSettings",
    "chunks_for_task",
    "load_settings",
]
