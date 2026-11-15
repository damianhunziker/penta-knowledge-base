#!/usr/bin/env python3
"""Retrieval → injection blocks for the knowledge base plugin.

Two entry points, both budget- and count-limited by
:mod:`langgraph_crew_knowledge_base.settings`:

``build_task_block(task, state)``
    Cold injection: the chunks a task gets when it starts.

``build_mid_run_block(query, state, seen)``
    Mid-run injection: the chunks an agent gets after
    ``KNOWLEDGE_INJECT_AFTER_TOOLS`` tool calls, excluding what it already saw.

Both return a rendered ``str`` (or ``""``), never raise: a knowledge base that
is unreachable must not break the run.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Sequence

from langgraph_crew_knowledge_base.render import Chunk, Reference, render_chunk
from langgraph_crew_knowledge_base.settings import (
    KnowledgeSettings,
    chunks_for_task,
    load_settings,
)

logger = logging.getLogger(__name__)

#: Markers the executor uses to recognise an injected block and avoid doubles.
TASK_BLOCK_HEADER = "KNOWLEDGE DISCOVERY RESULT"
MID_RUN_BLOCK_HEADER = "KNOWLEDGE BASE INFO"


def _search(query: str, limit: int, settings: KnowledgeSettings) -> List[Chunk]:
    """Run a corpus search and normalise the answer to ``Chunk`` objects.

    Uses the plugin's own tool wrapper when available (it owns the Qdrant
    connection details); anything unusable yields ``[]``.
    """
    if not (query or "").strip() or limit <= 0:
        return []
    try:
        from langgraph_crew_knowledge_base.tools import knowledge_base___search

        raw = knowledge_base___search(query=query, corpora=None, tier=None, top_k=limit)
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:  # noqa: BLE001 - retrieval is best effort
        logger.debug("[KNOWLEDGE_BASE] search failed", exc_info=True)
        return []

    chunks: List[Chunk] = []
    for hit in payload.get("results") or []:
        chunk = _to_chunk(hit)
        if chunk is None:
            continue
        if chunk.score and chunk.score < settings.min_score:
            continue
        chunks.append(chunk)
    return chunks


def _to_chunk(hit: Dict[str, Any]) -> Chunk | None:
    """Build a ``Chunk`` from one search hit (``None`` when malformed)."""
    if not isinstance(hit, dict):
        return None
    kb_ref = hit.get("kb_ref") or hit.get("reference") or ""
    if not kb_ref:
        return None
    ref = Reference(
        kb_ref=kb_ref,
        title=hit.get("title") or "",
        authors=tuple(hit.get("authors") or ()),
        year=hit.get("year"),
        page_from=hit.get("page_from"),
        page_to=hit.get("page_to"),
        section_path=hit.get("section_path") or "",
        tier=hit.get("tier") or "unspecified",
        book_id=hit.get("book_id") or "",
        corpus=hit.get("corpus") or "",
        isbn=hit.get("isbn"),
        doi=hit.get("doi"),
    )
    return Chunk(
        text=hit.get("text") or "",
        reference=ref,
        score=float(hit.get("score") or 0.0),
    )


def _trim_to_budget(chunks: Sequence[Chunk], budget_tokens: int) -> List[Chunk]:
    """Keep chunks until the rough token budget is spent.

    A token is approximated as four characters — good enough for a ceiling and
    cheap enough to run on every injection.
    """
    if budget_tokens <= 0:
        return list(chunks)
    kept: List[Chunk] = []
    used = 0
    for chunk in chunks:
        cost = (len(chunk.text) + 300) // 4  # text + header/citation/invitation
        if kept and used + cost > budget_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept


def _seen_refs(state: Dict[str, Any] | None) -> set:
    """kb_refs already injected in this run (so a chunk is never repeated)."""
    if not isinstance(state, dict):
        return set()
    return set(state.get("knowledge_injected_refs") or [])


def render_block(
    chunks: Iterable[Chunk],
    header: str,
    intro: str,
) -> str:
    """Render chunks under a header, one invitation line per chunk."""
    chunks = list(chunks)
    if not chunks:
        return ""
    parts = [header, intro, ""]
    for chunk in chunks:
        parts.append(render_chunk(chunk))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def build_task_block(
    task: Dict[str, Any] | None,
    state: Dict[str, Any] | None = None,
    settings: KnowledgeSettings | None = None,
) -> str:
    """Cold injection: the chunks a task receives when it starts.

    The count comes from ``task["knowledge_chunks"]`` when the planner or the
    operator set it, otherwise from ``KNOWLEDGE_TASK_CHUNKS`` (default 3).
    """
    settings = settings or load_settings()
    if settings.disabled or not isinstance(task, dict):
        return ""
    count = chunks_for_task(task, settings)
    if count <= 0:
        return ""

    query = " ".join(
        str(task.get(key) or "")
        for key in ("description", "title", "goal")
    ).strip()
    if not query:
        return ""

    chunk_limit = count
    chunks = _search(query, chunk_limit, settings)
    chunks = _trim_to_budget(chunks, settings.budget_tokens)
    return render_block(
        chunks,
        header=TASK_BLOCK_HEADER,
        intro=(
            "The knowledge base found these passages for this task. Use them as "
            "the factual basis and cite them; read deeper with "
            'knowledge_base___read(kb_ref="…") when a passage is not enough.'
        ),
    )


def build_mid_run_block(
    query: str,
    state: Dict[str, Any] | None = None,
    settings: KnowledgeSettings | None = None,
    seen: Iterable[str] | None = None,
) -> str:
    """Mid-run injection after N tool calls.

    ``query`` should describe what the agent is currently working on (its task
    plus the latest tool activity). Chunks already injected in this run are
    skipped, so repeated injections add new material instead of noise.
    """
    settings = settings or load_settings()
    if not settings.mid_run_enabled:
        return ""
    count = settings.inject_chunks
    if count <= 0:
        return ""

    already = set(seen or ())
    already |= _seen_refs(state)

    # Over-fetch so filtering out what the agent has seen still leaves `count`.
    chunks = _search(query, count + len(already), settings)
    fresh = [c for c in chunks if c.reference.kb_ref not in already][:count]
    fresh = _trim_to_budget(fresh, settings.budget_tokens)
    if not fresh:
        return ""

    return render_block(
        fresh,
        header=MID_RUN_BLOCK_HEADER,
        intro=(
            "Background material from the knowledge base for what you are "
            "currently doing. Consider it, then continue your task."
        ),
    )


__all__ = [
    "MID_RUN_BLOCK_HEADER",
    "TASK_BLOCK_HEADER",
    "build_mid_run_block",
    "build_task_block",
    "render_block",
]
