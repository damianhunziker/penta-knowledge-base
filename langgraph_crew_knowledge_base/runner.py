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
import re
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

    No score filtering happens here: the relevance decision belongs to
    :func:`apply_relevance`, which derives its floor from this query's own
    score scale. The old ``score < settings.min_score`` cut used one global
    number (0.35) for a corpus whose matching pages score 0.75-0.81 and whose
    unrelated ones score 0.66 — it filtered nothing, and an index page at 0.77
    displaced a real chapter.
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
        chunks.append(chunk)
    return chunks


def apply_relevance(
    chunks: Sequence[Chunk],
    query: str,
    settings: KnowledgeSettings,
    fill_target: int = 0,
) -> List[Chunk]:
    """Gate + rerank a candidate list, using the shared memory gate.

    The knowledge corpus is one memory type, so it uses the same decision code
    as episodes, rules and the agent's own history
    (:mod:`app.core.memory_gate`): a per-query floor, an optional adaptive fill
    that never crosses the measured noise ceiling, and a lexical rerank that
    separates content from index/bibliography/credit pages.

    Returns the surviving chunks in final order (possibly fewer than input).
    """
    if not chunks:
        return []
    from app.core import memory_gate

    by_ref = {c.reference.kb_ref: c for c in chunks}
    pairs = [(c.reference.kb_ref, float(c.score or 0.0)) for c in chunks]

    result = memory_gate.gate(
        pairs,
        memory_type="knowledge",
        query=query,
        text_of=lambda ref: by_ref[ref].text if ref in by_ref else "",
        fill_target=fill_target,
        rerank=True,
    )

    ordered: List[Chunk] = []
    for ref, decision in result.kept:
        chunk = by_ref.get(ref)
        if chunk is None:
            continue
        # Keep the gate's verdict with the chunk so a debug view can show WHY
        # it was injected (floor vs adaptive_fill, and its lexical coverage).
        # ``Chunk`` is frozen, so the field is replaced rather than assigned.
        from dataclasses import replace as _replace

        ordered.append(_replace(chunk, relevance=decision.as_dict()))

    logger.debug(
        "[KNOWLEDGE_BASE] gate: %d in, %d out (floor %.3f, %d adaptive%s)",
        len(chunks),
        len(ordered),
        result.floor,
        result.adaptive,
        ", ambiguous" if result.ambiguous else "",
    )
    return ordered


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
    """Fit ``chunks`` into ``budget_tokens`` while keeping all of them.

    A token is approximated as four characters — good enough for a ceiling and
    cheap enough to run on every injection.

    The budget is split **per chunk**, not handed out first-come-first-served.
    The old greedy version gave the whole budget to the first hit (a 3'300-char
    Meggs page = 898 of 900 tokens) and dropped the rest, so a task configured
    for ``KNOWLEDGE_TASK_CHUNKS=3`` received one passage and nothing said so.
    Every requested chunk now appears, shortened at a paragraph boundary when
    it has to be.
    """
    chunks = list(chunks)
    if budget_tokens <= 0 or not chunks:
        return chunks

    share = max(200, budget_tokens // len(chunks))
    kept: List[Chunk] = []
    for chunk in chunks:
        cut = _cut_to_tokens(chunk, share)
        if cut is not None:
            kept.append(cut)
    return kept


def _cut_to_tokens(chunk: Chunk, tokens: int) -> Chunk | None:
    """Cut ``chunk.text`` at a paragraph boundary to fit ``tokens``.

    Returns a new ``Chunk`` (the reference stays intact, so the invitation line
    still opens the *whole* section), or ``None`` when even a short cut would
    be meaningless.
    """
    limit = max(200, tokens * 4)
    text = chunk.text.strip()
    if len(text) <= limit:
        return chunk
    cut = text[:limit]
    # Prefer a paragraph, then a sentence, then a word boundary.
    for marker in ("\n\n", ". ", " "):
        boundary = cut.rfind(marker)
        if boundary > limit // 2:
            cut = cut[: boundary + len(marker)]
            break
    cut = cut.rstrip()
    if len(cut) < 200:
        return None
    return Chunk(
        text=cut + "\n… (passage shortened to fit the token budget; read the "
        'section with knowledge_base___read(kb_ref="…") for the full text)',
        reference=chunk.reference,
        score=chunk.score,
    )


def _seen_refs(state: Dict[str, Any] | None) -> set:
    """kb_refs already injected in this run (so a chunk is never repeated)."""
    if not isinstance(state, dict):
        return set()
    return set(state.get("knowledge_injected_refs") or [])


def _is_index_page(text: str) -> bool:
    """True for a book's index/table-of-contents/bibliography page.

    These pages match a topic search strongly (every term of the query appears
    in them) but tell a model nothing: they are lists of page numbers or book
    titles. Injected as "the passage for this task" they displace the real
    content — exactly what happened with ``page 216`` of Wu ("Dynamic Diagrams'
    interface design, 179 / interaction with, 172-174 / …") and with the Meggs
    and Pettersson bibliographies.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) < 5:
        return False

    # A. Index pages: a large share of lines end in a page reference
    #    ("Dynamic Diagrams' interface design, 179", "interaction with, 172-174").
    refs = sum(
        1
        for line in lines
        if re.search(r"\d{1,3}(\s*[-\u2013]\s*\d{1,3})?\s*$", line)
        and len(line) < 90
    )
    if refs / len(lines) >= 0.5:
        return True

    # A'. Two-column indexes put the page number mid-line
    #     ("Kieser, Günther | peace poster, 148 | King, Jessie M.").
    #     Count lines whose LAST number sits after the last comma with no
    #     sentence punctuation after it — a name list, not prose.
    paged = 0
    for line in lines:
        if len(line) > 160:
            continue
        tail = line.rsplit(",", 1)[-1].strip()
        if re.fullmatch(r"\d{1,3}(\s*[-\u2013]\s*\d{1,3})?", tail):
            paged += 1
    if paged / len(lines) >= 0.4:
        return True

    # B. Bibliographies / reference lists: many entries look like
    #    "Author, A. & Author, B. (2007). Title. Publisher."
    #    A bibliography is dense in years in parentheses and initials.
    citations = sum(
        1
        for line in lines
        if re.search(r"\(\d{4}[a-z]?\)", line) and re.search(r"\b[A-Z]\.\s", line)
    )
    if citations / len(lines) >= 0.35:
        return True

    # C. Table of contents: numbered headings without running prose.
    numbered = sum(1 for line in lines if re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+\S", line))
    words = sum(len(line.split()) for line in lines)
    if numbered / len(lines) >= 0.5 and words / len(lines) < 8:
        return True

    # D. Image credits / copyright page: "Photograph of X © 1996-2003 Y Ltd."
    #    Dense in © signs and permission wording; carries no design content.
    credits = sum(
        1
        for line in lines
        if "\u00a9" in line or re.search(r"\b(used with permission|reprinted|courtesy of)\b", line, re.I)
    )
    if credits / len(lines) >= 0.25:
        return True

    return False


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

    # Over-fetch so the gate can drop index pages and still leave `count`
    # real passages. The gate (not the old 0.35 cut) decides what survives:
    # per-query floor, adaptive fill to `count`, lexical rerank.
    chunks = _search(query, count * 3, settings)
    usable = [c for c in chunks if not _is_index_page(c.text)]
    if usable:
        chunks = usable
    else:
        logger.debug(
            "[KNOWLEDGE_BASE] every hit looked like an index page for %r", query
        )
    chunks = apply_relevance(chunks, query, settings, fill_target=count)[:count]
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
    chunks = _search(query, (count + len(already)) * 2, settings)
    fresh = [c for c in chunks if c.reference.kb_ref not in already]
    fresh = apply_relevance(fresh, query, settings, fill_target=count)[:count]
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
