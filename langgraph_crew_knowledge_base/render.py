#!/usr/bin/env python3
"""
Knowledge Base Renderer

The contract: ONE invitation line per chunk. The read tool offers broader resolutions.

This is the reference implementation for rendering chunks with invitations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """Complete reference attached to every chunk. Required, not Optional."""
    kb_ref: str              # knowledge_base:academic/meadows-2008/ch3#s:1.2
    title: str
    authors: tuple[str, ...]
    year: int | None
    page_from: int | None
    page_to: int | None
    section_path: str        # ch3.1.2 — the heading path
    tier: str                # foundational | standard | specialist | task
    book_id: str             # meadows-2008
    corpus: str              # academic
    isbn: str | None = None
    doi: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A retrieved chunk with its reference. The reference is required."""
    text: str
    reference: Reference
    score: float
    # The following are for budget management, not rendering:
    parent_ref: str | None = None   # if this is a child, what's the parent
    chunk_index: int = 0
    char_count: int = 0
    # The relevance gate's verdict (floor/adaptive_fill, similarity, lexical
    # coverage). Diagnostics only: it is never rendered into the prompt, but a
    # debug view can show WHY a passage was injected.
    relevance: dict | None = None


# ── RENDERING ────────────────────────────────────────────────────────────────

def render_chunk(chunk: Chunk, max_text_chars: int = 2000) -> str:
    """
    Render a chunk with ONE invitation line.
    
    This is the contract between retrieval and the agent.
    The agent never composes a kb_ref — it copies one that was shown.
    
    Output shape:
    
        [kb:academic/meadows-2008/ch3#s:1.2 · p.118–119 · tier=foundational · score 0.81]
        Meadows, D. H., *Thinking in Systems* (2008), "1.2 The Signal Chain"
        
        > …chunk text…
        
        ↳ read section   knowledge_base___read(kb_ref="knowledge_base:academic/meadows-2008/ch3#s:1.2")
    """
    ref = chunk.reference
    
    # Truncate text if needed (rare — budget is usually enforced before rendering)
    text = chunk.text
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + "…"
    
    # Header: [kb:corpus/book/section#heading · pages · tier · score]
    header = _render_header(ref, chunk.score)
    
    # Citation: Authors, *Title* (Year), "Section Path"
    citation = _render_citation(ref)
    
    # Text block with >
    text_block = "> " + text.replace("\n", "\n> ")
    
    # THE ONE INVITATION LINE
    invitation = f'↳ read section   knowledge_base___read(kb_ref="{ref.kb_ref}")'
    
    return f"{header}\n{citation}\n\n{text_block}\n\n{invitation}"


def _render_header(ref: Reference, score: float) -> str:
    """Render the [kb:ref · pages · tier · score] header line."""
    # Strip the "knowledge_base:" prefix for display (cleaner, still copyable)
    display_ref = ref.kb_ref.split(":", 1)[1] if ":" in ref.kb_ref else ref.kb_ref
    
    parts = [f"kb:{display_ref}"]
    
    # Pages
    if ref.page_from and ref.page_to:
        parts.append(f"p.{ref.page_from}–{ref.page_to}")
    elif ref.page_from:
        parts.append(f"p.{ref.page_from}")
    
    parts.append(f"tier={ref.tier}")
    parts.append(f"score {score:.2f}")
    
    return "[" + " · ".join(parts) + "]"


def _render_citation(ref: Reference) -> str:
    """Render the citation line: Authors, *Title* (Year), "Section"."""
    # Authors: first two, then "et al." if more
    if ref.authors:
        if len(ref.authors) <= 2:
            authors = ", ".join(ref.authors)
        else:
            authors = f"{ref.authors[0]}, {ref.authors[1]} et al."
    else:
        authors = "Unknown"
    
    # Title (italicized in markdown)
    title_part = f"*{ref.title}*"
    
    # Year
    year_part = f" ({ref.year})" if ref.year else ""
    
    # Section path (in quotes if present)
    section_part = f', "{ref.section_path}"' if ref.section_path else ""
    
    return f"{authors}, {title_part}{year_part}{section_part}"


# ── DERIVING BROADER REFS (used by the read tool, not the renderer) ──────────

def derive_broader_refs(kb_ref: str) -> dict[str, str | None]:
    """
    Derive chapter/book from a section kb_ref.
    
    The read tool uses this to offer broader invitations in its response.
    The agent never calls this — it just sees the invitations.
    
    Input:  knowledge_base:academic/meadows-2008/ch3#s:1.2
    Output: {
        "section":   "knowledge_base:academic/meadows-2008/ch3#s:1.2",
        "chapter":   "knowledge_base:academic/meadows-2008/ch3",
        "neighbours_call": 'kb_ref="...", before=1, after=1',
    }
    """
    result: dict[str, str | None] = {"section": kb_ref}
    
    # Strip the "knowledge_base:" prefix for path analysis
    path = kb_ref.split(":", 1)[1] if ":" in kb_ref else kb_ref
    
    if "#s:" in kb_ref:
        # Section-level: derive chapter by removing the #s:... part
        base, _ = kb_ref.rsplit("#s:", 1)
        result["chapter"] = base
        result["neighbours_call"] = f'kb_ref="{kb_ref}", before=1, after=1'
    elif path.count("/") == 2:
        # Chapter-level: corpus/book/chapter (2 slashes after prefix)
        result["chapter"] = kb_ref
        # Book is one level up: remove last path segment
        parts = kb_ref.rsplit("/", 1)
        result["book"] = parts[0] if len(parts) > 1 else None
    else:
        # Book-level or corpus-level (0-1 slashes)
        result["book"] = kb_ref
        result["chapter"] = None
    
    return result


def render_read_response(
    text: str,
    ref: Reference,
    budget_used: int,
    budget_max: int,
    truncated: bool = False,
) -> str:
    """
    Render the response from knowledge_base___read.
    
    This includes the content AND the invitations to go broader.
    The agent learns here that chapter/book are available.
    """
    broader = derive_broader_refs(ref.kb_ref)
    
    invitations = []
    
    # If we showed a section, offer the chapter
    if broader.get("chapter") and broader["chapter"] != ref.kb_ref:
        invitations.append(
            f'↳ read chapter   knowledge_base___read(kb_ref="{broader["chapter"]}")'
        )
    
    # If we showed a chapter, offer the book
    if broader.get("book") and broader["book"] != ref.kb_ref:
        invitations.append(
            f'↳ read book      knowledge_base___read(book_id="{ref.book_id}", '
            f'around="{ref.section_path}")'
        )
    
    # Neighbours (only for sections, not chapters)
    if broader.get("neighbours_call"):
        invitations.append(
            f'↳ neighbours     knowledge_base___read({broader["neighbours_call"]})'
        )
    
    # Budget notice if truncated
    budget_note = ""
    if truncated:
        remaining_inv = invitations[0] if invitations else "search for related content"
        budget_note = f"\n\n[truncated at {budget_used}/{budget_max} tok — remaining: {remaining_inv}]"
    
    inv_block = "\n".join(invitations) if invitations else ""
    if inv_block:
        inv_block = "\n\n" + inv_block
    
    return f"{text}{budget_note}{inv_block}"


# ── PUBLIC EXPORTS ────────────────────────────────────────────────────────────

__all__ = [
    "Reference",
    "Chunk",
    "render_chunk",
    "render_read_response",
    "derive_broader_refs",
]