#!/usr/bin/env python3
"""Chunking and indexing of knowledge documents into Qdrant.

The knowledge corpus is a real vector collection, not a mock catalogue. This
module is the write side:

    index_markdown(path, book, corpus, ...) -> (chunks, collection)
    ensure_collection()

Chunking follows the document's heading hierarchy. One chunk is the text below
one heading, so a chunk always has a `section_path` (``3.1.2``) and can be read
back precisely with ``knowledge_base___read(kb_ref="…#s:3.1.2")``. A section
that is too long is split at paragraph boundaries, and every part keeps the
same section path with a suffix.

Embeddings are produced locally with BGE-M3 (1024 dims, normalised for cosine).
If the model cannot be loaded, indexing fails loudly rather than writing
vectors of the wrong dimension into the collection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Collection that holds every knowledge chunk. One collection; `corpus`,
#: `tier` and `book_id` live in the payload so a search can be scoped.
COLLECTION = "knowledge_base_chunks"

#: Embedding model + vector size. The size is resolved from the configured
#: embedding server at runtime (``EMBED_DIM`` is only the fallback), because a
#: collection created with the wrong dimension rejects every write.
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

#: Target chunk size in characters. Larger sections are split at paragraphs.
MAX_CHUNK_CHARS = 1800

#: Below this a chunk is merged into the next one (avoids one-line chunks).
MIN_CHUNK_CHARS = 120

#: Chunker version — stored with every point so a re-index is traceable.
CHUNKER_VERSION = "kb-1.0"


# ── embedding ────────────────────────────────────────────────────────────────

def _embeddings():
    """Embedding backend: the project's BGE-M3 server (OpenAI-compatible).

    Reuses ``tools.fragment_vector_store.BGE_M3_Embeddings`` so the knowledge
    corpus and deep-reader share one embedding space and one server; that also
    handles the 512-token limit of the server by chunking long texts.
    """
    from tools.fragment_vector_store import BGE_M3_Embeddings

    return BGE_M3_Embeddings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts (normalised upstream by the server, cosine-ready)."""
    backend = _embeddings()
    return [list(map(float, v)) for v in backend.embed_documents(list(texts))]


def embed_dim() -> int:
    """Vector size of the configured embedding server (cached after 1st probe)."""
    global _dim
    if _dim:
        return _dim
    try:
        probe = embed_texts(["dimension probe"])
        _dim = len(probe[0]) if probe else EMBED_DIM
    except Exception as exc:  # noqa: BLE001 - fall back to the documented size
        logger.warning("[KNOWLEDGE_BASE] embedding probe failed: %s", exc)
        _dim = EMBED_DIM
    return _dim


_dim = 0


# ── collection ───────────────────────────────────────────────────────────────

def ensure_collection(client) -> None:
    """Create the knowledge collection with the embedding server's dimension."""
    from qdrant_client import models as qm

    size = embed_dim()
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        existing = getattr(info.config.params.vectors, "size", None)
        if existing and existing != size:
            raise RuntimeError(
                f"collection '{COLLECTION}' was created with {existing} dimensions "
                f"but the embedding server returns {size}. Re-create the "
                "collection (it holds no valid vectors then)."
            )
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=size, distance=qm.Distance.COSINE),
    )
    # Filtering happens on these, so index them up front.
    for field in ("corpus", "tier", "book_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001 - index creation is best effort
            logger.debug("[KNOWLEDGE_BASE] payload index %s: %s", field, exc)


# ── chunking ─────────────────────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _section_path(stack: List[Tuple[int, str, str]]) -> str:
    """``[(level, number, title), …]`` → ``3.1.2`` (numbers when present)."""
    parts = [num or _slug(title) for _lvl, num, title in stack if (num or title)]
    return ".".join(parts)


def split_markdown(content: str) -> List[Dict[str, Any]]:
    """Split markdown into heading-scoped chunks.

    Returns entries with ``text``, ``heading``, ``section_path`` and
    ``page_from`` (the synthetic ``## Page N`` headings the PDF-to-markdown
    conversion writes are used as page markers).
    """
    lines = content.splitlines()
    chunks: List[Dict[str, Any]] = []

    stack: List[Tuple[int, str, str]] = []
    current: Optional[Dict[str, Any]] = None
    buffer: List[str] = []
    counters: Dict[int, int] = {}
    page = None

    def flush() -> None:
        nonlocal buffer, current
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return
        if current is None:
            current = {
                "heading": "",
                "section_path": "",
                "page_from": page,
            }
        for part in _split_long(text):
            chunks.append(
                {
                    **current,
                    "text": part,
                    "page_from": current.get("page_from") or page,
                }
            )

    for raw in lines:
        line = raw.rstrip()
        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # The PDF converter emits "## Page 42" — a page marker, not a section.
            page_match = re.fullmatch(r"Page\s+(\d+)", title, re.IGNORECASE)
            if page_match:
                # A page boundary ends a chunk. Without this flush all pages of a
                # PDF collapse into the section above them (one huge chunk whose
                # page_from is the last page), which is the opposite of what a
                # reader opening a chunk needs.
                flush()
                page = int(page_match.group(1))
                current = None
                continue

            flush()
            counters[level] = counters.get(level, 0) + 1
            for deeper in [k for k in counters if k > level]:
                del counters[deeper]
            stack = [s for s in stack if s[0] < level]
            stack.append((level, None, title))
            current = {
                "heading": title,
                "section_path": _section_path(stack),
                "page_from": page,
            }
            continue

        buffer.append(line)

    flush()
    return [c for c in chunks if len(c["text"]) >= 1]


def _split_long(text: str) -> List[str]:
    """Split a long section at paragraph boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts: List[str] = []
    current: List[str] = []
    size = 0
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if size and size + len(block) > MAX_CHUNK_CHARS:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block) + 2
    if current:
        parts.append("\n\n".join(current))

    # Merge away slivers so the corpus does not fill with one-liners.
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + part
        else:
            merged.append(part)
    return merged or [text]


# ── indexing ─────────────────────────────────────────────────────────────────

def book_id_for(title: str, fallback: str = "") -> str:
    """Stable, readable id for a book (``bollinger-on-bollinger-bands``)."""
    bid = _slug(title)
    return bid or fallback or str(uuid.uuid4())[:8]


def index_markdown(
    client,
    path: str | Path,
    *,
    book_id: str,
    title: str,
    corpus: str,
    authors: Optional[Iterable[str]] = None,
    year: Optional[int] = None,
    tier: str = "standard",
    categories: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    source_ref: str = "",
    raw_ref: str = "",
) -> Dict[str, Any]:
    """Chunk a markdown file, embed it and upsert it into Qdrant.

    Thin wrapper over :func:`index_text` (which the PDF/EPUB path uses too).
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return index_text(
        client,
        content,
        book_id=book_id,
        title=title,
        corpus=corpus,
        authors=authors,
        year=year,
        tier=tier,
        categories=categories,
        tags=tags,
        source_ref=source_ref or str(path),
    )


def index_text(
    client,
    content: str,
    *,
    book_id: str,
    title: str,
    corpus: str,
    authors: Optional[Iterable[str]] = None,
    year: Optional[int] = None,
    tier: str = "standard",
    categories: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    source_ref: str = "",
    raw_ref: str = "",
) -> Dict[str, Any]:
    """Chunk already-extracted text, embed it and upsert it into Qdrant.

    The one indexing path: markdown files and PDFs/EPUBs (extracted to markdown
    first, see :mod:`extract`) end up in exactly the same shape here.

    Existing points of the same ``book_id`` are removed first, so re-ingesting
    replaces the book instead of duplicating it.
    """
    from qdrant_client import models as qm

    chunks = split_markdown(content)
    if not chunks:
        return {"status": "empty", "book_id": book_id, "chunks": 0}

    ensure_collection(client)

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    created_at = datetime.now(timezone.utc).isoformat()
    authors_list = list(authors or [])
    categories_list = list(categories or [])
    tags_list = list(tags or [])

    points = []
    for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
        section = chunk.get("section_path") or ""
        chunk_index = chunk.get("chunk_index", position)
        payload = {
            "book_id": book_id,
            "title": title,
            "corpus": corpus,
            "tier": tier,
            "authors": authors_list,
            "year": year,
            "categories": categories_list,
            "tags": tags_list,
            "section_path": section,
            "heading": chunk.get("heading") or "",
            "page_from": chunk.get("page_from"),
            "page_to": chunk.get("page_to"),
            "text": chunk["text"],
            "char_count": len(chunk["text"]),
            "chunk_index": chunk_index,
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": EMBED_MODEL,
            "source_ref": source_ref or "",
            # The untouched original this chunk came from, in the shared raw
            # store (knowledge-raw:<sha256>.<ext>). Without it a citation can
            # only name a file path that may no longer exist.
            "raw_ref": raw_ref or "",
            "content_sha256": hashlib.sha256(
                chunk["text"].encode("utf-8")
            ).hexdigest(),
            "created_at": created_at,
            "kb_ref": f"knowledge_base:{corpus}/{book_id}"
            + (f"#s:{section}" if section else ""),
        }
        # The id must distinguish the parts of one section: a section split at
        # paragraph boundaries keeps the same section_path, so section +
        # chunk_index (which is the position) is the unique key.
        points.append(
            qm.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{corpus}/{book_id}/{section}#{position}",
                    )
                ),
                vector=vector,
                payload=payload,
            )
        )

    # Re-ingest replaces the book: drop its old points first.
    try:
        client.delete(
            collection_name=COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="book_id", match=qm.MatchValue(value=book_id)
                        )
                    ]
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a failed delete must not abort
        logger.debug("[KNOWLEDGE_BASE] delete before reindex failed: %s", exc)

    client.upsert(collection_name=COLLECTION, points=points)

    return {
        "status": "indexed",
        "book_id": book_id,
        "title": title,
        "corpus": corpus,
        "tier": tier,
        "chunks": len(points),
        "collection": COLLECTION,
        "kb_ref": f"knowledge_base:{corpus}/{book_id}",
        "sections": len({c.get("section_path") for c in chunks}),
    }


__all__ = [
    "CHUNKER_VERSION",
    "COLLECTION",
    "EMBED_DIM",
    "EMBED_MODEL",
    "book_id_for",
    "embed_dim",
    "embed_texts",
    "ensure_collection",
    "index_markdown",
    "index_text",
    "split_markdown",
]


# ── Per-book score profile (ingest diagnostic) ──────────────────────────────

#: Reference queries used to profile a book. Deliberately broad: the point is
#: to measure how this book answers a *typical* question, not a curated one.
PROFILE_QUERIES = (
    "color theory contrast",
    "grid system layout",
    "brand identity logo",
    "typography legibility",
    "visual hierarchy",
    "page composition",
    "editorial design",
    "user interface navigation",
    "print production",
    "design process",
)


def profile_book(client, book_id: str, corpus: str) -> Dict[str, Any] | None:
    """Measure a book's score profile against :data:`PROFILE_QUERIES`.

    The result is an **outlier diagnostic**, not a runtime threshold: on this
    deployment the five books differ by 0.036 in median while the spread inside
    one book is 0.127-0.189, so a per-book threshold would move by ±0.02 and
    change nothing. What a profile *does* catch is a book that scores
    systematically low — different language, worse text, odd chunking — which
    would otherwise produce weak hits forever with no explanation.

    Returns the profile dict (also written next to the chunks as a payload-less
    sidecar file), or ``None`` when Qdrant is unreachable.
    """
    try:
        from app.core import memory_gate
        from langgraph_crew_knowledge_base.tools import knowledge_base___search
    except Exception as exc:  # noqa: BLE001 - profiling is optional
        logger.warning("[KNOWLEDGE_BASE] book profiling unavailable: %s", exc)
        return None

    scores: List[float] = []
    for query in PROFILE_QUERIES:
        try:
            # Query THIS book's vectors directly, not a shared corpus list: in a
            # shared top-k the biggest book crowds a small one out entirely
            # (measured: 40 hits covered only 3 of 5 books), and a profile built
            # from the leftovers would describe rank share, not the book.
            raw = knowledge_base___search(
                query=query, corpora=[corpus], book_id=book_id, top_k=5
            )
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("[KNOWLEDGE_BASE] profile query %r failed: %s", query, exc)
            continue
        for hit in payload.get("results") or []:
            if hit.get("book_id") == book_id and hit.get("score") is not None:
                scores.append(float(hit["score"]))

    if not scores:
        return None
    profile = memory_gate.build_profile(book_id, scores)
    logger.info(
        "[KNOWLEDGE_BASE] book profile %s: median %.3f (p10 %.3f, p90 %.3f, n=%d)",
        book_id, profile.median, profile.p10, profile.p90, profile.samples,
    )
    return profile.as_dict()


def write_profile(book_id: str, profile: Dict[str, Any]) -> str:
    """Store a book profile beside the plugin state (readable, not indexed)."""
    from langgraph_crew_knowledge_base.state_paths import resolve_db_path

    base = resolve_db_path(None).parent / "book_profiles"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{book_id}.json"
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return str(path)


def report_outliers() -> List[Dict[str, Any]]:
    """Compare all stored book profiles and return the statistical outliers."""
    from app.core import memory_gate
    from langgraph_crew_knowledge_base.state_paths import resolve_db_path

    base = resolve_db_path(None).parent / "book_profiles"
    if not base.is_dir():
        return []
    profiles = []
    for path in sorted(base.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(
                memory_gate.BookProfile(
                    book_id=str(data.get("book_id") or path.stem),
                    median=float(data.get("median") or 0.0),
                    p10=float(data.get("p10") or 0.0),
                    p90=float(data.get("p90") or 0.0),
                    samples=int(data.get("samples") or 0),
                )
            )
        except Exception:  # noqa: BLE001 - a broken file must not break the report
            continue
    return memory_gate.outliers(profiles)


__all__ = sorted(set(__all__) | {"PROFILE_QUERIES", "profile_book", "report_outliers", "write_profile"})
