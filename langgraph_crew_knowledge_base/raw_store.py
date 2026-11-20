#!/usr/bin/env python3
"""The raw documents behind the corpus — kept, not discarded.

Chunking used to be one-way: a PDF was extracted to text, embedded, and the
file itself stayed wherever it happened to lie (a project's sandbox workspace, a
client vault, somebody's Downloads). Every chunk therefore pointed at a path that
could disappear, and an agent that wanted to show the operator "page 39 of the
actual book" had nothing to point at.

This module keeps the untouched original of every ingested document:

* **Content-addressed.** The file is stored under the sha256 of its bytes, so
  ingesting the same book twice stores one file, and two books can never
  collide the way `<title>.pdf` would.
* **One store, every namespace.** It lives at ``knowledge-raw:`` — a root that
  resolves for every customer and project, unlike the project vault the
  document usually came from.
* **Read-only for agents.** The root is ``ro``: agents read and cite it.
  Classification (which book, which corpus, which chunks) is the knowledge
  layer's job — an agent editing the source PDF would be editing evidence.

Layout::

    knowledge-raw:<sha256>.pdf            the file
    knowledge-raw:index/<sha256>.json     what it is and where it came from
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Reference prefix a chunk/source uses to point into the store.
RAW_REF_PREFIX = "knowledge-raw:"

#: Extensions worth keeping. Anything else is not a knowledge document.
SUPPORTED_SUFFIXES = {".pdf", ".epub", ".md", ".txt", ".html", ".htm"}


def raw_root() -> Optional[Path]:
    """The first writable directory behind ``knowledge-raw:``.

    Resolution order mirrors :mod:`state_paths`: an explicit
    ``KNOWLEDGE_RAW_ROOT``, then the configured root path, then the legacy
    per-user location. Returns ``None`` when nothing is usable — the caller
    then reports that the raw copy could not be stored instead of silently
    dropping it.
    """
    candidates = []
    explicit = os.getenv("KNOWLEDGE_RAW_ROOT", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    try:
        from app.core.roots import spec_for

        spec = spec_for("knowledge-raw")
        if spec.path:
            candidates.append(Path(spec.path).expanduser())
        for alt in spec.alt_paths or ():
            candidates.append(Path(alt).expanduser())
    except Exception as exc:  # noqa: BLE001 - fall through to the user dir
        logger.debug("[KNOWLEDGE_RAW] root spec unavailable: %s", exc)
    candidates.append(Path.home() / ".penta" / "knowledge-raw")

    for directory in candidates:
        if not directory:
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".kb-probe"
            probe.write_bytes(b"")
            probe.unlink()
            return directory
        except OSError as exc:
            logger.warning("[KNOWLEDGE_RAW] %s is not writable: %s", directory, exc)
            continue
    logger.warning(
        "[KNOWLEDGE_RAW] no writable raw store (tried %s) — raw copies are "
        "unavailable and chunks will carry no document link",
        ", ".join(str(c) for c in candidates if c),
    )
    return None


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def store_document(
    source: str | Path,
    *,
    book_id: str = "",
    corpus: str = "",
    original_ref: str = "",
) -> Dict[str, Any]:
    """Keep a copy of ``source`` in the raw store; return its reference.

    Returns ``{"ok": True, "raw_ref", "sha256", "bytes", "path"}`` or
    ``{"ok": False, "error": …}``. Never raises: an ingest that cannot archive
    its source still indexes the text, and the caller reports the missing link.
    """
    try:
        path = Path(source)
        if not path.is_file():
            return {"ok": False, "error": f"source not found: {source}"}
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return {"ok": False, "error": f"unsupported document type {suffix!r}"}

        root = raw_root()
        if root is None:
            return {"ok": False, "error": "no writable raw store (see KNOWLEDGE_RAW_ROOT)"}

        sha = _sha256_of(path)
        target = root / f"{sha}{suffix}"
        # Content-addressed: the same bytes are stored once, and a re-ingest
        # of an unchanged file costs one hash instead of a copy.
        if not target.exists():
            tmp = root / f".{sha}{suffix}.part"
            shutil.copyfile(path, tmp)
            os.replace(tmp, target)

        index_dir = root / "index"
        index_dir.mkdir(exist_ok=True)
        index_file = index_dir / f"{sha}.json"
        record = {
            "sha256": sha,
            "bytes": target.stat().st_size,
            "suffix": suffix,
            "original_name": path.name,
            "original_ref": original_ref,
            "book_id": book_id,
            "corpus": corpus,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            index_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("[KNOWLEDGE_RAW] index not written for %s: %s", sha, exc)

        logger.info(
            "[KNOWLEDGE_RAW] stored %s as %s%s (%s bytes)",
            path.name, sha[:12], suffix, record["bytes"],
        )
        return {
            "ok": True,
            "raw_ref": f"{RAW_REF_PREFIX}{sha}{suffix}",
            "sha256": sha,
            "bytes": record["bytes"],
            "path": str(target),
        }
    except Exception as exc:  # noqa: BLE001 - archiving must not break ingest
        logger.warning("[KNOWLEDGE_RAW] could not store %s: %s", source, exc)
        return {"ok": False, "error": str(exc)}


def resolve(raw_ref: str) -> Optional[Path]:
    """The file behind ``knowledge-raw:<name>``, or ``None`` when missing."""
    text = str(raw_ref or "").strip()
    if not text:
        return None
    name = text[len(RAW_REF_PREFIX):] if text.startswith(RAW_REF_PREFIX) else text
    # A ref must stay inside the store: no traversal out of it.
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    root = raw_root()
    if root is None:
        return None
    candidate = root / name
    return candidate if candidate.is_file() else None


def link_for(raw_ref: str, *, page: Optional[int] = None) -> Dict[str, Any]:
    """A citation-ready link: the reference plus whether it still resolves.

    ``available: False`` is an honest answer — a chunk whose source file is gone
    must not look like a working link.
    """
    text = str(raw_ref or "").strip()
    if not text:
        return {"raw_ref": "", "available": False, "reason": "no raw copy stored"}
    path = resolve(text)
    if path is None:
        return {
            "raw_ref": text,
            "available": False,
            "reason": "the stored original is missing (store moved or cleaned)",
        }
    out = {
        "raw_ref": text,
        "available": True,
        "bytes": path.stat().st_size,
        "name": path.name,
    }
    if page:
        out["page"] = int(page)
        # A page anchor is a fragment on the same reference, so a reader can
        # jump straight to the passage the chunk came from.
        out["page_ref"] = f"{text}#page={int(page)}"
    return out


__all__ = [
    "RAW_REF_PREFIX",
    "backfill",
    "SUPPORTED_SUFFIXES",
    "link_for",
    "raw_root",
    "resolve",
    "store_document",
]


def backfill(book_id: str = "", *, dry_run: bool = False) -> Dict[str, Any]:
    """Archive the source of an already-indexed book and tag its chunks.

    Books indexed before the raw store existed carry no ``raw_ref``, so their
    citations have no document link even though the file may still be where it
    was ingested from. This finds the book's ``source_ref``, archives the file
    and writes ``raw_ref`` onto every chunk — an in-place repair, no re-embed.

    Returns a report; ``ok: False`` with ``reason`` when the source is gone,
    which is the honest answer for a book whose PDF no longer exists.
    """
    from qdrant_client import models as qm

    from langgraph_crew_knowledge_base.indexer import COLLECTION
    from langgraph_crew_knowledge_base.ingest import resolve_root_path
    from langgraph_crew_knowledge_base.tools import _qdrant

    client = _qdrant()
    if client is None:
        return {"ok": False, "reason": "Qdrant unreachable"}

    filt = (
        qm.Filter(must=[qm.FieldCondition(key="book_id", match=qm.MatchValue(value=book_id))])
        if book_id
        else None
    )
    points, offset = [], None
    try:
        while True:
            page, offset = client.scroll(
                COLLECTION,
                limit=500,
                offset=offset,
                with_payload=["book_id", "source_ref", "corpus", "raw_ref"],
                with_vectors=False,
                scroll_filter=filt,
            )
            points.extend(page or [])
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"scroll failed: {exc}"}

    if not points:
        return {"ok": False, "reason": f"no chunks for book {book_id!r}"}

    books: Dict[str, Dict[str, Any]] = {}
    for point in points:
        payload = point.payload or {}
        key = str(payload.get("book_id") or "")
        entry = books.setdefault(
            key,
            {"source_ref": payload.get("source_ref") or "", "corpus": payload.get("corpus") or "",
             "ids": [], "already": bool(payload.get("raw_ref"))},
        )
        entry["ids"].append(point.id)

    report: Dict[str, Any] = {"ok": True, "books": {}, "archived": 0, "missing": 0, "tagged": 0}
    for name, entry in books.items():
        if entry["already"]:
            report["books"][name] = {"ok": True, "skipped": "already has a raw_ref"}
            continue
        source_ref = entry["source_ref"]
        try:
            path = resolve_root_path(source_ref) if source_ref else None
        except Exception as exc:  # noqa: BLE001
            path = None
            logger.warning("[KNOWLEDGE_RAW] cannot resolve %s: %s", source_ref, exc)
        if not path or not Path(path).is_file():
            # The exact case this store exists to prevent: chunks without the
            # document they were cut from. It cannot be repaired from here.
            report["missing"] += 1
            report["books"][name] = {
                "ok": False,
                "reason": (
                    f"source no longer exists: {source_ref or '(no source_ref)'} — "
                    "the chunks remain, the original must be re-ingested"
                ),
            }
            continue
        stored = store_document(path, book_id=name, corpus=entry["corpus"], original_ref=source_ref)
        if not stored.get("ok"):
            report["missing"] += 1
            report["books"][name] = {"ok": False, "reason": stored.get("error")}
            continue
        report["archived"] += 1
        report["books"][name] = {"ok": True, "raw_ref": stored["raw_ref"], "chunks": len(entry["ids"])}
        if not dry_run:
            try:
                client.set_payload(
                    collection_name=COLLECTION,
                    payload={"raw_ref": stored["raw_ref"]},
                    points=entry["ids"],
                )
                report["tagged"] += len(entry["ids"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("[KNOWLEDGE_RAW] tagging %s failed: %s", name, exc)
                report["books"][name]["tag_error"] = str(exc)
    return report
