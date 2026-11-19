#!/usr/bin/env python3
"""PDF/EPUB → markdown, so the existing chunker can read a real book.

Why this file exists
--------------------
The ingest path could chunk markdown and only *queued* PDFs: the PDF branch said
"Docling extraction pending" and never wrote a chunk. A queue that is never
drained looks exactly like a slow ingest (null job ids, no corpus), which is how
it reached the operator.

The output format is the one :func:`indexer.split_markdown` already understands:
a ``## Page N`` marker before the text of each page. The chunker treats that
marker as a page number (not a section), so a PDF becomes ordinary paragraph
chunks with a real ``page_from``.

Docling is preferred when installed (it preserves reading order and tables
better), PyMuPDF is the always-present fallback. Both produce the same shape;
nothing downstream knows which one ran.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: A page shorter than this is a scanned/image page with no text layer.
MIN_PAGE_CHARS = 20


class PdfExtractionError(RuntimeError):
    """The document could not be turned into text (no text layer, broken file)."""


def _docling_markdown(path: Path) -> Optional[str]:
    """Docling → markdown (``None`` when Docling is not installed)."""
    try:
        from docling.document_converter import DocumentConverter
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:  # noqa: BLE001 - fall through to PyMuPDF
        logger.info("[INGEST] docling failed for %s (%s); using PyMuPDF", path.name, exc)
        return None


def _pymupdf_markdown(path: Path) -> str:
    """PyMuPDF → markdown with a ``## Page N`` marker per page."""
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - ships with the devcontainer
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise PdfExtractionError(
                "no PDF reader available (install pymupdf)"
            ) from exc

    document = pymupdf.open(str(path))
    parts: List[str] = []
    text_pages = 0
    try:
        for number, page in enumerate(document, start=1):
            text = (page.get_text("text") or "").strip()
            if len(text) < MIN_PAGE_CHARS:
                continue
            text_pages += 1
            parts.append(f"## Page {number}\n\n{text}")
    finally:
        document.close()

    if text_pages == 0:
        raise PdfExtractionError(
            f"{path.name} has no text layer on any page — it is a scan. "
            "Run OCR first (the text would otherwise be an empty knowledge base)."
        )
    logger.info(
        "[INGEST] %s: %d page(s) with text via PyMuPDF", path.name, text_pages
    )
    return "\n\n".join(parts)


def pdf_to_markdown(path: str | Path) -> str:
    """Text of a PDF as markdown with page markers.

    Raises :class:`PdfExtractionError` when the document has no usable text, so
    the ingest reports "scanned, needs OCR" instead of writing an empty book.
    """
    path = Path(path)
    if not path.is_file():
        raise PdfExtractionError(f"{path} does not exist")

    # Docling's markdown carries its own structure (no page numbers).
    # PyMuPDF is the fallback and emits the ``## Page N`` markers the chunker
    # uses for ``page_from``.
    markdown = _docling_markdown(path)
    if markdown and markdown.strip():
        return markdown.strip()
    return _pymupdf_markdown(path)


def epub_to_markdown(path: str | Path) -> str:
    """Text of an EPUB as markdown (Docling when available, else EPUB reader)."""
    path = Path(path)
    markdown = _docling_markdown(path)
    if markdown and markdown.strip():
        return markdown.strip()

    try:
        import zipfile

        from xml.etree import ElementTree
    except ImportError as exc:  # pragma: no cover - stdlib
        raise PdfExtractionError(f"cannot read {path.name}: {exc}") from exc

    # Minimal EPUB: every XHTML document is one section, headings keep their
    # level so the chunker can build a section path.
    parts: List[str] = []
    names = ("html", "xhtml", "htm")
    with zipfile.ZipFile(str(path)) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(names):
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError:
                continue
            for element in root.iter():
                tag = str(element.tag).rsplit("}", 1)[-1].lower()
                text = "".join(element.itertext()).strip()
                if not text:
                    continue
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    level = int(tag[1])
                    parts.append(f"{'#' * min(level, 6)} {text}")
                elif tag in ("p", "li", "blockquote"):
                    parts.append(text)

    if not parts:
        raise PdfExtractionError(f"{path.name}: no readable text in the EPUB")
    return "\n\n".join(parts)


def to_markdown(path: str | Path) -> Dict[str, Any]:
    """Convert by extension; returns ``{"markdown": str, "kind": str}``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return {"markdown": pdf_to_markdown(path), "kind": "pdf"}
    if suffix == ".epub":
        return {"markdown": epub_to_markdown(path), "kind": "epub"}
    raise PdfExtractionError(f"unsupported document type {suffix!r} ({path.name})")


__all__ = [
    "PdfExtractionError",
    "epub_to_markdown",
    "pdf_to_markdown",
    "to_markdown",
]
