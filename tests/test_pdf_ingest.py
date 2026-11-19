#!/usr/bin/env python3
"""A PDF must become real chunks — not a queued job that never runs.

The bug this covers: ``chunk_document`` answered "queued" with a job id, stored
it in a dict and reported 42 chunks, while nothing was ever written; the PDF
branch of the ingest said "Docling extraction pending". A caller saw null job
status and an unchanged corpus and could not tell it apart from a slow ingest.

The tests below pin the three properties that make it honest:
  * a PDF is converted to text (page markers per page),
  * the chunker keeps one chunk per page (not one giant chunk),
  * a scan without a text layer fails loudly instead of writing an empty book.
"""

from pathlib import Path

import pytest

sys_path = Path(__file__).resolve().parent.parent
import sys

if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from langgraph_crew_knowledge_base.extract import (  # noqa: E402
    PdfExtractionError,
    pdf_to_markdown,
    to_markdown,
)
from langgraph_crew_knowledge_base.indexer import split_markdown  # noqa: E402

PAGES = (
    "Chapter 1: Eliminate Billable Hours. Pricing an agency by the hour "
    "punishes efficiency and rewards slowness. Move to value-based pricing.",
    "Chapter 2: Build a Hybrid Agency. A hybrid agency combines traditional "
    "creative services with digital execution in one integrated team.",
    "Chapter 3: Attract and Develop Talent. Hiring for cultural fit and growth "
    "potential beats hiring for a checklist of tools.",
)


@pytest.fixture
def pdf(tmp_path):
    """A real 3-page PDF, written with PyMuPDF."""
    pymupdf = pytest.importorskip("pymupdf")
    target = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 700), text, fontsize=11)
    doc.save(str(target))
    doc.close()
    return target


@pytest.fixture
def blank_pdf(tmp_path):
    """A PDF with pages but no text layer (what a scan looks like)."""
    pymupdf = pytest.importorskip("pymupdf")
    target = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(target))
    doc.close()
    return target


class TestExtraction:
    def test_text_comes_out_with_page_markers(self, pdf):
        markdown = pdf_to_markdown(pdf)
        assert "## Page 1" in markdown
        assert "## Page 2" in markdown
        assert "Billable Hours" in markdown

    def test_to_markdown_reports_the_kind(self, pdf):
        result = to_markdown(pdf)
        assert result["kind"] == "pdf"
        assert result["markdown"]

    def test_a_scan_fails_loudly(self, blank_pdf):
        """An empty book is worse than an error: it looks like knowledge."""
        with pytest.raises(PdfExtractionError) as exc:
            pdf_to_markdown(blank_pdf)
        assert "no text layer" in str(exc.value)

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(PdfExtractionError):
            pdf_to_markdown(tmp_path / "nope.pdf")

    def test_unsupported_type_is_an_error(self, tmp_path):
        other = tmp_path / "x.txt"
        other.write_text("hi")
        with pytest.raises(PdfExtractionError):
            to_markdown(other)


class TestChunking:
    def test_one_chunk_per_page(self, pdf):
        """Pages must not collapse into one section-sized chunk."""
        chunks = split_markdown(pdf_to_markdown(pdf))
        assert len(chunks) == 3
        assert [c["page_from"] for c in chunks] == [1, 2, 3]

    def test_chunks_carry_real_text_not_metadata(self, pdf):
        chunks = split_markdown(pdf_to_markdown(pdf))
        for chunk in chunks:
            assert not chunk["text"].lstrip().startswith("---")
            assert len(chunk["text"]) >= 50


class TestNoSilentQueue:
    def test_chunk_document_does_not_fake_a_job(self, monkeypatch):
        """The tool must report what it wrote, not invent a completed job."""
        from langgraph_crew_knowledge_base import tools

        seen = {}

        def fake_ingest(root_ref, **kwargs):
            seen["called"] = True
            seen["background"] = kwargs.get("background")
            return tools.json.dumps({
                "status": "indexed",
                "book_id": "b1",
                "chunks": 7,
            })

        import langgraph_crew_knowledge_base.ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "knowledge_base___ingest_from_root", fake_ingest)
        payload = tools.json.loads(
            tools.knowledge_base___chunk_document("/tmp/x.pdf", corpus="c", background=True)
        )
        assert seen["called"] is True
        assert seen["background"] is False, "indexing must run synchronously"
        assert payload["status"] == "indexed"
        assert payload["chunks"] == 7
        assert "no background job to poll" in payload["note"]

    def test_unknown_job_is_not_an_error(self):
        from langgraph_crew_knowledge_base import tools

        payload = tools.json.loads(tools.knowledge_base___chunk_status("nope"))
        assert payload["status"] == "unknown"
        assert "synchronous" in payload["message"]
