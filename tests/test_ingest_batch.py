#!/usr/bin/env python3
"""Batch ingestion must find files, index them, and report what it wrote.

Three defects lived here, each of which made ``ingest_directory`` look like it
worked while it produced nothing:

1. ``Path.glob`` does not expand ``{md,pdf,epub}``. The default pattern matched
   zero files on a directory full of documents.
2. ``discover_files`` spliced ``f"{root}:{rel}"`` unconditionally, so an
   absolute-path root produced ``/tmp/x:book.md`` — an unresolvable reference.
   Every discovered file then failed ingestion as "File not found".
3. The payload reported ``files_found`` but dropped per-file ``chunks`` and
   ``book_id``, and hand ``processed``/``total_chunks`` that were always None.
   A caller could not tell an empty batch from a successful one.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def ingest():
    from langgraph_crew_knowledge_base import ingest as module

    return module


@pytest.fixture
def docs(tmp_path):
    """Two markdown files and one the pattern must ignore."""
    for i in (1, 2):
        (tmp_path / f"book{i}.md").write_text(
            f"---\ntitle: Probe Book {i}\n---\n\n# Probe\n\nBody text. "
        )
    (tmp_path / "notes.txt").write_text("not a document")
    return tmp_path


class TestGlob:
    def test_brace_pattern_matches_each_extension(self, ingest, docs):
        found = {p.name for p in ingest._iter_glob(docs, "**/*.{md,pdf,epub}")}
        assert found == {"book1.md", "book2.md"}

    def test_a_plain_pattern_still_works(self, ingest, docs):
        assert len(list(ingest._iter_glob(docs, "**/*.md"))) == 2


class TestDiscover:
    def test_an_absolute_root_yields_resolvable_refs(self, ingest, docs):
        result = json.loads(
            ingest.knowledge_base___discover_files(str(docs), "**/*.{md,pdf,epub}", 50)
        )
        assert result["files"]
        for entry in result["files"]:
            ref = entry["root_ref"]
            # A bare path (no root name) or a name-based ref — never "/tmp/x:b.md"
            assert ingest.resolve_root_path(ref) == str(docs / entry["path"])


class TestIngestDirectory:
    def test_the_payload_reports_what_was_written(self, ingest, docs, monkeypatch):
        # Indexing itself is covered elsewhere; this is about the reporting.
        monkeypatch.setattr(
            ingest,
            "knowledge_base___ingest_from_root",
            lambda ref, corpus, **kw: json.dumps(
                {
                    "status": "indexed",
                    "title": Path(ref).stem,
                    "book_id": Path(ref).stem,
                    "kb_ref": f"knowledge_base:{corpus}/{Path(ref).stem}",
                    "chunks": 3,
                }
            ),
        )
        result = json.loads(
            ingest.knowledge_base___ingest_directory(
                root=str(docs), corpus="probe", background=False
            )
        )
        assert result["files_found"] == 2
        assert result["processed"] == 2
        assert result["total_chunks"] == 6
        assert [b["book_id"] for b in result["books"]] == ["book1", "book2"]
        assert result["failed"] == 0
        assert result["errors"] == []

    def test_a_failing_file_is_counted_and_named(self, ingest, docs, monkeypatch):
        def fake(ref, corpus, **kw):
            if ref.endswith("book2.md"):
                return json.dumps({"status": "error", "error": "embedding service down"})
            return json.dumps({"status": "indexed", "book_id": "book1", "chunks": 2})

        monkeypatch.setattr(ingest, "knowledge_base___ingest_from_root", fake)
        result = json.loads(
            ingest.knowledge_base___ingest_directory(
                root=str(docs), corpus="probe", background=False
            )
        )
        assert result["processed"] == 1
        assert result["failed"] == 1
        assert result["total_chunks"] == 2
        assert result["errors"][0]["error"] == "embedding service down"
        assert result["errors"][0]["file"].endswith("book2.md")
        # A partial success must say so rather than read like a clean run.
        assert "failed" in result["message"]
