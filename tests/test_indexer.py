#!/usr/bin/env python3
"""Chunking + indexing tests (the write side of the corpus)."""

import pytest

from langgraph_crew_knowledge_base import indexer as I


class TestSplitMarkdown:
    def test_headings_become_section_paths(self):
        md = "# Book\n\nintro\n\n## Chapter 1\n\ntext one\n\n## Chapter 2\n\ntext two\n"
        chunks = I.split_markdown(md)
        paths = [c["section_path"] for c in chunks]
        assert "book.chapter-1" in paths or any("chapter-1" in p for p in paths)
        assert any("chapter-2" in p for p in paths)

    def test_nested_headings_nest_the_path(self):
        md = "# A\n\n## B\n\n### C\n\ndeep text\n"
        chunks = I.split_markdown(md)
        assert any(c["section_path"].count(".") >= 2 for c in chunks)

    def test_page_markers_are_not_sections(self):
        """'## Page 42' from the PDF converter is a page marker."""
        md = "# Book\n\n## Page 42\n\nbody text\n\n## Page 43\n\nmore text\n"
        chunks = I.split_markdown(md)
        assert all("page" not in c["section_path"].lower() for c in chunks)
        # The trailing page marker sets the page of the buffered text.
        assert any(c["page_from"] in (42, 43) for c in chunks)

    def test_long_sections_are_split(self):
        long_body = "\n\n".join(["paragraph " * 40] * 8)
        md = f"# Book\n\n## Section\n\n{long_body}\n"
        chunks = I.split_markdown(md)
        assert len(chunks) > 1
        assert all(len(c["text"]) <= I.MAX_CHUNK_CHARS * 1.2 for c in chunks)

    def test_empty_document(self):
        assert I.split_markdown("") == []


class TestBookId:
    def test_slug_from_title(self):
        assert I.book_id_for("Bollinger On Bollinger Bands") == \
            "bollinger-on-bollinger-bands"

    def test_fallback(self):
        assert I.book_id_for("", "abc") == "abc"


class TestIndexMarkdown:
    """The indexer writes real points, one per chunk (no collisions)."""

    def test_same_section_parts_get_distinct_ids(self, monkeypatch, tmp_path):
        """Parts of one section must not overwrite each other."""
        captured = {}

        class Client:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                class I_:
                    class config:
                        class params:
                            class vectors:
                                size = 768
                return I_()

            def delete(self, **kw):
                pass

            def upsert(self, collection_name=None, points=None, **kw):
                captured["points"] = points

        monkeypatch.setattr(I, "embed_dim", lambda: 768)
        monkeypatch.setattr(I, "embed_texts", lambda texts: [[0.1] * 768 for _ in texts])

        body = "\n\n".join(["paragraph " * 40] * 8)
        md = f"# Book\n\n## Section\n\n{body}\n"
        path = tmp_path / "book.md"
        path.write_text(md, encoding="utf-8")

        out = I.index_markdown(
            Client(), path, book_id="book", title="Book", corpus="c"
        )
        ids = [str(p.id) for p in captured["points"]]
        assert out["chunks"] == len(ids)
        assert len(set(ids)) == len(ids), "duplicate point ids would drop chunks"
