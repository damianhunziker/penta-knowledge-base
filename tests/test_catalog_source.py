#!/usr/bin/env python3
"""The catalog tools must read the CORPUS, not the demo fixture.

Four tools (``list``, ``describe``, ``recommend_titles``, ``read``) read
``_MOCK_CATALOG`` — a two-book demo fixture — so they answered "Book not found"
for every real book while the vector search next door returned hits from those
very books. An agent cannot trust a set of tools that disagree about what
exists.
"""

import json

import pytest


def _tool(name):
    from langgraph_crew_knowledge_base import tools

    return getattr(tools, name)


BOOK = {
    "title": "Universal Principles of Design",
    "authors": ["Lidwell, William"],
    "year": 2003,
    "tier": "standard",
    "categories": ["design"],
    "chunks": 61,
}


@pytest.fixture
def corpus(monkeypatch):
    """A one-book live catalog, so nothing touches Qdrant."""
    from langgraph_crew_knowledge_base import tools

    live = {
        "corpora": {"graphic-design": {"name": "Graphic Design", "books": {"real-book": dict(BOOK)}}},
        "total_books": 1,
        "total_chunks": 61,
    }
    monkeypatch.setattr(tools, "_build_catalog_from_qdrant", lambda: live)
    return tools


class TestDescribe:
    def test_a_real_book_is_found(self, corpus):
        data = json.loads(_tool("knowledge_base___describe")("real-book"))
        assert data["title"] == BOOK["title"]
        assert data["chunk_count"] == 61
        assert data["corpus"] == "graphic-design"

    def test_an_unknown_book_still_reports_an_error(self, corpus):
        data = json.loads(_tool("knowledge_base___describe")("nope"))
        assert "error" in data


class TestList:
    def test_lists_the_live_catalog(self, corpus):
        data = json.loads(_tool("knowledge_base___list")())
        books = data.get("books") or []
        assert [b.get("book_id") for b in books] == ["real-book"]


class TestRecommend:
    def test_recommends_from_passage_hits_not_the_title(self, corpus, monkeypatch):
        """"color theory" must be able to match a book whose title lacks it."""
        hits = [
            {"book_id": "real-book", "title": BOOK["title"], "corpus": "graphic-design",
             "score": 0.83, "authors": BOOK["authors"]},
            {"book_id": "real-book", "title": BOOK["title"], "corpus": "graphic-design",
             "score": 0.81, "authors": BOOK["authors"]},
            {"book_id": "real-book", "title": BOOK["title"], "corpus": "graphic-design",
             "score": 0.80, "authors": BOOK["authors"]},
        ]
        monkeypatch.setattr(corpus, "_vector_search", lambda q, k, *a, **kw: hits)
        data = json.loads(_tool("knowledge_base___recommend_titles")("color theory"))
        assert data["total"] == 1
        rec = data["recommendations"][0]
        assert rec["book_id"] == "real-book"
        assert rec["passages"] == 3
        assert data["confidence"] == "high"

    def test_a_weak_match_says_so_instead_of_hiding_it(self, corpus, monkeypatch):
        """Overlapping score distributions cannot be filtered — so report."""
        hits = [{"book_id": "real-book", "title": BOOK["title"], "corpus": "graphic-design",
                 "score": 0.77, "authors": []}]
        monkeypatch.setattr(corpus, "_vector_search", lambda q, k, *a, **kw: hits)
        data = json.loads(_tool("knowledge_base___recommend_titles")("nonsense xyz"))
        assert data["confidence"] in ("low", "medium")
        assert data["note"], "a weak match must be flagged, not silently returned"

    def test_no_hits_reports_none(self, corpus, monkeypatch):
        monkeypatch.setattr(corpus, "_vector_search", lambda q, k, *a, **kw: [])
        data = json.loads(_tool("knowledge_base___recommend_titles")("anything"))
        assert data["total"] == 0
        assert data["confidence"] == "none"


class TestMockIsOnlyAFallback:
    def test_the_mock_catalog_is_used_only_when_nothing_is_indexed(self, corpus, monkeypatch):
        monkeypatch.setattr(
            corpus, "_build_catalog_from_qdrant",
            lambda: {"corpora": {}, "total_books": 0, "total_chunks": 0},
        )
        assert corpus._live_catalog()["source"] == "mock"

    def test_a_live_catalog_is_not_labelled_mock(self, corpus):
        assert corpus._live_catalog().get("source") != "mock"
