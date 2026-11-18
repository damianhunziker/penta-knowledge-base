#!/usr/bin/env python3
"""The manager must show the truth: real text vs a summary, and which scope a
binding applies to.

The failure it exists for is the quiet one: a corpus full of hand-written
summaries looks like a working knowledge base in every list until a reader opens
a chunk and finds metadata. The verdict is therefore part of the data layer, not
a display detail — the CLI, the tools and the tests all read the same row.
"""

import importlib

import pytest


# ── fixtures ────────────────────────────────────────────────────────────────


class _Point:
    def __init__(self, payload):
        self.payload = payload


def _chunk(corpus, book_id, text, title="T", authors=None, source="", categories=None):
    return _Point(
        {
            "corpus": corpus,
            "book_id": book_id,
            "title": title,
            "authors": authors or [],
            "text": text,
            "source_ref": source,
            "tier": "standard",
            "categories": categories or [],
        }
    )


REAL_TEXT = "word " * 260  # ≈1300 chars: a real paragraph
FRONTMATTER = "---\ntitle: X\nauthors: [Y]\n---\n"
BULLET = "- Move from time-based to value-based pricing\n"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "kb.db"))
    from langgraph_crew_knowledge_base import assignments

    importlib.reload(assignments)
    monkeypatch.setattr(assignments, "current_scope", lambda: {"customer": "", "project": ""})
    return assignments


@pytest.fixture
def corpus(monkeypatch):
    """Replace the Qdrant read with a fixed set of points."""
    from langgraph_crew_knowledge_base import manager as M

    points = (
        [_chunk("real-base", "real-book", REAL_TEXT, "A Real Book", ["Auth"], "src/a.md")] * 6
        + [_chunk("summary-base", "summary-book", FRONTMATTER, "A Summary")] * 1
        + [_chunk("summary-base", "summary-book", BULLET, "A Summary")] * 7
    )
    monkeypatch.setattr(M, "_scroll_points", lambda: points, raising=False)
    return M


# ── corpus quality ──────────────────────────────────────────────────────────


class TestCorpusVerdict:
    def test_real_text_is_marked_real(self, corpus):
        rows = {r["book_id"]: r for r in corpus.corpus_rows()}
        assert rows["real-book"]["is_real"] is True

    def test_frontmatter_and_bullets_are_not_real(self, corpus):
        rows = {r["book_id"]: r for r in corpus.corpus_rows()}
        assert rows["summary-book"]["is_real"] is False

    def test_frontmatter_never_counts_as_real_text(self, corpus):
        rows = {r["book_id"]: r for r in corpus.corpus_rows()}
        assert rows["summary-book"]["real_chunks"] == 0

    def test_counts_are_per_book(self, corpus):
        rows = {r["book_id"]: r for r in corpus.corpus_rows()}
        assert rows["real-book"]["chunks"] == 6
        assert rows["summary-book"]["chunks"] == 8

    def test_bases_group_by_corpus(self, corpus):
        bases = {b["base"]: b for b in corpus.bases(corpus.corpus_rows())}
        assert set(bases) == {"real-base", "summary-base"}
        assert len(bases["summary-base"]["books"]) == 1
        assert bases["summary-base"]["real_books"] == 0
        assert bases["real-base"]["real_books"] == 1


class TestFind:
    def test_matches_title_author_and_id(self, corpus):
        books = corpus.corpus_rows()
        assert [b["book_id"] for b in corpus.find("real book", books, ("title",))] == ["real-book"]
        assert [b["book_id"] for b in corpus.find("auth", books, ("authors",))] == ["real-book"]
        assert [b["book_id"] for b in corpus.find("summary-book", books, ("book_id",))] == ["summary-book"]

    def test_empty_query_returns_everything(self, corpus):
        books = corpus.corpus_rows()
        assert len(corpus.find("", books, ("title",))) == len(books)


# ── bindings ────────────────────────────────────────────────────────────────


class TestBindings:
    def test_grant_resolves_to_a_title_and_base(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="deployment")
        rows = corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        assert rows[0]["title"] == "A Real Book"
        assert rows[0]["base"] == "real-base"
        assert rows[0]["missing"] is False

    def test_grant_to_a_deleted_book_is_flagged(self, corpus, store):
        store.assign("a_agent", books=["gone"], scope="deployment")
        rows = corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        assert rows[0]["missing"] is True

    def test_scope_order_is_widest_first(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="deployment")
        store.assign("a_agent", books=["real-book"], scope="client", customer="acme")
        store.assign("a_agent", books=["real-book"], scope="project",
                     customer="acme", project="p1")
        rows = corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        assert [r["scope"] for r in rows] == ["deployment", "client", "project"]


class TestNamespaces:
    def test_namespace_rows_carry_the_change_counts(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="client", customer="acme")
        store.assign("b_agent", books=["real-book"], scope="project",
                     customer="acme", project="p1")
        store.assign("c_agent", books=["real-book"], scope="deployment")
        rows = corpus.namespaces(
            corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        )
        # One row per existing project; the client-wide grant folds into it.
        assert [(r["customer"], r["project"]) for r in rows] == [("acme", "p1")]
        assert rows[0]["project_specific"] == 1
        assert rows[0]["client_wide"] == 1

    def test_a_client_only_grant_is_its_own_row(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="client", customer="acme")
        rows = corpus.namespaces(
            corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        )
        assert [(r["customer"], r["project"]) for r in rows] == [("acme", "")]
        assert "whole client" in rows[0]["label"]

    def test_deployment_grants_are_not_namespace_rows(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="deployment")
        rows = corpus.namespaces(
            corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        )
        assert rows == []


class TestEffective:
    def test_the_ladder_is_cumulative(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="deployment")
        store.assign("a_agent", books=["summary-book"], scope="client", customer="acme")
        bindings = corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())

        in_scope = corpus.effective_for("a_agent", "acme", "p1", bindings)
        assert {r["book_id"] if "book_id" in r else r["value"] for r in in_scope} == {
            "real-book", "summary-book"
        }

        other_client = corpus.effective_for("a_agent", "beta", "p1", bindings)
        assert [r["value"] for r in other_client] == ["real-book"]

    def test_a_project_grant_does_not_leak(self, corpus, store):
        store.assign("a_agent", books=["real-book"], scope="project",
                     customer="acme", project="p1")
        bindings = corpus.binding_rows(books=corpus.corpus_rows(), grants_rows=store.grants())
        assert corpus.effective_for("a_agent", "acme", "p2", bindings) == []


# ── the command ─────────────────────────────────────────────────────────────


class TestChatCommand:
    def test_overview_never_raises(self, corpus, store):
        from langgraph_crew_knowledge_base import manager_cli

        result = manager_cli.knowledge_cli("")
        assert result["success"] is True
        assert "Knowledge bases" in result["message"]

    def test_unknown_subcommand_is_reported(self, corpus, store):
        from langgraph_crew_knowledge_base import manager_cli

        result = manager_cli.knowledge_cli("nonsense")
        assert result["success"] is False
        assert "unknown sub-command" in result["message"]

    def test_wider_scope_is_refused_without_a_note(self, corpus, store):
        from langgraph_crew_knowledge_base import manager_cli

        result = manager_cli.knowledge_cli("assign a_agent real-book deployment")
        assert result["success"] is False
        assert "authorisation note" in result["message"]

    def test_project_scope_writes(self, corpus, store, monkeypatch):
        from langgraph_crew_knowledge_base import manager_cli

        # The script's Manager reads the real corpus; keep it on our fixture.
        monkeypatch.setattr(manager_cli._manager_module(), "Manager", _FakeManager)
        result = manager_cli.knowledge_cli("assign a_agent real-book project acme p1")
        assert result["success"] is True, result
        assert store.titles_for("a_agent", customer="acme", project="p1") == ["real-book"]


class _FakeManager:
    """The script's Manager with the fixture corpus instead of Qdrant."""

    def __init__(self):
        from langgraph_crew_knowledge_base import manager as M
        from langgraph_crew_knowledge_base.assignments import db_path

        self._M = M
        self.books = M.corpus_rows()
        self.bases = M.bases(self.books)
        self.bindings = M.binding_rows(self.books)
        self.namespaces = M.namespaces(self.bindings)
        self.db_path = str(db_path())

    find = staticmethod(lambda q, rows, fields: __import__(
        "langgraph_crew_knowledge_base.manager", fromlist=["find"]).find(q, rows, fields))

    def effective(self, agent_id, customer="", project=""):
        return self._M.effective_for(agent_id, customer, project, self.bindings)
