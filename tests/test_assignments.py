#!/usr/bin/env python3
"""Agent→title assignments must persist and be readable in description language.

Before this the assignment lived in a module dict: one process could set it and
the next run saw nothing, which looked like "the title was not assigned".
"""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "kb.db"))
    from langgraph_crew_knowledge_base import assignments

    importlib.reload(assignments)
    return assignments


class TestPersistence:
    def test_assign_then_read(self, store):
        store.assign("dmm", books=["eyal-2014-hooked"])
        assert store.list_for_agent("dmm") == ["book:eyal-2014-hooked"]

    def test_survives_a_new_connection(self, store):
        """The data is on disk, not in a dict that dies with the process."""
        store.assign("dmm", books=["a-book"])
        # A fresh module instance reads the same file.
        importlib.reload(store)
        assert store.list_for_agent("dmm") == ["book:a-book"]

    def test_idempotent(self, store):
        store.assign("dmm", books=["x"])
        second = store.assign("dmm", books=["x"])
        assert second["assigned"] == [], "a repeated assignment must be a no-op"
        assert store.list_for_agent("dmm") == ["book:x"]

    def test_all_kinds(self, store):
        store.assign("dmm", books=["b"], categories=["c"], tags=["t"])
        assert set(store.list_for_agent("dmm")) == {"book:b", "category:c", "tag:t"}

    def test_books_for_agent_filters(self, store):
        store.assign("dmm", books=["b1", "b2"], categories=["c"])
        assert store.books_for_agent("dmm") == ["b1", "b2"]

    def test_agents_are_separate(self, store):
        store.assign("a", books=["x"])
        store.assign("b", books=["y"])
        assert store.list_for_agent("a") == ["book:x"]
        assert store.list_for_agent("b") == ["book:y"]

    def test_unassign(self, store):
        store.assign("dmm", books=["x"])
        assert store.unassign("dmm", "book", "x") is True
        assert store.list_for_agent("dmm") == []

    def test_unknown_agent_is_empty(self, store):
        assert store.list_for_agent("nobody") == []

    def test_empty_agent_id_is_an_error(self, store):
        assert store.assign("", books=["x"]).get("error")


class TestToolsUseTheStore:
    def test_assign_tool_reports_all(self, store, monkeypatch):
        from langgraph_crew_knowledge_base import tools

        payload = tools.knowledge_base___assign_to_agent(
            "dmm", book_ids=["eyal-2014-hooked"], assigned_by="operator"
        )
        import json

        data = json.loads(payload)
        assert data["assigned"] == ["book:eyal-2014-hooked"]
        assert data["total_assignments"] == 1
        assert data["assigned_by"] == "operator"

    def test_list_tool_reads_the_store(self, store):
        from langgraph_crew_knowledge_base import tools
        import json

        store.assign("dmm", books=["x"])
        data = json.loads(tools.knowledge_base___list_assignments("dmm"))
        assert data["assignments"] == ["book:x"]
