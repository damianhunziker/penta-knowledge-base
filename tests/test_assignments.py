#!/usr/bin/env python3
"""Agent→title assignments must persist across processes.

Before this the assignment lived in a module dict: one process could set it and
the next run saw nothing, which looked like "the title was not assigned".

Scope-aware behaviour (which rung a grant applies to) is covered by
``test_scopes.py``; this file is about the store itself.
"""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "kb.db"))
    from langgraph_crew_knowledge_base import assignments

    importlib.reload(assignments)
    # No active namespace: pin the scope so the tests never read the real one.
    monkeypatch.setattr(assignments, "current_scope", lambda: {"customer": "", "project": ""})
    return assignments


class TestPersistence:
    def test_assign_then_read(self, store):
        store.assign("dmm", books=["eyal-2014-hooked"], scope="deployment")
        assert store.list_for_agent("dmm", scope="deployment") == ["book:eyal-2014-hooked"]

    def test_survives_a_new_connection(self, store):
        """The data is on disk, not in a dict that dies with the process."""
        store.assign("dmm", books=["a-book"], scope="deployment")
        importlib.reload(store)
        assert store.list_for_agent("dmm", scope="deployment") == ["book:a-book"]

    def test_idempotent(self, store):
        store.assign("dmm", books=["x"], scope="deployment")
        second = store.assign("dmm", books=["x"], scope="deployment")
        assert second["assigned"] == [], "a repeated assignment must be a no-op"
        assert store.list_for_agent("dmm", scope="deployment") == ["book:x"]

    def test_all_kinds(self, store):
        store.assign(
            "dmm", books=["b"], categories=["c"], tags=["t"], scope="deployment"
        )
        assert set(store.list_for_agent("dmm", scope="deployment")) == {
            "book:b",
            "category:c",
            "tag:t",
        }

    def test_books_for_agent_filters(self, store):
        store.assign("dmm", books=["b1", "b2"], categories=["c"], scope="deployment")
        assert store.books_for_agent("dmm") == ["b1", "b2"]

    def test_agents_are_separate(self, store):
        store.assign("a", books=["x"], scope="deployment")
        store.assign("b", books=["y"], scope="deployment")
        assert store.books_for_agent("a") == ["x"]
        assert store.books_for_agent("b") == ["y"]

    def test_unassign(self, store):
        store.assign("dmm", books=["x"], scope="deployment")
        assert store.unassign("dmm", "book", "x", scope="deployment") is True
        assert store.list_for_agent("dmm", scope="deployment") == []

    def test_unknown_agent_is_empty(self, store):
        assert store.list_for_agent("nobody", scope="deployment") == []

    def test_empty_agent_id_is_an_error(self, store):
        assert store.assign("", books=["x"]).get("error")


class TestToolsUseTheStore:
    def test_assign_tool_reports_all(self, store, monkeypatch):
        from langgraph_crew_knowledge_base import tools

        payload = tools.knowledge_base___assign_to_agent(
            "dmm", book_ids=["eyal-2014-hooked"], assigned_by="operator",
            scope="deployment",
        )
        import json

        data = json.loads(payload)
        assert data["assigned"] == ["book:eyal-2014-hooked"]
        assert data["assigned_by"] == "operator"
        assert data["scope"]["scope"] == "deployment"

    def test_list_tool_reads_the_store(self, store):
        from langgraph_crew_knowledge_base import tools
        import json

        store.assign("dmm", books=["x"], scope="deployment")
        data = json.loads(tools.knowledge_base___list_assignments("dmm", scope="deployment"))
        assert data["assignments"] == ["book:x"]
