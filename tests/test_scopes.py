#!/usr/bin/env python3
"""Assignments must be SCOPED: a grant applies where it was made, no wider.

The failure this prevents is quiet and expensive: a book given to an agent in
one project silently becoming available to that agent in every other project
of the customer (or of every customer). Each case below is one rung of the
ladder, plus the migration of rows that predate scoping.
"""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "kb.db"))
    from langgraph_crew_knowledge_base import assignments

    importlib.reload(assignments)
    return assignments


@pytest.fixture
def no_scope(store, monkeypatch):
    """No active customer/project (the CLI outside a namespace)."""
    monkeypatch.setattr(store, "current_scope", lambda: {"customer": "", "project": ""})
    return store


class TestScopeLadder:
    def test_deployment_is_visible_everywhere(self, store):
        store.assign("pm", books=["global-book"], scope="deployment")
        assert store.titles_for("pm", customer="acme", project="web") == ["global-book"]
        assert store.titles_for("pm", customer="beta", project="other") == ["global-book"]

    def test_client_is_visible_in_every_project_of_that_customer(self, store):
        store.assign("pm", books=["acme-book"], scope="client", customer="acme")
        assert store.titles_for("pm", customer="acme", project="web") == ["acme-book"]
        assert store.titles_for("pm", customer="acme", project="shop") == ["acme-book"]
        assert store.titles_for("pm", customer="beta", project="web") == []

    def test_project_does_not_leak(self, store):
        store.assign("pm", books=["web-book"], scope="project", customer="acme", project="web")
        assert store.titles_for("pm", customer="acme", project="web") == ["web-book"]
        assert store.titles_for("pm", customer="acme", project="shop") == [], (
            "a project-level grant must not reach a sibling project"
        )
        assert store.titles_for("pm", customer="beta", project="web") == []

    def test_rungs_are_cumulative(self, store):
        store.assign("pm", books=["d"], scope="deployment")
        store.assign("pm", books=["c"], scope="client", customer="acme")
        store.assign("pm", books=["p"], scope="project", customer="acme", project="web")
        assert store.titles_for("pm", customer="acme", project="web") == ["c", "d", "p"]
        assert store.titles_for("pm", customer="acme", project="other") == ["c", "d"]

    def test_agents_stay_separate_per_scope(self, store):
        store.assign("a", books=["x"], scope="deployment")
        store.assign("b", books=["y"], scope="project", customer="acme", project="web")
        assert store.titles_for("a", customer="acme", project="web") == ["x"]
        assert store.titles_for("b", customer="acme", project="web") == ["y"]


class TestScopeRefusals:
    def test_unknown_scope_is_an_error(self, store):
        result = store.assign("pm", books=["x"], scope="galaxy")
        assert "scope must be one of" in result["error"]

    def test_project_scope_needs_a_project(self, no_scope):
        result = no_scope.assign("pm", books=["x"], scope="project")
        assert "needs a customer and a project" in result["error"]

    def test_client_scope_needs_a_customer(self, no_scope):
        result = no_scope.assign("pm", books=["x"], scope="client")
        assert "needs a customer" in result["error"]

    def test_deployment_drops_scope_identifiers(self, store):
        result = store.assign(
            "pm", books=["x"], scope="deployment", customer="acme", project="web"
        )
        assert result["scope"] == {"scope": "deployment", "customer": "", "project": ""}


class TestRows:
    def test_same_value_at_two_scopes_are_two_rows(self, store):
        store.assign("pm", books=["x"], scope="deployment")
        store.assign("pm", books=["x"], scope="client", customer="acme")
        rows = [g for g in store.grants() if g["value"] == "x"]
        assert {r["scope"] for r in rows} == {"deployment", "client"}

    def test_repeat_is_still_a_no_op(self, store):
        store.assign("pm", books=["x"], scope="deployment")
        assert store.assign("pm", books=["x"], scope="deployment")["assigned"] == []

    def test_unassign_is_scoped(self, store):
        store.assign("pm", books=["x"], scope="deployment")
        store.assign("pm", books=["x"], scope="client", customer="acme")
        assert store.unassign("pm", "book", "x", scope="client", customer="acme") is True
        rows = [g for g in store.grants() if g["value"] == "x"]
        assert [r["scope"] for r in rows] == ["deployment"], (
            "unassigning one rung must leave the other alone"
        )

    def test_grants_report_the_scope(self, store):
        store.assign("pm", books=["x"], scope="project", customer="acme", project="web")
        grant = store.grants()[0]
        assert grant["scope"] == "project"
        assert grant["customer"] == "acme"
        assert grant["project"] == "web"
        assert grant["agent_id"] == "pm"


class TestLegacyRows:
    def test_rows_without_scope_stay_visible(self, tmp_path, monkeypatch):
        """An upgrade must not silently delete every existing grant."""
        import sqlite3

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE assignment ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " agent_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,"
            " assigned_by TEXT NOT NULL DEFAULT 'operator', created_at TEXT NOT NULL,"
            " UNIQUE (agent_id, kind, value))"
        )
        conn.execute(
            "INSERT INTO assignment (agent_id, kind, value, created_at)"
            " VALUES ('old', 'book', 'old-book', '2020-01-01')"
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(db))
        from langgraph_crew_knowledge_base import assignments

        importlib.reload(assignments)

        assert assignments.titles_for("old", customer="c", project="p") == ["old-book"]
        assert assignments.grants()[0]["scope"] == assignments.LEGACY_SCOPE

    def test_write_works_after_migration(self, tmp_path, monkeypatch):
        import sqlite3

        db = tmp_path / "legacy2.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE assignment ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " agent_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,"
            " assigned_by TEXT NOT NULL DEFAULT 'operator', created_at TEXT NOT NULL,"
            " UNIQUE (agent_id, kind, value))"
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(db))
        from langgraph_crew_knowledge_base import assignments

        importlib.reload(assignments)

        result = assignments.assign("pm", books=["new"], scope="deployment")
        assert result.get("error") is None
        assert result["assigned"] == ["book:new"]


class TestStatePath:
    def test_repo_state_dir_is_outside_the_plugin(self, tmp_path, monkeypatch):
        """The default must not be the plugin dir: it is read-only in the container."""
        from langgraph_crew_knowledge_base import state_paths

        monkeypatch.delenv("KNOWLEDGE_DB_PATH", raising=False)
        path = state_paths.resolve_db_path(None)
        assert path.name == "knowledge_base.db"
        assert path.parent != state_paths.PLUGIN_DIR, (
            f"state must not live in the (read-only) plugin dir: {path}"
        )

    def test_explicit_file_is_honoured(self, tmp_path, monkeypatch):
        from langgraph_crew_knowledge_base import state_paths

        monkeypatch.delenv("KNOWLEDGE_DB_PATH", raising=False)
        target = tmp_path / "custom.db"
        assert state_paths.resolve_db_path(str(target)) == target
