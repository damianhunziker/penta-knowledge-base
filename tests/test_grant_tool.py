#!/usr/bin/env python3
"""Granting access must write the RIGHT file, and only that file.

The access decision has two halves: the store row (what an agent may read) and
the manifest at the matching ladder level (where the decision is durable). The
dangerous failure is writing the second half into the wrong scope — a grant
meant for one project landing in the deployment folder, or a grant for the
deployment silently editing a project's hand-written manifest.
"""

import importlib

import pytest


AGENT_MANIFEST = """---
spec_version: 1
kind: agent
name: tester
description: A hand-written description that must survive a grant.
x-penta:
  knowledge_base:
    titles: [already-here]
    chunks: 7
---

The body must survive too: it is the agent's role prompt.
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A repo with custom/, a project vault, and an isolated store."""
    repo = tmp_path / "repo"
    for sub in ("app", "agents", "custom/agents", "plugins"):
        (repo / sub).mkdir(parents=True)
    project = tmp_path / "client_vaults" / "acme" / "Projekte" / "web"
    for sub in ("agents", "workflows", "policies", "skills"):
        (project / sub).mkdir(parents=True)

    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "kb.db"))
    monkeypatch.setenv("ACTIVE_VAULT_PATH", str(project))
    monkeypatch.setenv("CLIENT_VAULTS_BASE_PATH", str(project.parents[2]))

    from app.core import access_scope as sc
    from app.core import global_state
    from app.core.spec import loader
    from langgraph_crew_knowledge_base import assignments

    sc.clear_scope()
    sc.set_scope("acme", "web")
    global_state.set_active_vault_path(str(project))
    monkeypatch.setattr(loader, "_repo_root", lambda: str(repo))
    importlib.reload(assignments)

    yield {"repo": repo, "project": project, "tmp": tmp_path}

    sc.clear_scope()
    global_state.clear_active_vault_path()


def _tools():
    from langgraph_crew_knowledge_base import tools

    return tools


class TestProjectScope:
    def test_project_grant_writes_the_project_manifest(self, env):
        import json

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent", scope="project", titles=["book-a"]
            )
        )
        path = env["project"] / "agents" / "tester.md"
        assert result["manifest"]["written"] is True, result["manifest"]
        assert path.exists()
        text = path.read_text()
        assert "book-a" in text
        assert "this project only" in result["active"]

    def test_project_grant_carries_the_active_project(self, env):
        import json

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access("tester_agent", scope="project",
                                                titles=["book-a"])
        )
        assert result["scope"] == {
            "scope": "project",
            "customer": "acme",
            "project": "web",
        }

    def test_no_scope_is_refused_with_a_hint(self, env, monkeypatch):
        import json

        from langgraph_crew_knowledge_base import assignments

        monkeypatch.setattr(
            assignments, "current_scope", lambda: {"customer": "", "project": ""}
        )
        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access("tester_agent", scope="project",
                                                titles=["book-a"])
        )
        assert result["stored"] is False
        assert "namespace___set_namespace" in json.dumps(result)


class TestWiderScopes:
    def test_client_requires_authorization(self, env):
        import json

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent", scope="client", customer="acme", titles=["book-a"]
            )
        )
        # The store row is written (the grant is active) but the durable
        # declaration is refused — the caller must ask the human first.
        assert result["manifest"]["written"] is False
        assert "explicit instruction" in str(result["manifest"]["error"])

    def test_deployment_writes_the_custom_manifest(self, env):
        import json

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent",
                scope="deployment",
                titles=["book-a"],
                authorized_by="user",
                authorization_note="operator: everyone may use this title",
            )
        )
        path = env["repo"] / "custom" / "agents" / "tester.md"
        assert result["manifest"]["written"] is True, result["manifest"]
        assert path.exists()
        assert "book-a" in path.read_text()

    def test_wider_scope_never_edits_a_more_specific_manifest(self, env):
        """A deployment grant must not rewrite a project's hand-written file."""
        import json

        project_manifest = env["project"] / "agents" / "tester.md"
        project_manifest.write_text(AGENT_MANIFEST)

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent",
                scope="deployment",
                titles=["book-a"],
                authorized_by="user",
                authorization_note="operator",
            )
        )
        assert project_manifest.read_text() == AGENT_MANIFEST, (
            "the project manifest must be untouched by a deployment-level grant"
        )
        assert result["manifest"]["path"] == str(
            env["repo"] / "custom" / "agents" / "tester.md"
        )


class TestMerge:
    def test_existing_manifest_is_merged_not_replaced(self, env):
        import json

        custom = env["repo"] / "custom" / "agents" / "tester.md"
        custom.write_text(AGENT_MANIFEST)

        tools = _tools()
        result = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent",
                scope="deployment",
                titles=["book-b"],
                authorized_by="user",
                authorization_note="operator",
            )
        )
        text = custom.read_text()
        assert result["manifest"]["written"] is True, result["manifest"]
        assert "already-here" in text, "an existing title was dropped"
        assert "book-b" in text, "the new title was not added"
        assert "chunks: 7" in text, "another x-penta key was dropped"
        assert "The body must survive too" in text, "the body was dropped"
        assert "A hand-written description" in text

    def test_grant_is_idempotent_in_the_manifest(self, env):
        import json

        tools = _tools()
        first = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent", scope="deployment", titles=["book-a"],
                authorized_by="user", authorization_note="operator",
            )
        )
        second = json.loads(
            tools.knowledge_base___grant_access(
                "tester_agent", scope="deployment", titles=["book-a"],
                authorized_by="user", authorization_note="operator",
            )
        )
        assert first["manifest"]["titles"] == second["manifest"]["titles"] == ["book-a"]


class TestAccessMatrix:
    def test_matrix_lists_every_rung(self, env):
        import json

        from langgraph_crew_knowledge_base import assignments

        assignments.assign("a_agent", books=["d"], scope="deployment")
        assignments.assign("a_agent", books=["c"], scope="client", customer="acme")
        assignments.assign(
            "b_agent", books=["p"], scope="project", customer="acme", project="web"
        )

        tools = _tools()
        matrix = json.loads(tools.knowledge_base___access_matrix())
        assert matrix["total_rows"] == 3
        assert matrix["active_scope"]["customer"] == "acme"
        assert {g["scope"] for g in matrix["agents"]["a_agent"]} == {
            "deployment",
            "client",
        }
        assert matrix["titles"]["book:p"][0]["agent_id"] == "b_agent"
        assert "grant_access" in matrix["how_to_write"]
