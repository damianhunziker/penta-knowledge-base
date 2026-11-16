#!/usr/bin/env python3
"""Root and Qdrant resolution must work in both deployment shapes.

Two failures made an agent report "structural blocker: Qdrant down":

1. The client was bound to ONE hardcoded URL. Inside the container the service
   is ``qdrant:6333``, on the host it is ``127.0.0.1:16333`` — whichever was
   not first looked like an outage.
2. ``project-vault:`` was rooted at the customer vault, while callers write
   ``Projects/<p>/Knowledge/x.md`` (vaultforge does). The result was
   ``…/Projects/<p>/Projects/<p>/…`` and "File not found".
"""

import pytest

from langgraph_crew_knowledge_base import ingest as I
from langgraph_crew_knowledge_base import tools as T


class TestQdrantUrls:
    def test_tries_both_shapes(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("QDRANT_HOST_URL", "http://127.0.0.1:16333")
        urls = T._qdrant_urls()
        assert "http://qdrant:6333" in urls
        assert "http://127.0.0.1:16333" in urls

    def test_deduplicates(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("QDRANT_HOST_URL", "http://qdrant:6333")
        assert T._qdrant_urls().count("http://qdrant:6333") == 1

    def test_no_env_still_has_a_candidate(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_HOST_URL", raising=False)
        assert T._qdrant_urls()


class TestRootResolution:
    def test_project_prefixed_path_is_not_doubled(self, monkeypatch):
        """A project-rooted base must strip a duplicated Projects/<p> prefix."""
        monkeypatch.setattr(
            I, "_get_vault_path_for_root",
            lambda name: "/vault/client_vaults/Damian/Projects/SandboxTest",
        )
        monkeypatch.setattr(I, "_host_path", lambda p: p)
        resolved = I._resolve_root_path(
            "project-vault:Projects/SandboxTest/Knowledge/a.md"
        )
        assert resolved == "/vault/client_vaults/Damian/Projects/SandboxTest/Knowledge/a.md"
        assert "Projects/Projects" not in resolved

    def test_plain_relative_path_works_too(self, monkeypatch):
        monkeypatch.setattr(
            I, "_get_vault_path_for_root",
            lambda name: "/vault/client_vaults/Damian/Projects/SandboxTest",
        )
        monkeypatch.setattr(I, "_host_path", lambda p: p)
        resolved = I._resolve_root_path("project-vault:Knowledge/a.md")
        assert resolved == "/vault/client_vaults/Damian/Projects/SandboxTest/Knowledge/a.md"

    def test_unknown_root_still_raises(self, monkeypatch):
        monkeypatch.setattr(I, "_host_path", lambda p: p)
        with pytest.raises(ValueError):
            I._resolve_root_path("not-a-root:x.md")


class TestHostPathTranslation:
    def test_existing_path_is_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setattr(I.os.path, "exists", lambda p: p == str(tmp_path))
        assert I._host_path(str(tmp_path)) == str(tmp_path)

    def test_container_path_maps_to_host_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VAULT_CONTAINER_ROOT", "/vault")
        monkeypatch.setenv("VAULT_HOST_ROOT", str(tmp_path))
        target = tmp_path / "client_vaults" / "x.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

        real_exists = I.os.path.exists
        monkeypatch.setattr(
            I.os.path, "exists",
            lambda p: (p == str(target)) if str(p).startswith("/vault/") else real_exists(p),
        )
        assert I._host_path("/vault/client_vaults/x.md") == str(target)


class TestNotationTolerance:
    """Every notation a model produced in the field must resolve.

    An agent wrote ``project-vault:Projects/<customer>/Projects/<p>/…``
    although the tree is ``<customer>/Projects/<p>/…``; the result was
    "File not found" and the agent reported the knowledge base as broken.
    """

    @pytest.fixture
    def vault(self, tmp_path, monkeypatch):
        base = tmp_path / "client_vaults"
        proj = base / "E2E_Stage2Test" / "Projects" / "ProjectX" / "Knowledge"
        proj.mkdir(parents=True)
        (proj / "book.md").write_text("# Book\n\ntext\n", encoding="utf-8")
        return base

    @pytest.mark.parametrize(
        "rel",
        [
            "Knowledge/book.md",
            "Projects/ProjectX/Knowledge/book.md",
            "Projects/E2E_Stage2Test/Projects/ProjectX/Knowledge/book.md",
            "E2E_Stage2Test/Projects/ProjectX/Knowledge/book.md",
        ],
    )
    def test_all_notations_find_the_file(self, vault, rel):
        found = I._probe_vault_for(rel, str(vault))
        assert found is not None, f"{rel!r} did not resolve"
        assert found.endswith("Knowledge/book.md")

    def test_unknown_file_still_misses(self, vault):
        assert I._probe_vault_for("Knowledge/nope.md", str(vault)) is None
