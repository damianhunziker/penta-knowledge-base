#!/usr/bin/env python3
"""The level view and the raw document store.

Two things this pins down:

**The ladder.** Manifests are the base, the store is the override layer, and
the runtime merges both. The manager used to show only the store, so a binding
that came from a manifest looked absent. ``ladder.resolve_agent`` answers one
question — what may this agent read, and where did each part come from — and
these tests hold it to that.

**The raw store.** Chunking used to be one-way: the text survived, the file did
not. Measured on the live corpus, 23 of 25 source PDFs were already gone. Every
passage now carries ``raw_ref`` into a content-addressed store, and a citation
can say whether the file still resolves.
"""

import json
from pathlib import Path

import pytest


# ── the ladder ───────────────────────────────────────────────────────────────


class TestLadder:
    def test_manifest_and_store_are_both_reported(self, monkeypatch):
        """A title from a file and one from the store must both appear."""
        from langgraph_crew_knowledge_base import ladder

        monkeypatch.setattr(
            ladder,
            "_manifest_entry",
            lambda agent_id: {
                "agent_id": agent_id,
                "level": "plugin",
                "path": "plugins/x/agents/x.md",
                "titles": ["from-manifest"],
                "categories": [],
                "tags": [],
                "chunks": None,
                "enabled": True,
                "declared": True,
            },
        )
        monkeypatch.setattr(
            ladder,
            "_store_entries",
            lambda agent_id: [
                {"kind": "book", "value": "from-store", "scope": "deployment", "level": "custom"}
            ],
        )
        monkeypatch.setattr(ladder, "live_books", lambda: {"from-manifest", "from-store"})
        monkeypatch.setattr(
            "langgraph_crew_knowledge_base.assignments.books_for_agent",
            lambda agent_id, customer="", project="": ["from-store"],
        )

        resolved = ladder.resolve_agent("x_agent")
        sources = {t["book_id"]: t["source"] for t in resolved["titles"]}
        assert sources == {"from-manifest": "manifest", "from-store": "store"}

    def test_a_title_in_both_sources_says_both(self, monkeypatch):
        from langgraph_crew_knowledge_base import ladder

        monkeypatch.setattr(
            ladder,
            "_manifest_entry",
            lambda agent_id: {"level": "plugin", "titles": ["shared"], "categories": [],
                              "tags": [], "path": "p", "declared": True, "enabled": True},
        )
        monkeypatch.setattr(
            ladder, "_store_entries",
            lambda agent_id: [{"kind": "book", "value": "shared", "scope": "deployment", "level": "custom"}],
        )
        monkeypatch.setattr(ladder, "live_books", lambda: {"shared"})
        monkeypatch.setattr(
            "langgraph_crew_knowledge_base.assignments.books_for_agent",
            lambda agent_id, customer="", project="": ["shared"],
        )
        resolved = ladder.resolve_agent("x_agent")
        assert resolved["titles"][0]["source"] == "both"

    def test_a_binding_to_a_deleted_book_is_marked_stale(self, monkeypatch):
        """12 of 30 live grants pointed at missing books — must be visible."""
        from langgraph_crew_knowledge_base import ladder

        monkeypatch.setattr(
            ladder, "_manifest_entry",
            lambda agent_id: {"level": "plugin", "titles": ["gone", "present"],
                              "categories": [], "tags": [], "path": "", "declared": True,
                              "enabled": True},
        )
        monkeypatch.setattr(ladder, "_store_entries", lambda agent_id: [])
        monkeypatch.setattr(ladder, "live_books", lambda: {"present"})
        monkeypatch.setattr(
            "langgraph_crew_knowledge_base.assignments.books_for_agent",
            lambda agent_id, customer="", project="": [],
        )
        resolved = ladder.resolve_agent("x_agent")
        assert resolved["stale"] == ["gone"]
        marks = {t["book_id"]: t["live"] for t in resolved["titles"]}
        assert marks == {"gone": False, "present": True}

    def test_an_unreadable_corpus_does_not_mark_everything_stale(self, monkeypatch):
        """`live_books() == set()` must read as 'unknown', never as 'all gone'."""
        from langgraph_crew_knowledge_base import ladder

        monkeypatch.setattr(
            ladder, "_manifest_entry",
            lambda agent_id: {"level": "plugin", "titles": ["a"], "categories": [],
                              "tags": [], "path": "", "declared": True, "enabled": True},
        )
        monkeypatch.setattr(ladder, "_store_entries", lambda agent_id: [])
        monkeypatch.setattr(ladder, "live_books", lambda: set())
        monkeypatch.setattr(
            "langgraph_crew_knowledge_base.assignments.books_for_agent",
            lambda agent_id, customer="", project="": [],
        )
        resolved = ladder.resolve_agent("x_agent")
        assert resolved["stale"] == []

    def test_store_scopes_map_onto_levels(self):
        from langgraph_crew_knowledge_base import ladder

        assert ladder.LEVEL_FOR_STORE_SCOPE["deployment"] == "custom"
        assert ladder.LEVEL_FOR_STORE_SCOPE["client"] == "client"
        assert ladder.LEVEL_FOR_STORE_SCOPE["project"] == "project"
        # Every level has a store scope it writes to.
        for level in ladder.LEVELS:
            assert level in ladder.STORE_SCOPE_FOR_LEVEL

    def test_matrix_places_each_agent_on_its_levels(self, monkeypatch):
        from langgraph_crew_knowledge_base import ladder

        monkeypatch.setattr(
            ladder, "_manifest_entry",
            lambda agent_id: {"level": "plugin", "titles": [f"{agent_id}-book"],
                              "categories": [], "tags": [], "path": "", "declared": True,
                              "enabled": True},
        )
        monkeypatch.setattr(ladder, "_store_entries", lambda agent_id: [])
        monkeypatch.setattr(ladder, "live_books", lambda: {f"{a}-book" for a in ("a_agent", "b_agent")})
        monkeypatch.setattr(
            "langgraph_crew_knowledge_base.assignments.books_for_agent",
            lambda agent_id, customer="", project="": [],
        )
        data = ladder.matrix(["a_agent", "b_agent"])
        assert data["agents"] == ["a_agent", "b_agent"]
        assert data["cells"]["plugin"]["a_agent"]["entries"][0]["value"] == "a_agent-book"


class TestPrune:
    def test_pruning_refuses_when_the_corpus_is_unreadable(self, monkeypatch):
        """Deleting on a failed read would wipe every binding."""
        from langgraph_crew_knowledge_base import ladder, level_editor

        monkeypatch.setattr(ladder, "live_books", lambda: set())
        monkeypatch.setattr(ladder, "resolve_agent", lambda *a, **k: {"titles": [], "store": []})
        result = level_editor.prune_agent("x_agent")
        assert result.get("error")
        assert "refusing" in result["error"]


# ── the raw store ────────────────────────────────────────────────────────────


@pytest.fixture
def raw(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_RAW_ROOT", str(tmp_path / "raw"))
    from langgraph_crew_knowledge_base import raw_store

    return raw_store


class TestRawStore:
    def test_a_document_is_stored_under_its_hash(self, raw, tmp_path):
        source = tmp_path / "book.pdf"
        source.write_bytes(b"%PDF-1.7 pretend")
        stored = raw.store_document(source, book_id="b", corpus="c", original_ref="vault:x.pdf")
        assert stored["ok"]
        assert stored["raw_ref"].startswith(raw.RAW_REF_PREFIX)
        assert Path(stored["path"]).is_file()
        index = Path(stored["path"]).parent / "index" / f"{stored['sha256']}.json"
        assert json.loads(index.read_text())["original_name"] == "book.pdf"

    def test_the_same_bytes_are_stored_once(self, raw, tmp_path):
        a = tmp_path / "a.pdf"
        a.write_bytes(b"identical")
        b = tmp_path / "b.pdf"
        b.write_bytes(b"identical")
        first = raw.store_document(a)
        second = raw.store_document(b)
        assert first["raw_ref"] == second["raw_ref"]
        assert len(list((tmp_path / "raw").glob("*.pdf"))) == 1

    def test_different_bytes_never_collide(self, raw, tmp_path):
        a = tmp_path / "same-name.pdf"
        a.write_bytes(b"one")
        ref_one = raw.store_document(a)["raw_ref"]
        a.write_bytes(b"two")
        ref_two = raw.store_document(a)["raw_ref"]
        assert ref_one != ref_two

    def test_a_link_carries_the_page_anchor(self, raw, tmp_path):
        source = tmp_path / "x.pdf"
        source.write_bytes(b"bytes")
        ref = raw.store_document(source)["raw_ref"]
        link = raw.link_for(ref, page=39)
        assert link["available"] is True
        assert link["page"] == 39
        assert link["page_ref"].endswith("#page=39")

    def test_a_missing_file_is_reported_not_hidden(self, raw):
        link = raw.link_for(f"{raw.RAW_REF_PREFIX}deadbeef.pdf")
        assert link["available"] is False
        assert link["reason"]

    def test_an_empty_ref_is_not_a_link(self, raw):
        assert raw.link_for("")["available"] is False

    def test_traversal_out_of_the_store_is_refused(self, raw):
        assert raw.resolve(f"{raw.RAW_REF_PREFIX}../../etc/passwd") is None

    def test_an_unsupported_type_is_refused_with_a_reason(self, raw, tmp_path):
        exe = tmp_path / "x.exe"
        exe.write_bytes(b"MZ")
        result = raw.store_document(exe)
        assert not result["ok"]
        assert "unsupported" in result["error"]
