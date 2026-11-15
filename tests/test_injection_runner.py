#!/usr/bin/env python3
"""Tests for the retrieval → injection block builder."""

import json

import pytest

from langgraph_crew_knowledge_base import runner
from langgraph_crew_knowledge_base.settings import KnowledgeSettings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "KNOWLEDGE_INJECT_AFTER_TOOLS",
        "KNOWLEDGE_INJECT_CHUNKS",
        "KNOWLEDGE_TASK_CHUNKS",
        "DISABLE_KNOWLEDGE_DISCOVERY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _hit(i: int) -> dict:
    return {
        "kb_ref": f"knowledge_base:academic/book/ch{i}",
        "title": "Book",
        "authors": ["Doe, Jane"],
        "year": 2020,
        "section_path": f"ch{i}",
        "tier": "foundational",
        "book_id": "book",
        "corpus": "academic",
        "text": f"Passage number {i}.",
        "score": 0.9,
    }


@pytest.fixture
def fake_search(monkeypatch):
    """Record the requested limit and return that many hits."""
    seen = {"limit": None, "queries": []}

    def _search(query, corpora=None, tier=None, top_k=10, **kw):
        seen["limit"] = top_k
        seen["queries"].append(query)
        return json.dumps({"results": [_hit(i) for i in range(1, top_k + 1)]})

    import langgraph_crew_knowledge_base.tools as tools_mod

    monkeypatch.setattr(tools_mod, "knowledge_base___search", _search, raising=False)
    return seen


class TestTaskBlock:
    def test_default_is_three_chunks(self, fake_search):
        block = runner.build_task_block({"id": "t1", "description": "systems thinking"})
        assert fake_search["limit"] == 3
        assert block.startswith(runner.TASK_BLOCK_HEADER)
        assert block.count("↳ read section") == 3

    def test_task_chunks_override(self, fake_search):
        block = runner.build_task_block(
            {"id": "t1", "description": "x", "knowledge_chunks": 5}
        )
        assert fake_search["limit"] == 5
        assert block.count("↳ read section") == 5

    def test_env_default(self, fake_search, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "2")
        block = runner.build_task_block({"id": "t1", "description": "x"})
        assert fake_search["limit"] == 2
        assert block.count("↳ read section") == 2

    def test_zero_chunks_gives_no_block(self, fake_search):
        block = runner.build_task_block(
            {"id": "t1", "description": "x", "knowledge_chunks": 0}
        )
        assert block == ""
        assert fake_search["limit"] is None

    def test_disabled_gives_no_block(self, fake_search, monkeypatch):
        monkeypatch.setenv("DISABLE_KNOWLEDGE_DISCOVERY", "true")
        assert runner.build_task_block({"id": "t1", "description": "x"}) == ""

    def test_no_query_gives_no_block(self, fake_search):
        assert runner.build_task_block({"id": "t1"}) == ""

    def test_task_value_beats_env_and_inject_chunks(self, fake_search, monkeypatch):
        """The per-task count is authoritative — env supplies only the default."""
        monkeypatch.setenv("KNOWLEDGE_INJECT_CHUNKS", "2")
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "1")
        runner.build_task_block({"id": "t1", "description": "x", "knowledge_chunks": 9})
        assert fake_search["limit"] == 9


class TestMidRunBlock:
    def test_uses_inject_chunks(self, fake_search, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_CHUNKS", "4")
        block = runner.build_mid_run_block("current work")
        assert fake_search["limit"] >= 4
        assert block.count("↳ read section") == 4
        assert block.startswith(runner.MID_RUN_BLOCK_HEADER)

    def test_default_is_three(self, fake_search):
        block = runner.build_mid_run_block("current work")
        assert block.count("↳ read section") == 3

    def test_seen_refs_are_skipped(self, fake_search):
        seen = ["knowledge_base:academic/book/ch1", "knowledge_base:academic/book/ch2"]
        block = runner.build_mid_run_block("q", seen=seen)
        assert "ch1" not in block
        assert "ch2" not in block
        assert "ch3" in block

    def test_state_refs_are_skipped(self, fake_search):
        state = {"knowledge_injected_refs": ["knowledge_base:academic/book/ch1"]}
        block = runner.build_mid_run_block("q", state=state)
        assert "ch1]" not in block

    def test_threshold_zero_disables(self, fake_search, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "0")
        assert runner.build_mid_run_block("q") == ""

    def test_empty_query(self, fake_search):
        assert runner.build_mid_run_block("") == ""


class TestLowScoreFiltering:
    def test_chunks_below_min_score_are_dropped(self, monkeypatch):
        import langgraph_crew_knowledge_base.tools as tools_mod

        def _search(query, corpora=None, tier=None, top_k=10, **kw):
            hits = [_hit(1), {**_hit(2), "score": 0.01}]
            return json.dumps({"results": hits})

        monkeypatch.setattr(tools_mod, "knowledge_base___search", _search, raising=False)
        block = runner.build_task_block({"id": "t1", "description": "x"})
        assert "ch1" in block
        assert "ch2" not in block


class TestSearchFailure:
    def test_conversion_error_yields_empty_block(self, monkeypatch):
        import langgraph_crew_knowledge_base.tools as tools_mod

        def _boom(*a, **kw):
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(tools_mod, "knowledge_base___search", _boom, raising=False)
        assert runner.build_task_block({"id": "t1", "description": "x"}) == ""
        assert runner.build_mid_run_block("x") == ""
