#!/usr/bin/env python3
"""The search must be honest about an unindexed corpus.

Before this, ``knowledge_base___search`` was a keyword mock over a hardcoded
catalog: a real query ("marketing strategie") returned an empty list with no
explanation, so knowledge discovery looked like a skipped step even though the
node had run correctly.
"""

import json

import pytest

from langgraph_crew_knowledge_base import tools as T


@pytest.fixture
def no_corpus(monkeypatch):
    """No indexed corpus (the state of a fresh install)."""
    monkeypatch.setattr(T, "_collection_ready", lambda: False)
    monkeypatch.setattr(
        T,
        "kb_status",
        lambda: {"ready": False, "reason": "collection missing", "chunks": 0},
    )


class TestStatus:
    def test_reports_missing_collection(self, monkeypatch):
        class Client:
            def collection_exists(self, name):
                return False

        monkeypatch.setattr(T, "_qdrant", lambda: Client())
        status = T.kb_status()
        assert status["ready"] is False
        assert "does not exist" in status["reason"]

    def test_reports_empty_collection(self, monkeypatch):
        class Info:
            points_count = 0

        class Client:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return Info()

        monkeypatch.setattr(T, "_qdrant", lambda: Client())
        status = T.kb_status()
        assert status["ready"] is False
        assert "empty" in status["reason"]

    def test_reports_ready(self, monkeypatch):
        class Info:
            points_count = 42

        class Client:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return Info()

        monkeypatch.setattr(T, "_qdrant", lambda: Client())
        status = T.kb_status()
        assert status["ready"] is True
        assert status["chunks"] == 42

    def test_unreachable_qdrant(self, monkeypatch):
        monkeypatch.setattr(T, "_qdrant", lambda: None)
        status = T.kb_status()
        assert status["ready"] is False
        assert "unreachable" in status["reason"]


class TestSearchHonesty:
    def test_unindexed_corpus_says_so(self, no_corpus):
        raw = T.knowledge_base___search(query="marketing strategie", top_k=3)
        payload = json.loads(raw)
        assert payload["results"] == []
        assert payload["source"] == "mock-catalog"
        assert "No indexed corpus" in payload["warning"]
        assert "ingest_from_root" in payload["warning"]

    def test_vector_hits_are_preferred(self, monkeypatch):
        hits = [
            {
                "kb_ref": "knowledge_base:academic/x/ch1",
                "title": "Book",
                "text": "passage",
                "score": 0.88,
                "book_id": "x",
                "corpus": "academic",
            }
        ]
        monkeypatch.setattr(T, "_vector_search", lambda *a, **kw: hits)
        payload = json.loads(T.knowledge_base___search(query="anything", top_k=3))
        assert payload["source"] == "qdrant"
        assert payload["results"][0]["kb_ref"] == "knowledge_base:academic/x/ch1"

    def test_no_vector_hits_falls_back_with_warning(self, no_corpus):
        payload = json.loads(T.knowledge_base___search(query="zzz unknown", top_k=3))
        assert payload["source"] == "mock-catalog"
        assert "warning" in payload


class TestNodeReportsMissingCorpus:
    def test_node_summary_explains_the_missing_corpus(self, no_corpus):
        import asyncio

        from langgraph_crew_knowledge_base import node as kb_node

        state = {
            "messages": [],
            "current_task": "marketing strategie",
            "knowledge_discovery_enabled": True,
            "knowledge_discovery_completed_for_tasks": [],
            "task_queue": [
                {"id": "task_1", "description": "marketing strategie",
                 "requires_knowledge_discovery": True}
            ],
            "current_plan": [],
        }
        kb_node._node_cache.clear()
        cmd = asyncio.run(kb_node.knowledge_discovery_node(state))
        kb_node._node_cache.clear()

        findings = cmd.update["knowledge_discovery_findings"]
        assert findings["corpus"]["ready"] is False
        last = cmd.update["messages"][-1].content
        assert "no corpus" in last.lower()
        assert "ingest" in last.lower()
