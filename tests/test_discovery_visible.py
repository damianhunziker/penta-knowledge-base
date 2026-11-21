#!/usr/bin/env python3
"""Knowledge discovery must be visible in the transcript, always.

The node routed in, ran, found that no passage cleared the relevance floor and
returned to the executor **printing nothing**. In the chat CLI that is
indistinguishable from the router ignoring the flag: the operator sees
"ROUTING KNOWLEDGE_DISCOVERY" followed immediately by the executor and has no
way to tell whether discovery ran, skipped, or failed.

These tests drive the REAL workflow (``_run_discovery_workflow``) — the one that
builds the summary and prints the line — with the search stubbed, so the
reporting code is the code under test.
"""

import asyncio
import json
import re

import pytest

pytest.importorskip("langgraph_crew_knowledge_base")


def _plain(text: str) -> str:
    """Strip ANSI: Rich styles every token, so a raw substring check fails."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _state(**over):
    base = {
        "messages": [],
        "task_queue": [
            {
                "id": "task_1",
                "description": "recherchier eüber klimwandel",
                "role": "project_manager_agent",
                "requires_knowledge_discovery": True,
                "status": "ready",
            }
        ],
        "current_plan": [],
        "knowledge_discovery_enabled": True,
        "knowledge_discovery_completed_for_tasks": [],
    }
    base.update(over)
    return base


@pytest.fixture
def node():
    from langgraph_crew_knowledge_base.node import KnowledgeDiscoveryNode

    return KnowledgeDiscoveryNode(verbose=False)


@pytest.fixture
def stub_search(monkeypatch):
    """Stub the corpus search + gate at the module the node imports from."""

    def _install(hits, kept_refs):
        import langgraph_crew_knowledge_base.tools as tools
        import langgraph_crew_knowledge_base.runner as runner

        monkeypatch.setattr(
            tools,
            "knowledge_base___search",
            lambda **kw: json.dumps({"results": hits, "total": len(hits)}),
        )
        monkeypatch.setattr(tools, "kb_status", lambda: {"ready": True, "reason": "", "chunks": 5613})

        class _Ref:
            def __init__(self, kb_ref):
                self.kb_ref = kb_ref
                self.book_id = kb_ref

        class _Chunk:
            def __init__(self, kb_ref):
                self.reference = _Ref(kb_ref)

        def fake_apply(chunks, query, settings, fill_target=0):
            return [c for c in chunks if c.reference.kb_ref in kept_refs]

        # The node imports both names from this module inside the function.
        monkeypatch.setattr(runner, "apply_relevance", fake_apply)
        monkeypatch.setattr(
            runner, "_to_chunk", lambda h: _Chunk(h["kb_ref"]) if isinstance(h, dict) else None
        )

    return _install


class TestVisibleOutcome:
    def test_zero_passages_is_reported_not_silent(self, node, stub_search, capsys):
        """A ready corpus with nothing above the floor must still say so."""
        hits = [
            {"kb_ref": "kb:1", "book_id": "b1", "text": "irrelevant", "score": 0.82, "title": "B1"},
            {"kb_ref": "kb:2", "book_id": "b1", "text": "irrelevant", "score": 0.81, "title": "B1"},
        ]
        stub_search(hits, kept_refs=set())  # the gate drops everything

        command = asyncio.run(node.execute(_state()))
        out = _plain(capsys.readouterr().out)

        assert "KNOWLEDGE DISCOVERY" in out, "the node printed nothing"
        assert "nothing above the relevance floor" in out
        assert "0 passage" in out

        findings = command.update["knowledge_discovery_findings"]
        assert findings["passages"] == []
        summary = command.update["messages"][-1].content
        assert "do not cover it" in summary or "relevance gate" in summary

    def test_a_passage_result_reports_the_count(self, node, stub_search, capsys):
        hits = [
            {"kb_ref": "kb:1", "book_id": "b1", "text": "relevant", "score": 0.80, "title": "B1"},
        ]
        stub_search(hits, kept_refs={"kb:1"})

        command = asyncio.run(node.execute(_state()))
        out = _plain(capsys.readouterr().out)

        assert "KNOWLEDGE DISCOVERY" in out
        assert "1 passage" in out
        assert command.update["knowledge_discovery_findings"]["passages"]
        assert "nothing above the relevance floor" not in out

    def test_a_repeat_entry_does_not_recount(self, node, stub_search):
        """The loop may re-enter; completion is recorded so it does not redo work."""
        stub_search([], kept_refs=set())
        first = asyncio.run(node.execute(_state()))
        done = first.update["knowledge_discovery_completed_for_tasks"]
        assert done

        second = asyncio.run(
            node.execute(_state(knowledge_discovery_completed_for_tasks=done))
        )
        assert second.goto == "executor"
        assert "knowledge_discovery_findings" not in (second.update or {})
