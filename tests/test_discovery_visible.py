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
def stub_scout(monkeypatch):
    """Replace the scout's LLM run with a canned two-round report."""

    def _install(found=(), candidates=(), error=""):
        import langgraph_crew_knowledge_base.discovery as discovery

        report = {
            "step": "knowledge_discovery",
            "task": "recherchier eüber klimwandel",
            "round1_existing": {"queries": ["climate"], "found": list(found),
                                "gap": "no climate title"},
            "round2_web": {"queries": ["ipcc"], "candidates": list(candidates),
                           "needs_decision": []},
            "round3_installed": {"chunked": [], "assigned": [], "skipped": []},
            "answer_for_worker": "Nothing matched." if not (found or candidates)
            else "Use the sources above.",
        }
        if error:
            report = discovery._failed({"task": "t"}, error)

        async def fake_run(task, state, *, agent=None, timeout_s=None):
            return report

        monkeypatch.setattr(discovery, "run_discovery", fake_run)

    return _install


class TestVisibleOutcome:
    def test_nothing_found_is_reported_not_silent(self, node, stub_scout, capsys):
        """A run that finds nothing must still say so."""
        stub_scout()

        command = asyncio.run(node.execute(_state()))
        out = _plain(capsys.readouterr().out)

        assert "KNOWLEDGE DISCOVERY" in out, "the node printed nothing"
        assert "0 corpus title" in out
        summary = command.update["messages"][-1].content
        assert "do not cover it" in summary or "Neither the corpus" in summary

    def test_a_found_title_reports_the_count(self, node, stub_scout, capsys):
        stub_scout(found=[{"book_id": "b1", "kb_ref": "kb:1", "title": "B1"}])

        command = asyncio.run(node.execute(_state()))
        out = _plain(capsys.readouterr().out)

        assert "KNOWLEDGE DISCOVERY" in out
        assert "1 corpus title" in out
        # The found title joins the agent's standing assignments — both are in
        # scope for the task, so only membership is asserted.
        assert "b1" in command.update["knowledge_discovery_findings"]["assigned_titles"]

    def test_a_failed_run_says_so(self, node, stub_scout, capsys):
        stub_scout(error="the scout run failed: TimeoutError")

        command = asyncio.run(node.execute(_state()))
        out = _plain(capsys.readouterr().out)
        summary = command.update["messages"][-1].content

        assert "failed" in out
        assert "did not complete" in summary

    def test_a_repeat_entry_does_not_recount(self, node, stub_scout):
        """The loop may re-enter; completion is recorded so it does not redo work."""
        stub_scout()
        first = asyncio.run(node.execute(_state()))
        done = first.update["knowledge_discovery_completed_for_tasks"]
        assert done

        second = asyncio.run(
            node.execute(_state(knowledge_discovery_completed_for_tasks=done))
        )
        assert second.goto == "executor"
        assert "knowledge_discovery_findings" not in (second.update or {})
