#!/usr/bin/env python3
"""Knowledge discovery runs ONE agent for both rounds, and injects independently.

Two things are pinned here:

**The run.** The node used to search the corpus in Python and stop. Three agent
manifests were declared and never executed. The scout now does round 1 (corpus),
round 2 (internet) and round 3 (install + assign) in one pass and returns one
JSON report of both rounds.

**The injection.** Tool discovery and knowledge discovery write into the same
worker prompt. They must not conflict: each block has its own heading, its own
guard against a duplicate, and neither may suppress the other. A worker with
both enabled receives both.
"""

import json
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ORCH = PLUGIN_ROOT.parent / "orchestration"
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

pytest.importorskip("langgraph_crew_knowledge_base")


REPORT = {
    "step": "knowledge_discovery",
    "task": "recherchier eüber klimwandel",
    "round1_existing": {
        "queries": ["climate change", "klimawandel"],
        "found": [
            {
                "book_id": "principles-project-management-williams",
                "kb_ref": "knowledge_base:project-manager/principles-project-management-williams",
                "title": "Principles of Project Management",
                "authors": ["Williams"],
                "confidence": "medium",
                "reason": "risk framing applies",
                "read_first": ["ch3"],
            }
        ],
        "gap": "no climate title in the corpus",
    },
    "round2_web": {
        "queries": ["IPCC AR6 synthesis pdf"],
        "candidates": [
            {
                "title": "IPCC AR6 Synthesis Report",
                "source_url": "https://ipcc.ch/report/ar6/syr.pdf",
                "license": "CC-BY-4.0",
                "license_verified": True,
                "relevance": 0.9,
                "reason": "authoritative",
            }
        ],
        "needs_decision": [
            {"title": "Paywalled Paper", "url": "https://x/y", "question": "licence unclear"}
        ],
    },
    "round3_installed": {
        "chunked": [{"book_id": "ipcc-ar6", "corpus": "climate", "status": "indexed", "chunks": 40}],
        "assigned": [
            {"agent_id": "project_manager_agent", "book_ids": ["ipcc-ar6"],
             "scope": "project", "customer": "C", "project": "P", "why": "task needs it"}
        ],
        "skipped": [],
    },
    "answer_for_worker": "No climate book locally; IPCC AR6 is licensed CC-BY and fills the gap.",
}


class TestJsonExtraction:
    def test_fenced_json_is_read(self):
        from langgraph_crew_knowledge_base.discovery import _extract_json

        assert _extract_json("noise\n```json\n{\"a\": 1}\n```\ntail") == {"a": 1}

    def test_bare_json_is_read(self):
        from langgraph_crew_knowledge_base.discovery import _extract_json

        assert _extract_json('before {"a": 2} after') == {"a": 2}

    def test_prose_without_json_returns_none(self):
        from langgraph_crew_knowledge_base.discovery import _extract_json

        assert _extract_json("I could not find anything useful.") is None


class TestReportToFindings:
    def test_both_rounds_reach_the_worker_findings(self):
        from langgraph_crew_knowledge_base.discovery import report_as_findings

        findings = report_as_findings(REPORT)
        assert findings["assigned_titles"] == ["principles-project-management-williams"]
        assert findings["recommendations"][0]["title"] == "IPCC AR6 Synthesis Report"
        assert findings["gap"] == "no climate title in the corpus"
        assert findings["needs_decision"][0]["title"] == "Paywalled Paper"
        assert findings["answer_for_worker"].startswith("No climate book locally")
        # The complete report survives: decisions and rejections are part of it.
        assert findings["report"]["round2_web"]["candidates"]
        assert findings["assignments"][0]["scope"] == "project"

    def test_a_failed_run_is_not_an_empty_success(self):
        from langgraph_crew_knowledge_base.discovery import _failed, report_as_findings

        failed = _failed({"task": "t"}, "no agent runtime")
        findings = report_as_findings(failed)
        assert findings["error"] == "no agent runtime"
        assert "did not complete" in findings["answer_for_worker"]


class TestRenderedBlock:
    def test_the_worker_block_carries_both_rounds(self):
        from langgraph_crew_orchestration.nodes.executor.utils.helpers import (
            build_knowledge_discovery_report_message,
        )
        from langgraph_crew_knowledge_base.discovery import report_as_findings

        task = {"id": "t", "description": "recherchier eüber klimwandel",
                "requires_knowledge_discovery": True}
        msg = build_knowledge_discovery_report_message(
            {"knowledge_discovery_findings": report_as_findings(REPORT)}, task
        )
        assert msg.startswith("KNOWLEDGE DISCOVERY RESULT")
        assert "ANSWER:" in msg
        assert "CORPUS GAP: no climate title" in msg
        assert "DISCOVERY REPORT (both rounds, complete)" in msg
        assert "round1_existing" in msg and "round2_web" in msg
        assert "WAITING FOR A HUMAN DECISION" in msg


class TestIndependentInjection:
    """Both discovery blocks must coexist in one worker prompt."""

    def _state(self):
        from langgraph_crew_knowledge_base.discovery import report_as_findings

        return {
            "knowledge_discovery_findings": report_as_findings(REPORT),
            # Minimal shape the tool block needs to render at all.
            "web_tool_discovery_report": {
                "recommendations": [],
                "mcp_installed": [],
                "api_installed": [],
                "refused": [],
                "deferred": [],
                "notes": [],
                "findings": {"scout": {"titles": ["some tool"]}},
            },
            "web_tool_discovery_injected_for_tasks": [],
        }

    def test_both_blocks_have_distinct_headings(self):
        from langgraph_crew_web_discovery.flow import SUGGESTIONS_HEADING
        from langgraph_crew_knowledge_base.discovery import report_as_findings

        kb_block = "KNOWLEDGE DISCOVERY RESULT"
        assert SUGGESTIONS_HEADING != kb_block
        # Neither heading may be a prefix of the other: the executor's duplicate
        # check is startswith(), so a shared prefix would hide one block.
        assert not SUGGESTIONS_HEADING.startswith(kb_block)
        assert not kb_block.startswith(SUGGESTIONS_HEADING)
        assert report_as_findings(REPORT)["step"] == "knowledge_discovery"

    def test_knowledge_block_ignores_the_tool_report(self):
        """Tool discovery being present or absent must not change the KB block."""
        from langgraph_crew_orchestration.nodes.executor.utils.helpers import (
            build_knowledge_discovery_report_message,
        )

        task = {"id": "t", "description": "recherchier eüber klimwandel",
                "requires_knowledge_discovery": True}
        with_tools = build_knowledge_discovery_report_message(self._state(), task)
        without = build_knowledge_discovery_report_message(
            {"knowledge_discovery_findings": self._state()["knowledge_discovery_findings"]},
            task,
        )
        assert with_tools == without
