#!/usr/bin/env python3
"""A discovery timeout must not erase what the run already did.

Reproduced from a real run: the scout searched the corpus, researched the web,
chunked two vault documents and assigned them to ``project_manager_agent`` at
project scope — then hit the 180 s limit. ``asyncio.wait_for`` cancelled it and
the node reported "the scout run failed: TimeoutError", discarding a report of
work that was already visible in Qdrant and the assignment table.

The rule: a timeout is reported as a timeout, AND the effects are read back from
the store so the worker is told what is now in effect.
"""

import asyncio
import json

import pytest

pytest.importorskip("langgraph_crew_knowledge_base")


def _challenge(**over):
    base = {"task": "recherchier eüber klimwandel", "agent": "project_manager_agent",
            "customer": "Damian", "project": "SandboxTest", "namespace": ""}
    base.update(over)
    return base


class TestTimeoutDefault:
    def test_the_budget_is_long_enough_for_three_rounds(self):
        """180 s cut off a run that had installed two documents."""
        from langgraph_crew_knowledge_base import discovery

        assert discovery.DEFAULT_TIMEOUT_S >= 600

    def test_the_budget_is_configurable(self, monkeypatch):
        import importlib

        from langgraph_crew_knowledge_base import discovery

        monkeypatch.setenv("KNOWLEDGE_DISCOVERY_TIMEOUT", "1200")
        reloaded = importlib.reload(discovery)
        assert reloaded.DEFAULT_TIMEOUT_S == 1200.0
        monkeypatch.delenv("KNOWLEDGE_DISCOVERY_TIMEOUT")
        importlib.reload(discovery)


class TestTimeoutReport:
    def test_assignments_made_before_the_deadline_are_reported(self, monkeypatch):
        from langgraph_crew_knowledge_base import discovery, ladder

        monkeypatch.setattr(
            ladder,
            "resolve_agent",
            lambda agent_id, customer="", project="": {
                "store": [
                    {"kind": "book", "value": "klimawandel-forschung-2024-baseline",
                     "scope": "project", "level": "project",
                     "customer": customer, "project": project},
                    {"kind": "book", "value": "climate-change-research-update-2026",
                     "scope": "project", "level": "project",
                     "customer": customer, "project": project},
                ]
            },
        )
        report = discovery._timed_out(_challenge(), 180, {})

        assert report["_meta"]["source"] == "timeout"
        assigned = report["round3_installed"]["assigned"]
        assert len(assigned) == 2
        assert assigned[0]["scope"] == "project"
        assert "assign" in report["answer_for_worker"].lower()
        assert "klimawandel-forschung-2024-baseline" in report["answer_for_worker"]
        # It must still say it timed out — the summary is genuinely missing.
        assert "limit" in report["error"]

    def test_a_timeout_with_no_side_effects_says_only_that(self, monkeypatch):
        from langgraph_crew_knowledge_base import discovery, ladder

        monkeypatch.setattr(
            ladder, "resolve_agent", lambda *a, **k: {"store": []}
        )
        report = discovery._timed_out(_challenge(), 180, {})
        assert report["round3_installed"]["assigned"] == []
        assert "limit" in report["error"]

    def test_a_real_timeout_reaches_the_node_as_a_visible_outcome(self, monkeypatch):
        """End to end: an agent that never answers, with the limit hit."""
        from langgraph_crew_knowledge_base import discovery

        class Hanging:
            async def _execute_task_with_tools(self, prompt, messages, system_prompt):
                await asyncio.sleep(10)

        monkeypatch.setattr(
            discovery, "_system_prompt", lambda challenge: "prompt"
        )
        report = asyncio.run(
            discovery.run_discovery("task", {}, agent=Hanging(), timeout_s=0.05)
        )
        assert report["_meta"]["source"] == "timeout"
        assert report["error"]
        assert "did not complete" in report["answer_for_worker"]

    def test_a_good_run_still_parses_its_json(self, monkeypatch):
        from langgraph_crew_knowledge_base import discovery

        payload = {"round1_existing": {"found": [], "gap": "none"},
                   "answer_for_worker": "done"}

        class Answering:
            async def _execute_task_with_tools(self, prompt, messages, system_prompt):
                class R:
                    content = "```json\n" + json.dumps(payload) + "\n```"
                return R()

        monkeypatch.setattr(discovery, "_system_prompt", lambda challenge: "prompt")
        report = asyncio.run(discovery.run_discovery("t", {}, agent=Answering(), timeout_s=5))
        assert report["_meta"]["source"] == "llm"
        assert report["answer_for_worker"] == "done"
