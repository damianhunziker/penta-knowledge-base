#!/usr/bin/env python3
"""Every task gets corpus passages — not only tasks that ran discovery.

The bug this pins: the plugin's cold path (``runner.build_task_block``) was
implemented and unit-tested but **never called by the runtime**. It was only
reachable through a knowledge-discovery run, which the executor triggers solely
for tasks carrying ``requires_knowledge_discovery``. So a normal task — the
common case — reached the worker with zero corpus material, and "the knowledge
base injected nothing" became the expected result instead of a bug.
"""

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

RENDERED = (
    "KNOWLEDGE DISCOVERY RESULT\n"
    "The knowledge base found these passages for this task.\n\n"
    "[kb:graphic-design/interaction-of-color · p.12 · score 0.81]\n"
    'Josef Albers, *Interaction of Color* (1963)\n\n'
    "> Colour is the most relative medium in art.\n"
)


@pytest.fixture
def cold_block(monkeypatch):
    """A runtime with a working cold path but no discovery findings."""
    from app.core import knowledge_seam

    monkeypatch.setattr(knowledge_seam, "get_knowledge_base_provider", lambda: object())

    def fake_build(task, state=None):
        return RENDERED if task and task.get("description") else ""

    monkeypatch.setattr(knowledge_seam, "build_task_knowledge_block", fake_build)
    # The helper imports the seam function lazily; make that import see ours.
    monkeypatch.setitem(
        sys.modules, "app.core.knowledge_seam", knowledge_seam
    )
    return knowledge_seam


def _helper():
    # The orchestration plugin ships in this repo but is not installed into the
    # venv; import it from its own directory.
    orchestration_root = Path(__file__).resolve().parents[2] / "orchestration"
    if str(orchestration_root) not in sys.path:
        sys.path.insert(0, str(orchestration_root))
    from langgraph_crew_orchestration.nodes.executor.utils.helpers import (
        build_knowledge_discovery_report_message,
    )

    return build_knowledge_discovery_report_message


TASK = {"id": "task_1", "description": "Design decisions for frontend color palettes"}


class TestColdInjection:
    def test_a_task_without_the_flag_still_gets_a_block(self, cold_block):
        """The regression: this used to return ""."""
        block = _helper()({}, TASK)
        assert block, "a task without requires_knowledge_discovery must still get corpus material"
        assert "Interaction of Color" in block

    def test_no_findings_in_state_is_not_a_reason_to_skip(self, cold_block):
        block = _helper()({"knowledge_discovery_findings": None}, TASK)
        assert block

    def test_an_empty_corpus_yields_nothing(self, monkeypatch):
        from app.core import knowledge_seam

        monkeypatch.setattr(knowledge_seam, "get_knowledge_base_provider", lambda: None)
        assert _helper()({}, TASK) == ""

    def test_a_task_can_opt_out(self, monkeypatch):
        """``knowledge_chunks: 0`` is the task's explicit "no literature"."""
        from langgraph_crew_knowledge_base import runner

        monkeypatch.delenv("KNOWLEDGE_TASK_CHUNKS", raising=False)
        assert runner.build_task_block({**TASK, "knowledge_chunks": 0}) == ""


class TestDiscoveryWins:
    def test_findings_are_used_when_the_task_ran_discovery(self, cold_block):
        state = {
            "knowledge_discovery_findings": {
                "assigned_titles": ["interaction-of-color"],
                "assigned_categories": [],
                "recommendations": [],
                "passages": [
                    {"kb_ref": "knowledge_base:x/y#s:z", "text": "A passage."}
                ],
                "chunking_jobs": [],
            }
        }
        task = {**TASK, "requires_knowledge_discovery": True}
        block = _helper()(state, task)
        assert "assigned titles" in block.lower() or "ASSIGNED TITLES" in block
        assert "A passage." in block
        # Not the cold block: discovery findings are more specific.
        assert "Interaction of Color" not in block


class TestSeam:
    def test_seam_is_silent_without_the_plugin(self, monkeypatch):
        from app.core import knowledge_seam

        monkeypatch.setattr(knowledge_seam, "get_knowledge_base_provider", lambda: None)
        assert knowledge_seam.build_task_knowledge_block(TASK, {}) == ""

    def test_seam_survives_a_broken_provider(self, monkeypatch):
        from app.core import knowledge_seam

        class Boom:
            def build_task_block(self, task, state=None):
                raise RuntimeError("qdrant down")

        monkeypatch.setattr(
            knowledge_seam, "get_knowledge_base_provider", lambda: Boom()
        )
        assert knowledge_seam.build_task_knowledge_block(TASK, {}) == ""
