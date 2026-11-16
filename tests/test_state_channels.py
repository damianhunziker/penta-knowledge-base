#!/usr/bin/env python3
"""The knowledge-discovery state keys must be declared LangGraph channels.

Regression: the executor routed to ``knowledge_discovery``, the node returned
``goto="executor"`` and marked the task complete — but the key
``knowledge_discovery_completed_for_tasks`` was missing from
``EnhancedCrewState``. LangGraph drops updates for undeclared channels, so the
executor never saw the marker and the two nodes ping-ponged until the
recursion limit. The run appeared to "hang" in the executor.

These tests pin the channel declarations; without them the loop comes back.
"""

import pytest

from app.core.enhanced_state import EnhancedCrewState

pytest.importorskip("langgraph", reason="langgraph not installed")


class TestKnowledgeDiscoveryChannels:
    """Every key the node writes must be a declared state channel."""

    NODE_UPDATES = (
        "knowledge_discovery_completed_for_tasks",
        "knowledge_discovery_enabled",
        "knowledge_discovery_findings",
    )

    def test_keys_are_declared(self):
        for key in self.NODE_UPDATES:
            assert key in EnhancedCrewState.__annotations__, (
                f"{key} is not a declared channel - LangGraph will silently "
                "drop the node's update and the executor loops"
            )

    def test_completion_list_uses_a_reducer(self):
        """The declaration must carry a reducer, not a bare ``List[str]``.

        ``from __future__ import annotations`` turns the annotation into a
        string, so check the source instead of ``get_args``.
        """
        import inspect

        from app.core import enhanced_state

        src = inspect.getsource(enhanced_state)
        line = next(
            ln
            for ln in src.splitlines()
            if ln.strip().startswith("knowledge_discovery_completed_for_tasks:")
        )
        assert "Annotated" in line, (
            "knowledge_discovery_completed_for_tasks needs "
            "Annotated[List[str], reducer] so updates merge with the old list"
        )

    def test_tool_discovery_parity(self):
        """Knowledge discovery declares the same keys as tool discovery."""
        for key in self.NODE_UPDATES:
            assert key in EnhancedCrewState.__annotations__

    def test_defaults_exist(self):
        """The workflow reset must carry the keys, not rely on ``.get`` fallbacks."""
        from app.core.state_reset import workflow_reset_update

        defaults = workflow_reset_update()
        assert defaults.get("knowledge_discovery_completed_for_tasks") == []
        assert defaults.get("knowledge_discovery_enabled") is False
        assert "knowledge_discovery_findings" in defaults


class TestKnowledgeInjectionChannels:
    """The injection hook's counters must be declared channels too.

    The hook writes them during the ReAct loop; an undeclared key would be
    dropped by LangGraph and the counter would restart on every LLM call.
    """

    NODE_UPDATES = (
        "knowledge_inject_baseline",
        "knowledge_inject_fired",
        "knowledge_injected_refs",
    )

    def test_keys_are_declared(self):
        for key in self.NODE_UPDATES:
            assert key in EnhancedCrewState.__annotations__, (
                f"{key} is not a declared channel - the injection counter "
                "would reset on every LLM call"
            )

    def test_defaults_exist(self):
        from app.core.state_reset import workflow_reset_update

        defaults = workflow_reset_update()
        assert defaults.get("knowledge_inject_baseline") is None
        assert defaults.get("knowledge_inject_fired") == 0
        assert defaults.get("knowledge_injected_refs") == []
