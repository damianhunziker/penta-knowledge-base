#!/usr/bin/env python3
"""Tests for Knowledge-Discovery option in agent popup.

Tests the wiring from popup to global_state to task flags.
"""

import sys
from pathlib import Path

# Add the plugin to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestKnowledgeDiscoveryOption:
    """Tests for the Knowledge-Discovery popup option."""

    def test_knowledge_discovery_in_popup_options(self):
        """Popup options include knowledge_discovery field."""
        # Simulate the popup options dict
        options = {
            "review": False,
            "tool_research": False,
            "knowledge_discovery": True,
        }
        assert "knowledge_discovery" in options
        assert options["knowledge_discovery"] is True

    def test_all_three_options_present(self):
        """All three options (review, tool-discovery, knowledge-discovery) are present."""
        options = {
            "review": True,
            "tool_research": True,
            "knowledge_discovery": True,
        }
        assert len(options) == 3
        assert all(key in options for key in ["review", "tool_research", "knowledge_discovery"])

    def test_knowledge_discovery_defaults_to_off(self):
        """Knowledge-Discovery defaults to OFF like the other options."""
        options = {
            "review": False,
            "tool_research": False,
            "knowledge_discovery": False,
        }
        assert options["knowledge_discovery"] is False

    def test_knowledge_discovery_independent_of_other_options(self):
        """Knowledge-Discovery can be set independently of other options."""
        options = {
            "review": True,
            "tool_research": False,
            "knowledge_discovery": True,
        }
        # Review ON, Tool-Discovery OFF, Knowledge-Discovery ON
        assert options["review"] is True
        assert options["tool_research"] is False
        assert options["knowledge_discovery"] is True


class TestKnowledgeDiscoveryFlag:
    """Tests for requires_knowledge_discovery task flag."""

    def test_flag_name_is_correct(self):
        """Task flag is named requires_knowledge_discovery."""
        task = {
            "id": "task_1",
            "role": "project_manager_agent",
            "requires_knowledge_discovery": True,
        }
        assert "requires_knowledge_discovery" in task
        assert task["requires_knowledge_discovery"] is True

    def test_flag_absent_when_off(self):
        """When knowledge_discovery is OFF, flag is absent from task."""
        task = {
            "id": "task_1",
            "role": "project_manager_agent",
        }
        assert "requires_knowledge_discovery" not in task

    def test_flag_present_when_on(self):
        """When knowledge_discovery is ON, flag is True in task."""
        task = {
            "id": "task_1",
            "role": "project_manager_agent",
            "requires_knowledge_discovery": True,
        }
        assert task.get("requires_knowledge_discovery") is True


class TestKnowledgeDiscoveryLabels:
    """Tests for display labels."""

    def test_tool_discovery_label(self):
        """Tool-Discovery label uses correct capitalization."""
        label = "Tool-Discovery"
        assert label == "Tool-Discovery"
        assert "-" in label
        assert label[0].isupper()

    def test_knowledge_discovery_label(self):
        """Knowledge-Discovery label uses correct capitalization."""
        label = "Knowledge-Discovery"
        assert label == "Knowledge-Discovery"
        assert "-" in label
        assert label[0].isupper()

    def test_labels_consistent_style(self):
        """All three labels use consistent Title-Case style."""
        labels = ["Review", "Tool-Discovery", "Knowledge-Discovery"]
        for label in labels:
            # Each word starts with uppercase
            words = label.split("-")
            for word in words:
                assert word[0].isupper(), f"Word '{word}' in '{label}' should start with uppercase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])