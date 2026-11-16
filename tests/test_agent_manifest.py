#!/usr/bin/env python3
"""An agent may declare its corpus binding in description language.

``x-penta.knowledge_base`` in an agent manifest pins titles/categories to the
agent without a tool call, and follows the normal ladder
(project → client → plugin → custom → core), so a deployment can override a
plugin agent by dropping ``agents/<name>.md`` into ``custom/``.
"""

import pytest

from app.core.spec.models import AgentManifest, KnowledgeBaseConfig


class TestSchema:
    def test_block_is_optional(self):
        manifest = AgentManifest(name="xy", description="d")
        assert manifest.x_penta.knowledge_base is None

    def test_fields_are_parsed(self):
        manifest = AgentManifest.model_validate(
            {
                "name": "dmm",
                "description": "Digital marketing manager",
                "x-penta": {
                    "knowledge_base": {
                        "titles": ["a", "b"],
                        "categories": ["c"],
                        "tags": ["t"],
                        "chunks": 5,
                        "enabled": True,
                    }
                },
            }
        )
        kb = manifest.x_penta.knowledge_base
        assert isinstance(kb, KnowledgeBaseConfig)
        assert kb.titles == ["a", "b"]
        assert kb.categories == ["c"]
        assert kb.chunks == 5

    def test_defaults(self):
        kb = KnowledgeBaseConfig()
        assert kb.titles == []
        assert kb.chunks is None
        assert kb.enabled is True

    def test_unknown_field_is_rejected(self):
        with pytest.raises(Exception):
            KnowledgeBaseConfig.model_validate({"title": ["typo"]})


class TestNodeReadsTheManifest:
    def _state(self, role):
        return {
            "messages": [],
            "current_task": "make a plan",
            "knowledge_discovery_enabled": True,
            "knowledge_discovery_completed_for_tasks": [],
            "task_queue": [
                {"id": "task_1", "description": "make a plan", "status": "ready",
                 "role": role, "requires_knowledge_discovery": True}
            ],
            "current_plan": [],
        }

    def test_manifest_titles_reach_the_findings(self, monkeypatch):
        from langgraph_crew_knowledge_base import node as kb_node

        monkeypatch.setattr(
            kb_node,
            "_manifest_knowledge_config",
            lambda agent: {"titles": ["meadows-2008"], "categories": ["systems-thinking"],
                           "tags": [], "chunks": 2, "enabled": True},
        )
        monkeypatch.setattr(kb_node, "_assigned_titles_for", lambda a, s: [])
        monkeypatch.setattr(kb_node, "_knowledge_base_disabled", lambda: True)

        import asyncio

        cmd = asyncio.run(
            kb_node.KnowledgeDiscoveryNode(verbose=False)
            ._run_discovery_workflow(self._state("some_agent"))
        )
        findings = cmd.update["knowledge_discovery_findings"]
        assert findings["assigned_titles"] == ["meadows-2008"]
        assert findings["assigned_categories"] == ["systems-thinking"]

    def test_manifest_disabled_short_circuits(self, monkeypatch):
        from langgraph_crew_knowledge_base import node as kb_node

        monkeypatch.setattr(
            kb_node,
            "_manifest_knowledge_config",
            lambda agent: {"enabled": False},
        )
        import asyncio

        cmd = asyncio.run(
            kb_node.KnowledgeDiscoveryNode(verbose=False).execute(self._state("some_agent"))
        )
        findings = cmd.update["knowledge_discovery_findings"]
        assert findings["assigned_titles"] == []
        assert "manifest" in (findings["corpus"].get("reason") or "")
