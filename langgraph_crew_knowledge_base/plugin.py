#!/usr/bin/env python3
"""Knowledge base plugin entry point (LangGraph Crew plugin contract).

Registers:
  * Provider ``knowledge_base`` (kind ``knowledge_base``)
  * Provider ``knowledge_discovery_node`` (executor loop)
  * Provider ``flow`` (node for the workflow's handoff step)
  * Tools for search, read, chunk, assignment
  * ``pre_model`` hook: mid-run injection after N tool calls
  * ``run_start`` hook: resets the injection counters

The core consumes the provider through the seam ``app/core/knowledge_seam.py``
(never by importing this package). Without this plugin the core runs clean:
knowledge base tools/injection are simply absent (Graceful Degradation).

All injection tuning comes from the environment — see
:mod:`langgraph_crew_knowledge_base.settings`.
"""

from app.core.plugin_api import Plugin, PluginContext

from langgraph_crew_knowledge_base.injection import (
    knowledge_mid_run_injection,
    reset_run_counters,
)
from langgraph_crew_knowledge_base.provider import KnowledgeBaseProvider
from langgraph_crew_knowledge_base.settings import load_settings


class FlowProvider:
    """Node provider for the workflow's handoff step."""

    name = "flow"

    def node(self):
        from langgraph_crew_knowledge_base.flow import knowledge_discovery_handoff

        return knowledge_discovery_handoff


class KnowledgeBasePlugin(Plugin):
    name = "knowledge-base"
    version = "0.1.0"
    description = (
        "Knowledge base with BGE-M3 embeddings, hierarchical chunking, "
        "reasoning-driven injection, single-invitation rendering, "
        "MCP tools, and knowledge discovery node "
        "(provider kind 'knowledge_base')."
    )

    def activate(self, ctx: PluginContext) -> None:
        provider = KnowledgeBaseProvider()

        # Register the main provider
        ctx.register_provider("knowledge_base", provider)

        # Register the node provider for the executor loop
        ctx.register_provider("knowledge_discovery_node", provider)

        # Register the flow provider for the workflow's handoff step
        ctx.register_provider("node", FlowProvider())

        # Register tools
        tools = provider.get_tools()
        if tools:
            ctx.register_tools(tools, category="knowledge_base", tool_type="plugin")
            ctx.logger.info(f"Registered {len(tools)} knowledge base tools")

        # ── Mid-run injection ────────────────────────────────────────────
        # Runs before every LLM call inside a ReAct loop and injects a
        # knowledge block every KNOWLEDGE_INJECT_AFTER_TOOLS tool calls.
        settings = load_settings()
        if not settings.disabled:
            ctx.register_hook(
                "pre_model",
                knowledge_mid_run_injection,
                priority=120,
                name="knowledge_mid_run_injection",
            )
            ctx.logger.info(
                "Knowledge injection: every %s tool calls, %s chunks "
                "(task start: %s chunks, max %s per run)",
                settings.inject_after_tools,
                settings.inject_chunks,
                settings.task_chunks,
                settings.max_per_run,
            )

            # Per-run counters start clean on every run.
            def _on_run_start(state) -> None:
                reset_run_counters(state)

            ctx.register_hook(
                "run_start",
                _on_run_start,
                priority=120,
                name="knowledge_reset_run_counters",
            )


plugin = KnowledgeBasePlugin()
