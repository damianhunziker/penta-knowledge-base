#!/usr/bin/env python3
"""Knowledge base plugin entry point (LangGraph Crew plugin contract).

Registers:
  * Provider ``knowledge_base`` (kind ``knowledge_base``)
  * Tools for search, read, chunk, assignment
  * Node provider ``knowledge_discovery`` for the executor loop

The core consumes the provider through the seam ``app/core/knowledge_seam.py``
(never by importing this package). Without this plugin the core runs clean:
knowledge base tools/injection are simply absent (Graceful Degradation).
"""

from app.core.plugin_api import Plugin, PluginContext

from langgraph_crew_knowledge_base.provider import KnowledgeBaseProvider


class KnowledgeBasePlugin(Plugin):
    name = "knowledge_base"
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
        
        # Register tools
        tools = provider.get_tools()
        if tools:
            ctx.register_tools(tools, category="knowledge_base", tool_type="plugin")
            ctx.logger.info(f"Registered {len(tools)} knowledge base tools")


plugin = KnowledgeBasePlugin()