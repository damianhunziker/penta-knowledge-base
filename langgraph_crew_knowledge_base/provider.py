#!/usr/bin/env python3
"""Provider contract of the knowledge_base plugin (kind ``knowledge_base``).

The core never imports this package directly. It resolves the active provider
via ``PluginHost.providers("knowledge_base")`` (see ``app/core/knowledge_seam.py``)
and calls the methods below. Without an active plugin the provider is absent and
all knowledge base features degrade gracefully.

Contract (v1):

    is_available()                     -> bool
    get_catalog(corpus)                -> dict
    search(query, corpora, tier, ...)  -> list[Chunk]
    read(kb_ref, budget_tokens, ...)   -> str (rendered)
    render_chunk(chunk)                -> str (single invitation)
    derive_broader_refs(kb_ref)        -> dict
    node()                             -> knowledge_discovery_node callable
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeBaseProvider:
    """Provider exposed by the knowledge_base plugin for the core seams."""

    name = "knowledge_base"

    # -- availability -----------------------------------------------------

    def is_available(self) -> bool:
        """Check if the knowledge base is available (Qdrant reachable)."""
        # TODO: Check Qdrant connection
        return True

    # -- catalog ----------------------------------------------------------

    def get_catalog(self, corpus: str | None = None) -> Dict[str, Any]:
        """Return the catalog: available corpora, books, chunk counts."""
        from langgraph_crew_knowledge_base.tools import knowledge_base___get_catalog
        import json
        return json.loads(knowledge_base___get_catalog())

    # -- retrieval --------------------------------------------------------

    def search(
        self,
        query: str,
        corpora: List[str] | None = None,
        tier: str | None = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Any]:
        """Search the knowledge base. Returns list of result dicts."""
        from langgraph_crew_knowledge_base.tools import knowledge_base___search
        import json
        result = json.loads(knowledge_base___search(query, corpora, tier, top_k))
        return result.get("results", [])

    def read(
        self,
        kb_ref: str,
        budget_tokens: int | None = None,
        before: int = 0,
        after: int = 0,
        **kwargs,
    ) -> str:
        """Read a section/chapter/book."""
        from langgraph_crew_knowledge_base.tools import knowledge_base___read
        return knowledge_base___read(kb_ref, budget_tokens, before, after)

    # -- rendering --------------------------------------------------------

    def render_chunk(self, chunk: Any) -> str:
        """Render a chunk with ONE invitation line."""
        from langgraph_crew_knowledge_base.render import render_chunk
        return render_chunk(chunk)

    def derive_broader_refs(self, kb_ref: str) -> Dict[str, str | None]:
        """Derive chapter/book/neighbours from a section ref."""
        from langgraph_crew_knowledge_base.render import derive_broader_refs
        return derive_broader_refs(kb_ref)

    # -- injection (for hooks) --------------------------------------------

    def get_injection_context(
        self,
        trace: Any = None,
        budget_tokens: int = 800,
        kind: str = "cold",
    ) -> str:
        """Get context to inject into the agent."""
        # TODO: Implement injection logic
        if kind == "cold":
            catalog = self.get_catalog()
            total_books = catalog.get("total_books", 0)
            total_chunks = catalog.get("total_chunks", 0)
            return f"Knowledge base available: {total_books} books, {total_chunks} chunks. Use knowledge_base___search."
        return ""

    # -- node provider ----------------------------------------------------

    def node(self) -> Callable[..., Any]:
        """Return the knowledge_discovery_node callable for the executor loop."""
        from langgraph_crew_knowledge_base.node import knowledge_discovery_node
        return knowledge_discovery_node

    # -- tools ------------------------------------------------------------

    def get_tools(self) -> List[Any]:
        """Return all knowledge base tools for registration."""
        from langgraph_crew_knowledge_base.tools import (
            knowledge_base___search,
            knowledge_base___read,
            knowledge_base___list,
            knowledge_base___describe,
            knowledge_base___recommend_titles,
            knowledge_base___chunk_document,
            knowledge_base___chunk_status,
            knowledge_base___assign_to_agent,
            knowledge_base___list_assignments,
            knowledge_base___grant_access,
            knowledge_base___access_matrix,
            knowledge_base___get_catalog,
        )
        from langgraph_crew_knowledge_base.ingest import (
            knowledge_base___discover_files,
            knowledge_base___ingest_from_root,
            knowledge_base___ingest_directory,
            knowledge_base___list_vault_knowledge,
        )
        from langchain_core.tools import tool
        
        # Wrap each function as a LangChain tool
        tools = []
        for func in [
            # Search and read
            knowledge_base___search,
            knowledge_base___read,
            knowledge_base___list,
            knowledge_base___describe,
            # Recommendations
            knowledge_base___recommend_titles,
            # Chunking
            knowledge_base___chunk_document,
            knowledge_base___chunk_status,
            # Assignment + access control
            knowledge_base___assign_to_agent,
            knowledge_base___list_assignments,
            knowledge_base___grant_access,
            knowledge_base___access_matrix,
            # Catalog
            knowledge_base___get_catalog,
            # File ingestion
            knowledge_base___discover_files,
            knowledge_base___ingest_from_root,
            knowledge_base___ingest_directory,
            knowledge_base___list_vault_knowledge,
        ]:
            # Use the function's docstring as the tool description
            tools.append(tool(func))
        
        return tools


__all__ = ["KnowledgeBaseProvider"]