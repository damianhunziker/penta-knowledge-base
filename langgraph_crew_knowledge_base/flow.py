#!/usr/bin/env python3
"""
Flow module for knowledge discovery workflow.

Provides the handoff node that merges findings from the discovery agents
and returns them to the executor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


async def knowledge_discovery_handoff(state: Dict[str, Any]) -> Command:
    """
    Handoff node for knowledge discovery workflow.
    
    Knowledge discovery is one step (``knowledge_scout``), which does the corpus
    round, the web round and the install round in one pass. This handoff merges
    whatever findings reach it into one message for the worker; it no longer
    stitches together the three separate agents that used to exist.
    """
    findings_list = state.get("knowledge_discovery_findings", [])
    
    if not findings_list:
        findings_list = []
    
    # Merge findings
    merged = {
        "step": "handoff",
        "existing_titles": [],
        "new_candidates": [],
        "chunking_jobs": [],
        "assignments": [],
    }
    
    for findings in findings_list:
        if isinstance(findings, dict):
            step = findings.get("step", "")
            
            if step == "existing_corpus":
                merged["existing_titles"].extend(findings.get("titles", []))
            elif step == "new_titles":
                merged["new_candidates"].extend(findings.get("candidates", []))
            elif step == "curate":
                merged["chunking_jobs"].extend(findings.get("chunking_jobs", []))
                merged["assignments"].extend(findings.get("assignments", []))
    
    # Build the handoff message
    message_parts = ["📚 **Knowledge Discovery Handoff**\n"]
    
    if merged["existing_titles"]:
        message_parts.append(f"\n**Existing titles found:** {len(merged['existing_titles'])}")
        for title in merged["existing_titles"][:5]:
            message_parts.append(f"  - {title.get('title', 'Unknown')} ({title.get('kb_ref', '')})")
    
    if merged["new_candidates"]:
        message_parts.append(f"\n**New candidates:** {len(merged['new_candidates'])}")
        for candidate in merged["new_candidates"][:3]:
            message_parts.append(f"  - {candidate.get('title', 'Unknown')} (license: {candidate.get('license', 'unknown')})")
    
    if merged["chunking_jobs"]:
        message_parts.append(f"\n**Chunking jobs started:** {len(merged['chunking_jobs'])}")
    
    if merged["assignments"]:
        message_parts.append(f"\n**Titles assigned to worker:** {len(merged['assignments'])}")
    
    handoff_message = "\n".join(message_parts)
    
    return Command(
        goto="__end__",
        update={
            "messages": state.get("messages", []) + [AIMessage(content=handoff_message)],
            "knowledge_discovery_findings": [merged],
        }
    )


__all__ = ["knowledge_discovery_handoff"]