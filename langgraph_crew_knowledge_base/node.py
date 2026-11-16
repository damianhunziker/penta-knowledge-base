#!/usr/bin/env python3
"""
Knowledge Discovery Node - NodeAgent Implementation.

This node discovers relevant knowledge for task execution:
1. Searches existing corpus for relevant titles
2. Researches new PDFs with correct licenses
3. Recommends titles to the worker
4. Optionally triggers background chunking
5. Assigns titles to agents

Similar to tool_discovery_node, it runs:
- Pre-execution: Before worker agents start
- Runtime: When agent recognizes need for reference material
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from app.core.core_node import NodeAgent
from app.core.enhanced_state import EnhancedCrewState

logger = logging.getLogger(__name__)


def _assigned_titles_for(agent_id: str, state: Dict[str, Any]) -> List[str]:
    """Titles assigned to ``agent_id`` (planner task field or agent assignment).

    Two sources, both optional:
      * ``assigned_titles`` the planner put on the task (per run), and
      * a standing assignment in the knowledge base (``assign_to_agent``),
        which is how an operator pins a corpus to an agent once.
    """
    titles: List[str] = []
    try:
        from langgraph_crew_knowledge_base.tools import (
            knowledge_base___list_assignments,
        )

        raw = knowledge_base___list_assignments(agent_id)
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        for entry in payload.get("assignments") or []:
            if isinstance(entry, str) and entry.startswith("book:"):
                book = entry[len("book:"):]
                if book not in titles:
                    titles.append(book)
    except Exception:  # noqa: BLE001 - assignment is optional
        logger.debug("[KNOWLEDGE_DISCOVERY] assignment lookup failed", exc_info=True)
    return titles


def _knowledge_base_disabled() -> bool:
    """True when knowledge discovery is switched off by the environment."""
    return os.getenv("DISABLE_KNOWLEDGE_DISCOVERY", "false").lower() in (
        "true",
        "1",
        "yes",
    )


class KnowledgeDiscoveryNode(NodeAgent):
    """
    Knowledge Discovery Node - discovers relevant knowledge.
    
    This node:
    1. Analyzes the current task
    2. Determines if knowledge discovery is needed
    3. Searches existing corpus
    4. Researches new titles (with license check)
    5. Triggers chunking if approved
    6. Assigns titles to agents
    7. Returns findings for the worker
    
    Inherits from NodeAgent for:
    - ReactAgentWrapper with tool calling
    - Streaming support
    - Checkpoint/resume integration
    """
    
    # Node-specific color and icon
    color = "#9C27B0"
    icon = "📚 KNOWLEDGE DISCOVERY"
    llm_role = "knowledge_researcher"
    
    def __init__(self, verbose: bool = False):
        """Initialize KnowledgeDiscoveryNode.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(node_name="knowledge_discovery", verbose=verbose)
    
    async def execute(self, state: EnhancedCrewState) -> Command:
        """Execute knowledge discovery for the current task.
        
        This node:
        1. Checks if knowledge discovery is enabled in state
        2. Analyzes what knowledge is needed
        3. Runs discovery workflow
        4. Returns to executor
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Next node to go to (executor or worker)
        """
        if self.verbose:
            print("📚 [KNOWLEDGE_DISCOVERY_NODE] start")

        # Make the node visible in the transcript. Without this the run jumps
        # from "ROUTING KNOWLEDGE_DISCOVERY" straight to the worker and looks
        # like the routing was ignored (tool_discovery prints the same box).
        try:
            from app.core.cli_styling import display_node_execution

            display_node_execution("knowledge_discovery")
        except Exception:  # noqa: BLE001 - display only
            pass
        
        # ====================================================================
        # DISABLE_KNOWLEDGE_DISCOVERY: Skip knowledge discovery entirely
        # ====================================================================
        disable_knowledge_discovery = os.getenv(
            "DISABLE_KNOWLEDGE_DISCOVERY", "false"
        ).lower() in ("true", "1", "yes")
        
        if disable_knowledge_discovery:
            if self.verbose:
                print("📚 [KNOWLEDGE_DISCOVERY_NODE] skipped (DISABLE_KNOWLEDGE_DISCOVERY=true)")
            current_task = self._get_current_task(state)
            completed_for_tasks = set(state.get("knowledge_discovery_completed_for_tasks", []))
            completed_for_tasks.add(current_task)
            return Command(
                goto="executor",
                update={
                    "messages": state.get("messages", []),
                    "knowledge_discovery_completed_for_tasks": list(completed_for_tasks),
                    "current_task": None,
                }
            )
        
        # Check if knowledge discovery is enabled
        knowledge_discovery_enabled = state.get("knowledge_discovery_enabled", False)
        
        # Get current task for completion tracking
        current_task = self._get_current_task(state)
        
        # Check if already completed for this task
        kd_completed = state.get("knowledge_discovery_completed_for_tasks", [])
        completed_set = set(kd_completed) if kd_completed else set()
        
        if current_task in completed_set:
            if self.verbose:
                print("📚 [KNOWLEDGE_DISCOVERY_NODE] skipped (already completed for this task)")
            return Command(
                goto="executor",
                update={
                    "messages": state.get("messages", []),
                    "knowledge_discovery_completed_for_tasks": list(completed_set),
                    "current_task": None,
                }
            )
        
        # Check task definition for requires_knowledge_discovery
        if not knowledge_discovery_enabled:
            current_plan = state.get("current_plan", [])
            for task in current_plan:
                if task.get("requires_knowledge_discovery", False):
                    knowledge_discovery_enabled = True
                    state["knowledge_discovery_enabled"] = True
                    if self.verbose:
                        print(f"📚 Knowledge discovery enabled from current_plan: {task.get('id')}")
                    break
            
            if not knowledge_discovery_enabled:
                task_queue = state.get("task_queue", [])
                for task in task_queue:
                    if task.get("requires_knowledge_discovery", False):
                        knowledge_discovery_enabled = True
                        state["knowledge_discovery_enabled"] = True
                        if self.verbose:
                            print(f"📚 Knowledge discovery enabled from task_queue: {task.get('id')}")
                        break
        
        if not knowledge_discovery_enabled:
            if self.verbose:
                print("📚 [KNOWLEDGE_DISCOVERY_NODE] skipped (not enabled for this task)")
            
            completed_for_tasks = set(state.get("knowledge_discovery_completed_for_tasks", []))
            completed_for_tasks.add(current_task)
            
            return Command(
                goto="executor",
                update={
                    "messages": state.get("messages", []) + [
                        AIMessage(content="Knowledge discovery skipped - not enabled for this task")
                    ],
                    "knowledge_discovery_completed_for_tasks": list(completed_for_tasks),
                    "current_task": None,
                }
            )
        
        if self.verbose:
            print("📚 [KNOWLEDGE_DISCOVERY_NODE] running discovery")        
        # Run the discovery workflow
        try:
            command = await self._run_discovery_workflow(state)
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.exception("[KNOWLEDGE_DISCOVERY] knowledge discovery failed")
            
            completed = set(state.get("knowledge_discovery_completed_for_tasks", []) or [])
            completed.add(current_task)
            
            return Command(
                goto="executor",
                update={
                    "messages": state.get("messages", []) + [
                        AIMessage(
                            content=f"Knowledge discovery failed: {type(e).__name__}: {e}. "
                                   "Continuing without additional knowledge."
                        )
                    ],
                    "knowledge_discovery_completed_for_tasks": list(completed),
                    "current_task": None,
                },
            )
        
        updates = dict(command.update or {})
        updates.setdefault("current_task", None)
        
        # Mark as completed
        completed = set(state.get("knowledge_discovery_completed_for_tasks", []) or [])
        completed.add(current_task)
        updates["knowledge_discovery_completed_for_tasks"] = list(completed)
        
        return Command(goto=command.goto or "executor", update=updates)
    
    async def _run_discovery_workflow(self, state: EnhancedCrewState) -> Command:
        """Run the knowledge discovery workflow.

        Produces the finding block the executor injects into the worker prompt
        (:func:`build_knowledge_discovery_report_message`): assigned titles,
        categories, corpus recommendations and the passages read for the task.
        """
        current_task = self._get_current_task(state)

        # Titles/categories the planner already pinned on the task
        assigned_titles: List[str] = []
        assigned_categories: List[str] = []
        task_description = current_task

        task_queue = state.get("task_queue", [])
        current_plan = state.get("current_plan", []) or []
        task_role = ""
        for task in list(task_queue) + list(current_plan):
            if not isinstance(task, dict):
                continue
            if task.get("id") == current_task or task.get("description") == current_task:
                assigned_titles = list(task.get("assigned_titles") or [])
                assigned_categories = list(task.get("assigned_categories") or [])
                task_description = task.get("description") or current_task
                task_role = str(task.get("role") or task.get("assigned_to") or "")
                break

        # A standing assignment (assign_to_agent) pins a corpus to the agent
        # that runs the task, so its titles are always in scope.
        if not task_role:
            task_role = str(state.get("active_agent") or state.get("current_agent") or "")
        if task_role:
            for title in _assigned_titles_for(task_role, state):
                if title not in assigned_titles:
                    assigned_titles.append(title)

        # Search the corpus for titles the planner did not name.
        recommendations: List[Dict[str, Any]] = []
        passages: List[Dict[str, Any]] = []
        if not _knowledge_base_disabled():
            try:
                from langgraph_crew_knowledge_base.settings import (
                    chunks_for_task,
                    load_settings,
                )

                settings = load_settings()
                # How many chunks this task gets at start: the task's own
                # ``knowledge_chunks`` when set, else KNOWLEDGE_TASK_CHUNKS.
                limit = chunks_for_task(
                    next(
                        (
                            t
                            for t in list(task_queue) + list(current_plan)
                            if isinstance(t, dict)
                            and (
                                t.get("id") == current_task
                                or t.get("description") == current_task
                            )
                        ),
                        {},
                    ),
                    settings,
                )

                if limit > 0:
                    from langgraph_crew_knowledge_base.tools import (
                        knowledge_base___read,
                        knowledge_base___search,
                    )

                    import json as _json

                    raw = knowledge_base___search(
                        query=task_description, corpora=None, tier=None, top_k=limit
                    )
                    result = _json.loads(raw) if isinstance(raw, str) else (raw or {})
                    for hit in (result.get("results") or [])[:limit]:
                        if not isinstance(hit, dict):
                            continue
                        ref = hit.get("kb_ref") or ""
                        book = hit.get("book_id") or ""
                        if ref:
                            passages.append(
                                {
                                    "kb_ref": ref,
                                    "text": hit.get("text") or "",
                                    "score": hit.get("score"),
                                    "title": hit.get("title") or book,
                                }
                            )
                        elif book and book not in assigned_titles:
                            recommendations.append(
                                {"title": book, "reason": "corpus match"}
                            )
            except Exception:  # noqa: BLE001 - discovery must never break the run
                logger.debug("[KNOWLEDGE_DISCOVERY] corpus search failed", exc_info=True)

        # An empty corpus is a real answer, not a silent no-op: say why nothing
        # was found, so the transcript does not look like a skipped step.
        status: Dict[str, Any] = {"ready": True, "reason": "", "chunks": 0}
        try:
            from langgraph_crew_knowledge_base.tools import kb_status

            status = kb_status()
        except Exception:  # noqa: BLE001
            pass

        findings = {
            "step": "knowledge_discovery",
            "task": current_task,
            "assigned_titles": assigned_titles,
            "assigned_categories": assigned_categories,
            "recommendations": recommendations,
            "passages": passages,
            "chunking_jobs": list(state.get("knowledge_discovery_chunking_jobs") or []),
            "corpus": status,
        }

        if status.get("ready"):
            summary = (
                "Knowledge discovery completed. "
                f"{len(assigned_titles)} assigned title(s), "
                f"{len(passages)} passage(s)."
            )
        else:
            summary = (
                "Knowledge discovery found no corpus to search: "
                f"{status.get('reason') or 'not indexed'}. "
                "Continuing without additional knowledge. "
                "Ingest documents with knowledge_base___ingest_from_root."
            )
            try:
                from app.core.cli_styling import console

                console.print(f"[dim]📚 [KNOWLEDGE] {summary}[/dim]")
            except Exception:  # noqa: BLE001 - display only
                pass

        return Command(
            goto="executor",
            update={
                "knowledge_discovery_findings": findings,
                "messages": state.get("messages", []) + [
                    AIMessage(content=summary)
                ],
            }
        )
    
    def _get_current_task(self, state: EnhancedCrewState) -> str:
        """Extract current task from state."""
        current_task_from_state = state.get("current_task")
        
        if current_task_from_state:
            return current_task_from_state
        
        task_queue = state.get("task_queue", [])
        if task_queue:
            for task in task_queue:
                if task.get("requires_knowledge_discovery", False):
                    return task.get("description", task.get("id", "Unknown task"))
            return task_queue[0].get("description", "Unknown task")
        
        current_plan = state.get("current_plan", [])
        if current_plan:
            return current_plan[0].get("description", "Unknown task")
        
        return "General task"
    
    async def _execute_node_logic(self, state: EnhancedCrewState) -> Command:
        """Execute knowledge discovery logic (called by base execute())."""
        return await self.execute(state)


# Export for node factory
NODE_CLASS = KnowledgeDiscoveryNode


# Cache for node instances
_node_cache = {}


async def knowledge_discovery_node(state: EnhancedCrewState) -> Command:
    """Backward compatibility wrapper for knowledge_discovery_node function.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command: Next node to go to
    """
    global _node_cache
    node_name = "knowledge_discovery"
    
    if node_name not in _node_cache:
        from app.core.global_state import get_verbose
        verbose = get_verbose()
        _node_cache[node_name] = KnowledgeDiscoveryNode(verbose=verbose)
        success = await _node_cache[node_name].initialize()
        if not success:
            del _node_cache[node_name]
            raise RuntimeError(f"Failed to initialize {node_name} node")
    
    return await _node_cache[node_name].execute(state)