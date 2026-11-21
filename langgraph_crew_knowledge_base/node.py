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


def _manifest_knowledge_config(agent_id: str) -> Dict[str, Any]:
    """``x-penta.knowledge_base`` of the agent's manifest (``{}`` when absent).

    Resolution follows the description-language ladder (project → client →
    plugin → custom → core), so a deployment can override the corpus binding of
    a plugin agent by dropping ``agents/<name>.md`` into ``custom/`` or the
    client/project folder — no code change needed.
    """
    try:
        from app.core.spec.loader import load_agent_manifest

        # Manifests are named without the "_agent" suffix by convention; try
        # both spellings so either works.
        for name in (agent_id, agent_id.removesuffix("_agent")):
            manifest = load_agent_manifest(name)
            if manifest is None:
                continue
            config = getattr(getattr(manifest, "x_penta", None), "knowledge_base", None)
            if config is None:
                continue
            return {
                "titles": list(getattr(config, "titles", []) or []),
                "categories": list(getattr(config, "categories", []) or []),
                "tags": list(getattr(config, "tags", []) or []),
                "chunks": getattr(config, "chunks", None),
                "enabled": bool(getattr(config, "enabled", True)),
            }
    except Exception:  # noqa: BLE001 - manifest optional
        logger.debug("[KNOWLEDGE_DISCOVERY] manifest knowledge config failed", exc_info=True)
    return {}


def _assigned_titles_for(agent_id: str, state: Dict[str, Any]) -> List[str]:
    """Titles assigned to ``agent_id`` (persisted, survives a restart).

    An assignment is the standing "this agent always has these titles in
    scope" — set once with ``assign_to_agent``, read on every task the agent
    runs. The planner can still pin additional titles per task; both lists are
    merged by the caller.
    """
    if not agent_id:
        return []
    try:
        from langgraph_crew_knowledge_base.assignments import books_for_agent

        # Ladder-aware: deployment ∪ this customer ∪ this project. The active
        # scope comes from access_scope, so a run in another project does not
        # inherit a project-level grant.
        return books_for_agent(agent_id)
    except Exception:  # noqa: BLE001 - assignment is optional
        logger.debug("[KNOWLEDGE_DISCOVERY] assignment lookup failed", exc_info=True)
        return []


def _knowledge_base_disabled() -> bool:
    """True when knowledge discovery is switched off by the environment."""
    return os.getenv("DISABLE_KNOWLEDGE_DISCOVERY", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _knowledge_discovery_enabled_by_default() -> bool:
    """True when knowledge discovery should run for every task unless explicitly disabled."""
    try:
        from langgraph_crew_knowledge_base.settings import load_settings

        return load_settings().enabled_by_default
    except Exception:  # noqa: BLE001
        return os.getenv("ENABLE_KNOWLEDGE_DISCOVERY_DEFAULT", "false").lower() in (
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
        
        # Check if enabled by default (new: can be turned on via env for every task)
        if not knowledge_discovery_enabled and _knowledge_discovery_enabled_by_default():
            knowledge_discovery_enabled = True
            state["knowledge_discovery_enabled"] = True
            if self.verbose:
                print("📚 Knowledge discovery enabled by default (ENABLE_KNOWLEDGE_DISCOVERY_DEFAULT=true)")
        
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
        # that runs the task, so its titles are always in scope. The manifest
        # (x-penta.knowledge_base) declares the same thing in description
        # language; both are merged, the manifest first because it is the
        # operator's explicit statement about THIS agent.
        if not task_role:
            task_role = str(state.get("active_agent") or state.get("current_agent") or "")
        if task_role:
            manifest_config = _manifest_knowledge_config(task_role)
            if manifest_config.get("enabled") is False:
                if self.verbose:
                    print(f"📚 [KNOWLEDGE] disabled by manifest for {task_role}")
                return Command(
                    goto="executor",
                    update={
                        "knowledge_discovery_findings": {
                            "step": "knowledge_discovery",
                            "task": current_task,
                            "assigned_titles": [],
                            "assigned_categories": [],
                            "recommendations": [],
                            "passages": [],
                            "chunking_jobs": [],
                            "corpus": {"ready": True, "reason": "disabled by manifest",
                                       "chunks": 0},
                        },
                        "messages": state.get("messages", []),
                    },
                )
            for title in manifest_config.get("titles") or []:
                if title not in assigned_titles:
                    assigned_titles.append(title)
            for category in manifest_config.get("categories") or []:
                if category not in assigned_categories:
                    assigned_categories.append(category)
            for title in _assigned_titles_for(task_role, state):
                if title not in assigned_titles:
                    assigned_titles.append(title)

        # ── The discovery run ────────────────────────────────────────────
        # One agent does both halves in a single pass: what the corpus already
        # has, and what exists on the internet, then installs and assigns what
        # is missing (see discovery.py and agents/knowledge_scout.md). The old
        # code searched the corpus itself and stopped there — three capable
        # agent manifests were declared and never executed.
        recommendations: List[Dict[str, Any]] = []
        passages: List[Dict[str, Any]] = []
        report: Dict[str, Any] = {}
        if not _knowledge_base_disabled():
            try:
                from langgraph_crew_knowledge_base.discovery import (
                    report_as_findings,
                    run_discovery,
                )

                report = await run_discovery(
                    task_description,
                    dict(state),
                    agent=self,
                )
                derived = report_as_findings(report)
                for title in derived.get("assigned_titles") or []:
                    if title not in assigned_titles:
                        assigned_titles.append(title)
                recommendations = list(derived.get("recommendations") or [])
                passages = list(derived.get("passages") or [])
            except Exception as exc:  # noqa: BLE001 - discovery never breaks a run
                logger.warning("[KNOWLEDGE_DISCOVERY] discovery run failed: %s", exc)
                report = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "answer_for_worker": (
                        "Knowledge discovery failed before it could answer: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

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
            # The complete two-round answer, and the scout's own summary. The
            # executor renders these for the worker; keeping the report whole
            # means the worker sees what was searched and what was rejected,
            # not just the surviving titles.
            "report": report,
            "answer_for_worker": report.get("answer_for_worker", ""),
            "gap": (report.get("round1_existing") or {}).get("gap", ""),
            "needs_decision": (report.get("round2_web") or {}).get("needs_decision", []),
            "error": report.get("error", ""),
        }

        if not status.get("ready"):
            # A missing corpus explains everything else: the scout had nothing
            # to search, so its failure (if any) is a symptom, not the cause.
            summary = (
                "Knowledge discovery found no corpus to search: "
                f"{status.get('reason') or 'not indexed'}. "
                "Continuing without additional knowledge. "
                "Ingest documents with knowledge_base___ingest_from_root."
            )
            if report.get("error"):
                summary += f" (the scout run also reported: {report['error']})"
        elif report.get("error"):
            # The run itself failed — say that, not "completed with 0".
            summary = (
                f"Knowledge discovery did not complete: {report['error']}. "
                "The task continues without corpus discovery."
            )
        else:
            # status.ready is true here: a missing corpus and a failed run were
            # handled above, so this is the completed case.
            found = len(((report.get("round1_existing") or {}).get("found")) or [])
            candidates = len(((report.get("round2_web") or {}).get("candidates")) or [])
            assigned = len(((report.get("round3_installed") or {}).get("assigned")) or [])
            summary = (
                "Knowledge discovery completed. "
                f"{found} corpus title(s) found, {candidates} web candidate(s), "
                f"{assigned} assignment(s), {len(assigned_titles)} assigned title(s)."
            )
            if not found and not candidates:
                summary += (
                    " Neither the corpus nor the web research produced a usable "
                    "source for this task — the books do not cover it."
                )

        # Always visible: the node returning to the executor without a line was
        # indistinguishable from the routing being ignored.
        try:
            from app.core.cli_styling import console

            searched = status.get("chunks") or 0
            found = len(((report.get("round1_existing") or {}).get("found")) or [])
            candidates = len(((report.get("round2_web") or {}).get("candidates")) or [])
            assigned_now = len(
                ((report.get("round3_installed") or {}).get("assigned")) or []
            )
            if report.get("error"):
                detail = f"failed · {report['error'][:90]}"
            else:
                detail = (
                    f"{found} corpus title(s) · {candidates} web candidate(s) "
                    f"· {assigned_now} assignment(s) · {searched} chunk(s) searched"
                )
            console.print(
                f"[#9C27B0]📚 [KNOWLEDGE DISCOVERY][/#9C27B0] [dim]{detail}[/dim]"
            )
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