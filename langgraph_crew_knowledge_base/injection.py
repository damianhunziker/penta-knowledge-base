#!/usr/bin/env python3
"""Mid-run knowledge injection via the ``pre_model`` hook.

Contract
--------

The hook runs before every LLM call inside a ReAct loop (i.e. after each
completed tool turn). Every ``KNOWLEDGE_INJECT_AFTER_TOOLS`` tool calls it
appends a knowledge block to ``llm_input_messages`` — **never** to
``messages``: the block belongs in the prompt of this turn, not in the
persisted history, so it does not repeat in later turns.

Shape of the injected messages
------------------------------

Blocks are appended as a ``HumanMessage`` flagged with
``additional_kwargs["runtime_reminder"]`` (see
``app.core.react_agent_wrapper.runtime_reminder``). ``merge_system_messages()``
never hoists such a message to position 0, so the prompt prefix stays stable
and provider caching keeps working.

Counters
--------

The tool-call baseline and the "already injected" reference list live in the
plugin-owned state keys. They are reset at the start of every run by
:func:`reset_run_counters`, which the plugin calls from ``run_start``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from langgraph_crew_knowledge_base.runner import (
    MID_RUN_BLOCK_HEADER,
    build_mid_run_block,
)
from langgraph_crew_knowledge_base.settings import load_settings

logger = logging.getLogger(__name__)

#: ``additional_kwargs`` label so the frame is recognisable (and tail-safe).
REMINDER_KIND = "knowledge-base"

#: State keys owned by this plugin (declared as channels by the core).
KEY_BASELINE = "knowledge_inject_baseline"
KEY_FIRED = "knowledge_inject_fired"
KEY_REFS = "knowledge_injected_refs"


def _runtime_reminder(text: str) -> HumanMessage:
    """Build a tail-safe runtime message.

    Falls back to a plain ``HumanMessage`` carrying the same marker when the
    core helper is unavailable, so the contract holds in a bare install.
    """
    try:
        from app.core.react_agent_wrapper import runtime_reminder

        return runtime_reminder(text, REMINDER_KIND)
    except Exception:  # noqa: BLE001 - degrade, do not fail
        return HumanMessage(
            content=text, additional_kwargs={"runtime_reminder": REMINDER_KIND}
        )


def _tool_call_count(messages: List[Any]) -> int:
    """Number of AI messages that requested tools (= completed turns)."""
    return sum(1 for m in messages if getattr(m, "tool_calls", None))


def _current_task(state: Dict[str, Any]) -> Dict[str, Any]:
    """The task the agent is working on (``{}`` when unknown)."""
    current = state.get("current_task")
    for task in state.get("task_queue") or []:
        if not isinstance(task, dict):
            continue
        if task.get("description") == current or task.get("id") == current:
            return task
    return {}


def _query_for(state: Dict[str, Any], messages: List[Any]) -> str:
    """What the agent is currently doing — task text plus recent tool output.

    The task description alone would return the same chunks every time; the
    latest tool call/result is what the agent is actually looking at now.
    """
    parts: List[str] = []
    task = _current_task(state)
    for key in ("description", "title"):
        value = task.get(key)
        if value:
            parts.append(str(value))
    if not parts and state.get("current_task"):
        parts.append(str(state["current_task"]))

    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    parts.append(str(name))
            break

    return " ".join(parts).strip()


def reset_run_counters(state: Dict[str, Any]) -> None:
    """Clear the per-run counters (called from the ``run_start`` hook)."""
    state[KEY_BASELINE] = None
    state[KEY_FIRED] = 0
    state[KEY_REFS] = []


def knowledge_mid_run_injection(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """``pre_model`` hook: inject a knowledge block after N tool calls.

    Returns ``None`` when nothing is injected, so the hook chain simply
    continues with the previous result.
    """
    settings = load_settings()
    if not settings.mid_run_enabled:
        logger.debug("[KNOWLEDGE_INJECTION] Mid-run disabled")
        return None

    messages = state.get("messages") or []
    if not messages:
        # Nothing to count yet, but a run that starts with an empty history
        # must still arm the counter on its first real turn.
        if state.get(KEY_BASELINE) is None:
            state[KEY_BASELINE] = 0
            state[KEY_FIRED] = 0
            logger.debug("[KNOWLEDGE_INJECTION] First call, empty history - baseline=0")
        return None

    current = _tool_call_count(messages)
    baseline = state.get(KEY_BASELINE)
    logger.debug("[KNOWLEDGE_INJECTION] Tool calls: current=%s, baseline=%s, delta=%s, threshold=%s",
                current, baseline, (current - (baseline or 0)) if baseline else "N/A",
                settings.inject_after_tools)
    if baseline is None:
        # First entry of the run: start counting from here.
        #
        # ``run_start`` has no consumer in the core yet, so the reset hook is
        # best-effort. This branch is what actually guarantees a clean start:
        # a new agent run begins with no baseline in its state, so the first
        # LLM call arms the counter and the injection fires N turns later.
        state[KEY_BASELINE] = current
        state[KEY_FIRED] = 0
        return None

    try:
        baseline = int(baseline)
    except (TypeError, ValueError):
        state[KEY_BASELINE] = current
        return None

    delta = current - baseline
    if delta < settings.inject_after_tools:
        return None

    fired = int(state.get(KEY_FIRED) or 0)
    if fired >= settings.max_per_run:
        return None

    block = build_mid_run_block(
        _query_for(state, messages),
        state=state,
        settings=settings,
    )
    if not block:
        # Nothing to inject: keep the baseline so we retry after the next turn
        # instead of hammering the corpus on every single LLM call.
        state[KEY_BASELINE] = current - settings.inject_after_tools + 1
        return None

    # Count the threshold from the CURRENT turn, so the next injection comes
    # after another full window of tool calls.
    state[KEY_BASELINE] = current
    state[KEY_FIRED] = fired + 1

    refs = list(state.get(KEY_REFS) or [])
    for line in block.splitlines():
        if "kb_ref=\"" in line:
            try:
                refs.append(line.split('kb_ref="', 1)[1].split('"', 1)[0])
            except IndexError:
                continue
    state[KEY_REFS] = refs

    try:
        from app.core.cli_styling import display_knowledge_injection

        _chunk_count = sum(
            1 for line in block.splitlines() if line.startswith("↳ read section")
        )
        # Debug: always print to verify the hook is running
        logger.info("[KNOWLEDGE_INJECTION] Mid-run block built: %s chunks, %s chars",
                   _chunk_count, len(block))
        display_knowledge_injection(
            kind="mid-run",
            chunks=_chunk_count,
            tokens=(len(block) + 3) // 4,
            body=block,
            refs=refs,
        )
    except Exception as e:  # noqa: BLE001 - display only
        logger.warning("[KNOWLEDGE_INJECTION] display_knowledge_injection failed: %s", e)
        pass

    return {"llm_input_messages": list(messages) + [_runtime_reminder(block)]}


__all__ = [
    "KEY_BASELINE",
    "KEY_FIRED",
    "KEY_REFS",
    "REMINDER_KIND",
    "knowledge_mid_run_injection",
    "reset_run_counters",
]
