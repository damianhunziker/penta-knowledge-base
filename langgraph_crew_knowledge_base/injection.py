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
#: Index into ``messages`` where the previous profile ended. Everything
#: after it is the interval being profiled for the next search.
KEY_PROFILED_AT = "knowledge_profiled_at"


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
    """What the agent is working on *right now*, as a search query.

    The old version used the task description plus one tool name, so every
    mid-run injection searched with the same string and returned the same
    chunks — the agent could not tell the corpus had nothing new to say.

    Now the interval since the last check is profiled
    (:func:`app.core.memory_profile.profile_messages`): the agent's reasoning,
    the tools it called with their arguments, and the salient entities. That is
    what makes injection follow the work instead of the task opening line.
    """
    task = _current_task(state)
    goal = str(
        task.get("description")
        or task.get("title")
        or state.get("current_task")
        or ""
    ).strip()

    # Where the previous check ended: everything after it is "this interval".
    start = 0
    try:
        start = int(state.get(KEY_PROFILED_AT) or 0)
    except (TypeError, ValueError):
        start = 0
    if start < 0 or start > len(messages):
        start = 0

    try:
        from app.core.memory_profile import profile_messages, profile_with_llm

        profile = profile_with_llm(messages, goal) or profile_messages(
            messages, start=start, goal=goal
        )
        query = profile.query()
        if query:
            return query
    except Exception:  # noqa: BLE001 - profiling is optional
        logger.debug("[KNOWLEDGE_INJECTION] profiling failed", exc_info=True)

    # Fallback: the goal, so an unprofiled run still searches something sane.
    return goal


def reset_run_counters(state: Dict[str, Any]) -> None:
    """Clear the per-run counters (called from the ``run_start`` hook)."""
    state[KEY_BASELINE] = None
    state[KEY_FIRED] = 0
    state[KEY_REFS] = []
    state[KEY_PROFILED_AT] = 0


def knowledge_mid_run_injection(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """``pre_model`` hook: inject a knowledge block after N tool calls.

    Returns a state delta — never ``None`` once anything changed. The hook
    contract is RETURN-based: ``state`` is a merged COPY handed to the hook
    chain, so mutating ``state[...]`` alone is discarded and the counters would
    read back as ``None`` on the next model call (the injection then either
    never fires, or fires on every single turn). Non-message keys returned here
    are persisted by ``PreModelHooksMiddleware`` through a ``Command``.
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
            logger.debug("[KNOWLEDGE_INJECTION] First call, empty history - baseline=0")
            return {KEY_BASELINE: 0, KEY_FIRED: 0}
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
        return {KEY_BASELINE: current, KEY_FIRED: 0}

    try:
        baseline = int(baseline)
    except (TypeError, ValueError):
        return {KEY_BASELINE: current}

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
        # Nothing to inject: move the baseline so we retry after the next turn
        # instead of hammering the corpus on every single LLM call.
        return {KEY_BASELINE: current - settings.inject_after_tools + 1}

    # Count the threshold from the CURRENT turn, so the next injection comes
    # after another full window of tool calls.
    # Count the threshold from the CURRENT turn, so the next injection comes
    # after another full window of tool calls.
    refs = list(state.get(KEY_REFS) or [])
    for line in block.splitlines():
        if "kb_ref=\"" in line:
            try:
                refs.append(line.split('kb_ref="', 1)[1].split('"', 1)[0])
            except IndexError:
                continue

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

    return {
        "llm_input_messages": list(messages) + [_runtime_reminder(block)],
        # Counters MUST travel with the delta: the hook receives a copy of the
        # state, so a bare assignment would be thrown away.
        KEY_BASELINE: current,
        KEY_FIRED: fired + 1,
        KEY_REFS: refs,
        # The interval is consumed: the next profile starts here.
        KEY_PROFILED_AT: len(messages),
    }


__all__ = [
    "KEY_BASELINE",
    "KEY_FIRED",
    "KEY_PROFILED_AT",
    "KEY_REFS",
    "REMINDER_KIND",
    "knowledge_mid_run_injection",
    "reset_run_counters",
]
