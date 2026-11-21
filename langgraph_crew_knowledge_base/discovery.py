#!/usr/bin/env python3
"""The knowledge discovery run: one agent, two rounds, one JSON report.

The node used to do pure Python: search the corpus, assemble a dict, done. Three
agent manifests (``knowledge_scout``, ``knowledge_researcher``,
``knowledge_curator``) were declared and **never executed** — ``llm_role`` was
set on the node and nothing ever called the model. The two later stages are now
removed: the scout absorbs them, so there is no second place to keep in step.

This module runs the scout once, exactly like tool discovery runs its
researcher: an LLM with tools, a system prompt that says what to do, and a JSON
answer that is parsed back. The scout does both halves itself — the corpus it
already has, and what exists on the internet — then installs and assigns what is
missing. Doing it in one pass is what makes the report coherent: the second
round is informed by the gap the first round found.

Everything degrades visibly: no provider, no answer, an unparseable answer, a
run that times out — each returns a report that *says so* instead of an empty
dict that reads like "nothing to report".
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Agent that performs the run (see ``agents/knowledge_scout.md``).
SCOUT_AGENT = "knowledge_scout"

#: How long a discovery run may take before it is cut off.
#:
#: A three-round run is genuinely long: several corpus searches, web research,
#: then download/extract/embed for each document it decides to install. Measured
#: on a real run it chunked two vault documents and assigned them — and 180 s cut
#: it off before it could write its report. The side effects had already
#: happened, so the result looked like a failure that changed nothing.
#:
#: 15 minutes is the default; ``KNOWLEDGE_DISCOVERY_TIMEOUT`` overrides it.
DEFAULT_TIMEOUT_S = float(
    os.getenv("KNOWLEDGE_DISCOVERY_TIMEOUT", "900").strip() or 900
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in an LLM answer (fenced, bare, or surrounded by prose)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    # A bare object: from the first '{' to the matching last '}'.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _challenge(task: str, state: Dict[str, Any], role: str) -> Dict[str, Any]:
    """What the scout is asked to solve."""
    return {
        "task": task,
        "agent": role,
        "customer": str(state.get("customer_id") or ""),
        "project": str(state.get("project_id") or ""),
        "namespace": str(state.get("namespace") or ""),
    }


def _system_prompt(challenge: Dict[str, Any]) -> str:
    from app.core.spec.loader import load_agent_manifest

    instructions = ""
    try:
        manifest = load_agent_manifest(SCOUT_AGENT)
        if manifest is not None:
            instructions = str(getattr(manifest, "body", "") or "")
    except Exception as exc:  # noqa: BLE001 - the prompt still works without it
        logger.warning("[KNOWLEDGE_DISCOVERY] scout manifest unreadable: %s", exc)

    return "\n".join(
        [
            instructions or (
                "You are the KNOWLEDGE SCOUT. Search the corpus, research the "
                "internet for what is missing, install and assign, then answer "
                "with one JSON object."
            ),
            "",
            "---",
            "",
            "CHALLENGE (the task a worker is about to run):",
            "```json",
            json.dumps(challenge, indent=2),
            "```",
            "",
            "Do the work now: round 1 (existing corpus), round 2 (internet), "
            "round 3 (install + assign). Then answer with the JSON object "
            "described in your instructions and nothing else.",
        ]
    )


async def run_discovery(
    task: str,
    state: Dict[str, Any],
    *,
    agent: Any = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the scout for ``task``; return its report (never raises).

    ``agent`` is a NodeAgent with tools already initialised; ``None`` means
    discovery is unavailable and the report says exactly that.
    """
    role = str(
        state.get("active_agent")
        or state.get("current_agent")
        or (state.get("_selected_agent") or "")
        or ""
    )
    challenge = _challenge(task, state, role)
    started = time.monotonic()

    if agent is None:
        return _failed(
            challenge,
            "knowledge discovery has no agent runtime — the scout could not be "
            "started (is the knowledge-base plugin active?)",
        )

    try:
        import asyncio

        limit = float(timeout_s or DEFAULT_TIMEOUT_S)
        result = await asyncio.wait_for(
            agent._execute_task_with_tools(
                prompt=(
                    "Execute knowledge discovery as described in your system "
                    "prompt, then answer with the JSON report."
                ),
                messages=[],
                system_prompt=_system_prompt(challenge),
            ),
            timeout=limit,
        )
        content = result.content if hasattr(result, "content") else str(result)
    except asyncio.TimeoutError:
        # The run was cut off, but its TOOL CALLS already happened: in a real
        # run the scout chunked two vault documents and assigned them before
        # the deadline, and this branch then reported "failed" and dropped the
        # evidence of work that is visible in Qdrant and the assignment table.
        # Report the timeout AND what it managed to do, so the worker is not
        # told nothing happened when things did.
        logger.warning(
            "[KNOWLEDGE_DISCOVERY] scout hit the %gs limit; reporting what it "
            "completed before the deadline",
            limit,
        )
        return _timed_out(challenge, limit, state)
    except Exception as exc:  # noqa: BLE001 - a failed run is reported, not raised
        logger.warning("[KNOWLEDGE_DISCOVERY] scout run failed: %s", exc)
        return _failed(challenge, f"the scout run failed: {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - started
    report = _extract_json(content)
    if report is None:
        logger.warning(
            "[KNOWLEDGE_DISCOVERY] scout answered without a JSON report (%d chars)",
            len(content or ""),
        )
        return _failed(
            challenge,
            "the scout answered but not with a JSON report — its text is kept "
            "below so the run is not a blank",
            raw=str(content or "")[:4000],
        )

    report.setdefault("step", "knowledge_discovery")
    report.setdefault("task", task)
    report["_meta"] = {
        "agent": SCOUT_AGENT,
        "elapsed_s": round(elapsed, 1),
        "source": "llm",
    }
    return report


def _failed(
    challenge: Dict[str, Any], reason: str, raw: str = ""
) -> Dict[str, Any]:
    """A report that says the run did not produce an answer, and why."""
    report: Dict[str, Any] = {
        "step": "knowledge_discovery",
        "task": challenge.get("task", ""),
        "round1_existing": {"queries": [], "found": [], "gap": ""},
        "round2_web": {"queries": [], "candidates": [], "needs_decision": []},
        "round3_installed": {"chunked": [], "assigned": [], "skipped": []},
        "answer_for_worker": (
            "Knowledge discovery did not complete: " + reason + ". "
            "Use the corpus tools directly if you need material."
        ),
        "error": reason,
        "_meta": {"agent": SCOUT_AGENT, "source": "unavailable"},
    }
    if raw:
        report["raw_answer"] = raw
    return report


def _timed_out(
    challenge: Dict[str, Any], limit: float, state: Dict[str, Any]
) -> Dict[str, Any]:
    """A timeout report that still says what the run actually changed.

    Reads the store and the corpus, because those are the record of what
    happened — the scout's own JSON never arrived.
    """
    report = _failed(
        challenge,
        f"the scout reached the {limit:g}s limit before answering",
    )
    report["_meta"]["source"] = "timeout"

    role = str(challenge.get("agent") or "")
    assigned: List[Dict[str, Any]] = []
    if role:
        try:
            from langgraph_crew_knowledge_base.ladder import resolve_agent

            resolved = resolve_agent(
                role,
                customer=str(challenge.get("customer") or ""),
                project=str(challenge.get("project") or ""),
            )
            for entry in resolved.get("store") or []:
                assigned.append(
                    {
                        "agent_id": role,
                        "book_ids": [entry.get("value")],
                        "scope": entry.get("scope"),
                        "customer": entry.get("customer"),
                        "project": entry.get("project"),
                        "why": "recorded before the deadline",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - a report must not raise
            logger.warning("[KNOWLEDGE_DISCOVERY] timeout report: %s", exc)

    report["round3_installed"]["assigned"] = assigned
    if assigned:
        report["answer_for_worker"] = (
            f"Knowledge discovery hit its {limit:g}s limit before writing its "
            f"report, but it did assign {len(assigned)} title(s) to {role}: "
            + ", ".join(a["book_ids"][0] for a in assigned)
            + ". Those are in effect; the search summary is missing."
        )
    return report


def report_as_findings(report: Dict[str, Any]) -> Dict[str, Any]:
    """The scout's report in the shape the executor renders for the worker.

    ``build_knowledge_discovery_report_message`` reads ``assigned_titles``,
    ``assigned_categories``, ``passages``, ``recommendations``. The scout's
    answer is richer (two rounds + install), so the report is kept whole under
    ``report`` and the worker-facing keys are derived from it.
    """
    round1 = report.get("round1_existing") or {}
    round3 = report.get("round3_installed") or {}
    found = [f for f in (round1.get("found") or []) if isinstance(f, dict)]

    titles: List[str] = []
    for entry in found:
        book_id = str(entry.get("book_id") or "").strip()
        if book_id and book_id not in titles:
            titles.append(book_id)

    recommendations = [
        {
            "title": str(c.get("title") or ""),
            "reason": str(c.get("reason") or ""),
            "source_url": str(c.get("source_url") or ""),
            "license": str(c.get("license") or ""),
        }
        for c in ((report.get("round2_web") or {}).get("candidates") or [])
        if isinstance(c, dict) and c.get("title")
    ]

    return {
        "step": "knowledge_discovery",
        "task": report.get("task", ""),
        "assigned_titles": titles,
        "assigned_categories": [],
        "recommendations": recommendations,
        "passages": [],
        "chunking_jobs": [
            j for j in (round3.get("chunked") or []) if isinstance(j, dict)
        ],
        "assignments": [
            a for a in (round3.get("assigned") or []) if isinstance(a, dict)
        ],
        "answer_for_worker": report.get("answer_for_worker", ""),
        "gap": round1.get("gap", ""),
        "needs_decision": (report.get("round2_web") or {}).get("needs_decision", []),
        "error": report.get("error", ""),
        # The complete answer of both rounds — what the worker actually reads.
        "report": report,
    }


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "SCOUT_AGENT",
    "report_as_findings",
    "run_discovery",
]
