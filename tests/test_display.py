#!/usr/bin/env python3
"""A knowledge injection is reported like every other injection.

Memory, the middle ledger and chat history all print one
``💉 [INJECTION] <kind> — <size>`` line (body only when injection mode allows).
The corpus used to print its own 📚 panel instead, so the operator had to read
two formats for the same information. These tests pin the ONE format.
"""

import re

import pytest

from app.core.cli_styling import display_knowledge_injection
from app.core.global_state import (
    InjectionOutputMode,
    ToolOutputMode,
    get_injection_output_mode,
    get_tool_output_mode,
    set_injection_output_mode,
    set_tool_output_mode,
)


def _plain(text: str) -> str:
    """Strip Rich's ANSI markup (it splits numbers into per-digit spans)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


BODY = (
    "KNOWLEDGE DISCOVERY RESULT (complete answer)\n"
    "RELEVANT PASSAGES:\n"
    "  [knowledge_base:technical-analysis/bollinger#s:x]\n"
    "    The Squeeze signal is a setup condition, not a trade trigger.\n"
)


@pytest.fixture(autouse=True)
def _restore_mode():
    previous_mode = get_tool_output_mode()
    previous_injection = get_injection_output_mode()
    yield
    set_tool_output_mode(previous_mode)
    set_injection_output_mode(previous_injection)


class TestSharedInjectionFormat:
    @pytest.mark.parametrize(
        "mode",
        [ToolOutputMode.SUPER_SHORT, ToolOutputMode.SHORT, ToolOutputMode.VERBOSE],
    )
    def test_uses_the_injection_line(self, mode, capsys):
        """Same marker, same wording as memory — in every output mode."""
        set_injection_output_mode(InjectionOutputMode.FULL)
        set_tool_output_mode(mode)
        display_knowledge_injection("task-start", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "[INJECTION] knowledge (task-start)" in out, out
        assert "💉" in out

    @pytest.mark.parametrize(
        "mode",
        [ToolOutputMode.SUPER_SHORT, ToolOutputMode.SHORT, ToolOutputMode.VERBOSE],
    )
    def test_body_follows_injection_mode(self, mode, capsys):
        set_injection_output_mode(InjectionOutputMode.FULL)
        set_tool_output_mode(mode)
        display_knowledge_injection("task-start", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "Squeeze signal" in out, f"body hidden in {mode.name}"

    def test_summary_shows_counts(self, capsys):
        set_injection_output_mode(InjectionOutputMode.FULL)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection("mid-run", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "mid-run" in out
        assert "3 chunk(s)" in out
        assert "1'290" in out or "1290" in out

    def test_no_body_falls_back_to_refs(self, capsys):
        set_injection_output_mode(InjectionOutputMode.FULL)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection(
            "task-start", 0, 0, None, refs=["knowledge_base:a#s:1"]
        )
        out = _plain(capsys.readouterr().out)
        assert "knowledge_base:a#s:1" in out

    def test_never_raises_without_body_or_refs(self, capsys):
        set_injection_output_mode(InjectionOutputMode.FULL)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection("task-start")
        out = _plain(capsys.readouterr().out)
        assert "[INJECTION] knowledge" in out

    def test_compact_mode_hides_the_body(self, capsys):
        """Outside the shared 'show bodies' mode only the line is printed."""
        set_injection_output_mode(InjectionOutputMode.COMPACT)
        set_tool_output_mode(ToolOutputMode.SHORT)
        display_knowledge_injection("task-start", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "[INJECTION] knowledge (task-start)" in out
        assert "3 chunk(s)" in out
        assert "Squeeze signal" not in out, "the body must stay hidden"
