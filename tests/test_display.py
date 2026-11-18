#!/usr/bin/env python3
"""The knowledge base must show WHAT it injected — in every output mode.

Other injections print one count line outside verbose mode. The corpus is the
factual basis of a run, so the operator has to see the passages themselves.
"""

import re

import pytest

from app.core.cli_styling import display_knowledge_injection


def _plain(text: str) -> str:
    """Strip Rich's ANSI markup (it splits numbers into per-digit spans)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
from app.core.global_state import ToolOutputMode, get_tool_output_mode, set_tool_output_mode, set_memory_debug_mode


BODY = (
    "KNOWLEDGE DISCOVERY RESULT (complete answer)\n"
    "RELEVANT PASSAGES:\n"
    "  [knowledge_base:technical-analysis/bollinger#s:x]\n"
    "    The Squeeze signal is a setup condition, not a trade trigger.\n"
)


@pytest.fixture(autouse=True)
def _restore_mode():
    previous = get_tool_output_mode()
    yield
    set_tool_output_mode(previous)
    set_memory_debug_mode(False)


class TestEveryMode:
    @pytest.mark.parametrize(
        "mode",
        [ToolOutputMode.SUPER_SHORT, ToolOutputMode.SHORT, ToolOutputMode.VERBOSE],
    )
    def test_header_and_content_appear(self, mode, capsys):
        # Memory debug mode ON shows the full content
        set_memory_debug_mode(True)
        set_tool_output_mode(mode)
        display_knowledge_injection("task-start", 3, 1290, BODY,
                                    refs=["knowledge_base:technical-analysis/bollinger#s:x"])
        out = _plain(capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out, f"no header in {mode.name}"
        assert "Squeeze signal" in out, f"content hidden in {mode.name}"

    def test_summary_line_shows_counts(self, capsys):
        set_memory_debug_mode(True)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection("mid-run", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "mid-run" in out
        assert "3" in out
        assert "1'290" in out or "1290" in out

    def test_body_optional_uses_refs(self, capsys):
        set_memory_debug_mode(True)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection("task-start", 0, 0, None,
                                    refs=["knowledge_base:a#s:1"])
        out = _plain(capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out
        assert "knowledge_base:a#s:1" in out

    def test_never_raises_without_body_or_refs(self, capsys):
        set_memory_debug_mode(True)
        set_tool_output_mode(ToolOutputMode.SUPER_SHORT)
        display_knowledge_injection("task-start")
        assert "KNOWLEDGE BASE" in _plain(capsys.readouterr().out)
