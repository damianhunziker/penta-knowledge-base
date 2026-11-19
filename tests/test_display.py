#!/usr/bin/env python3
"""The corpus is a memory type, so it renders in the MEMORY INJECTION panel.

The knowledge base supplies past material for the same reason episodes, rules
and the agent's own history do. The operator has ONE place to look at what
reached the prompt — printing the corpus as a separate line elsewhere made it
easy to miss, and the request was explicit: show it in the same panel.

Memory debug OFF still shows the block, as one compact line, so the corpus never
becomes invisible.
"""

import re

import pytest

from app.core.cli_styling import display_knowledge_injection
from app.core.global_state import (
    InjectionOutputMode,
    ToolOutputMode,
    get_tool_output_mode,
    set_injection_output_mode,
    set_memory_debug_mode,
    set_tool_output_mode,
)


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


BODY = (
    "KNOWLEDGE DISCOVERY RESULT\n"
    "[kb:graphic-design/meggs p.267]\n"
    "> ordinary images turned into convincing photographs\n"
)


@pytest.fixture(autouse=True)
def _restore():
    mode = get_tool_output_mode()
    yield
    set_tool_output_mode(mode)
    set_memory_debug_mode(False)
    set_injection_output_mode(InjectionOutputMode.AUTO)


class TestMemoryPanel:
    def test_renders_as_a_panel_with_memory_debug_on(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out
        assert "task-start" in out
        # A real panel, not a plain line.
        assert "╭" in out and "╰" in out

    def test_the_passage_text_is_inside_the_panel(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "convincing photographs" in out

    def test_counts_are_in_the_title(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("mid-run", 3, 1290, BODY)
        out = _plain(capsys.readouterr().out)
        assert "3 chunk(s)" in out
        assert "1'290" in out or "1290" in out
        assert "mid-run" in out

    @pytest.mark.parametrize(
        "mode",
        [ToolOutputMode.SUPER_SHORT, ToolOutputMode.SHORT, ToolOutputMode.VERBOSE],
    )
    def test_panel_is_shown_in_every_output_mode(self, mode, capsys):
        """Output verbosity is about the tool UI, not about hiding injections."""
        set_memory_debug_mode(True)
        set_tool_output_mode(mode)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out, f"hidden in {mode.name}"
        assert "convincing photographs" in out


class TestCompactFallback:
    def test_debug_off_still_reports_the_block(self, capsys):
        set_memory_debug_mode(False)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "[INJECTION] knowledge (task-start)" in out
        assert "3 chunk(s)" in out
        # The body stays out of the compact line.
        assert "convincing photographs" not in out

    def test_refs_are_used_when_no_body(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("task-start", 0, 0, None,
                                    refs=["knowledge_base:a#s:1"])
        out = _plain(capsys.readouterr().out)
        assert "knowledge_base:a#s:1" in out

    def test_never_raises_without_anything(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("task-start")
        out = _plain(capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out
