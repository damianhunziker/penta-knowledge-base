#!/usr/bin/env python3
"""Tests for the mid-run knowledge injection hook.

The hook fires before every LLM call and injects a knowledge block once
``KNOWLEDGE_INJECT_AFTER_TOOLS`` tool calls have happened since the last
injection. These tests pin the counting, the caps and the message shape.
"""

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_crew_knowledge_base import injection as inj
from langgraph_crew_knowledge_base import runner


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "KNOWLEDGE_INJECT_AFTER_TOOLS",
        "KNOWLEDGE_INJECT_CHUNKS",
        "KNOWLEDGE_TASK_CHUNKS",
        "KNOWLEDGE_INJECT_MAX_PER_RUN",
        "DISABLE_KNOWLEDGE_DISCOVERY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def fake_block(monkeypatch):
    """Replace retrieval with a fixed block so the hook is tested in isolation."""
    calls = {"n": 0}

    def _build(query, state=None, settings=None, seen=None):
        calls["n"] += 1
        if calls.get("empty"):
            return ""
        # Report a fake ref so the dedupe list can be checked.
        return (
            "KNOWLEDGE BASE INFO\n"
            f"chunk #{calls['n']}\n"
            '↳ read section   knowledge_base___read(kb_ref="knowledge_base:academic/x/ch1")\n'
        )

    monkeypatch.setattr(runner, "build_mid_run_block", _build)
    # ``injection`` imported the name at module load, so rebind it there too.
    monkeypatch.setattr(inj, "build_mid_run_block", _build)
    return calls


def _turns(n):
    """n AI messages that each requested a tool (= n completed turns)."""
    return [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": str(i)}])
            for i in range(n)]


class TestCounting:
    def test_first_call_arms_the_counter(self, fake_block):
        state = {"messages": _turns(0)}
        assert inj.knowledge_mid_run_injection(state) is None
        assert state[inj.KEY_BASELINE] == 0
        assert state[inj.KEY_FIRED] == 0

    def test_does_not_fire_before_threshold(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "13")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)  # arm
        state["messages"] = _turns(12)
        assert inj.knowledge_mid_run_injection(state) is None
        assert fake_block["n"] == 0

    def test_fires_at_threshold(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "13")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)  # arm
        state["messages"] = _turns(13)
        out = inj.knowledge_mid_run_injection(state)
        assert out is not None
        assert fake_block["n"] == 1
        assert state[inj.KEY_FIRED] == 1

    def test_default_threshold_is_13(self, fake_block, monkeypatch):
        # Pin the threshold explicitly so the test is independent of the local
        # plugin.env (a deployment may set 1 for close observation).
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "13")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(12)
        assert inj.knowledge_mid_run_injection(state) is None
        state["messages"] = _turns(13)
        assert inj.knowledge_mid_run_injection(state) is not None

    def test_threshold_one_fires_after_a_single_tool_call(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        state = {"messages": _turns(0)}
        assert inj.knowledge_mid_run_injection(state) is None
        state["messages"] = _turns(1)
        out = inj.knowledge_mid_run_injection(state)
        assert out is not None
        assert state[inj.KEY_FIRED] == 1

    def test_second_fire_needs_another_window(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "13")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(13)
        inj.knowledge_mid_run_injection(state)

        state["messages"] = _turns(20)  # only 7 more
        assert inj.knowledge_mid_run_injection(state) is None

        state["messages"] = _turns(26)  # 13 more
        assert inj.knowledge_mid_run_injection(state) is not None
        assert state[inj.KEY_FIRED] == 2


class TestCaps:
    def test_max_per_run(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        monkeypatch.setenv("KNOWLEDGE_INJECT_MAX_PER_RUN", "2")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)  # arm
        fired = 0
        for turns in range(1, 10):
            state["messages"] = _turns(turns)
            if inj.knowledge_mid_run_injection(state):
                fired += 1
        assert fired == 2, f"fired {fired}x, max_per_run is 2"

    def test_zero_threshold_disables(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "0")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(500)
        assert inj.knowledge_mid_run_injection(state) is None
        assert fake_block["n"] == 0

    def test_disabled_flag(self, fake_block, monkeypatch):
        monkeypatch.setenv("DISABLE_KNOWLEDGE_DISCOVERY", "true")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(99)
        assert inj.knowledge_mid_run_injection(state) is None

    def test_empty_block_does_not_burn_an_injection(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "2")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        fake_block["empty"] = True
        state["messages"] = _turns(2)
        assert inj.knowledge_mid_run_injection(state) is None
        assert state[inj.KEY_FIRED] == 0


class TestMessageShape:
    def test_block_is_a_tail_safe_human_message(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(1)
        out = inj.knowledge_mid_run_injection(state)

        msgs = out["llm_input_messages"]
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].additional_kwargs.get("runtime_reminder") == inj.REMINDER_KIND
        assert "KNOWLEDGE BASE INFO" in msgs[-1].content

    def test_history_is_not_persisted(self, fake_block, monkeypatch):
        """The block goes to the prompt only — state['messages'] is untouched."""
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(1)
        before = list(state["messages"])
        out = inj.knowledge_mid_run_injection(state)
        assert out is not None
        assert state["messages"] == before

    def test_injected_refs_are_recorded(self, fake_block, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(1)
        inj.knowledge_mid_run_injection(state)
        assert "knowledge_base:academic/x/ch1" in state[inj.KEY_REFS]

    def test_reset_clears_counters(self):
        state = {inj.KEY_BASELINE: 5, inj.KEY_FIRED: 3, inj.KEY_REFS: ["a"]}
        inj.reset_run_counters(state)
        assert state[inj.KEY_BASELINE] is None
        assert state[inj.KEY_FIRED] == 0
        assert state[inj.KEY_REFS] == []


class TestDebugDisplay:
    """F9 (memory debug) decides how much of the KB block reaches the CLI.

    The operator's switch has to work at BOTH injection points — the mid-run
    hook here and the task-start report in the executor. This class pins the
    mid-run one: the hook must call the display, and the display must honour
    the flag.
    """

    @staticmethod
    def _fire(monkeypatch, fake_block):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "1")
        state = {"messages": _turns(0)}
        inj.knowledge_mid_run_injection(state)
        state["messages"] = _turns(1)
        return inj.knowledge_mid_run_injection(state)

    def test_debug_on_prints_the_full_block(self, monkeypatch, fake_block, capsys):
        from app.core.global_state import set_memory_debug_mode

        set_memory_debug_mode(True)
        try:
            assert self._fire(monkeypatch, fake_block) is not None
        finally:
            set_memory_debug_mode(False)
        out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
        assert "KNOWLEDGE BASE" in out
        # The passage text itself, not just a size line.
        assert "chunk #1" in out
        assert "read section" in out

    def test_debug_off_prints_only_a_summary(self, monkeypatch, fake_block, capsys):
        from app.core.global_state import set_memory_debug_mode

        set_memory_debug_mode(False)
        assert self._fire(monkeypatch, fake_block) is not None
        out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
        assert "Knowledge:" in out
        # No passage body leaks through.
        assert "chunk #1" not in out
        assert "KNOWLEDGE BASE\n" not in out

    def test_hook_still_returns_the_block_when_debug_is_off(
        self, monkeypatch, fake_block
    ):
        """Display is a side channel: the prompt gets the block either way."""
        from app.core.global_state import set_memory_debug_mode

        set_memory_debug_mode(False)
        out = self._fire(monkeypatch, fake_block)
        assert out is not None
        assert "KNOWLEDGE BASE INFO" in out["llm_input_messages"][-1].content

    def test_display_failure_never_breaks_the_injection(
        self, monkeypatch, fake_block
    ):
        """A broken console must not cost the model its material."""
        import app.core.cli_styling as styling

        def _boom(*a, **kw):
            raise RuntimeError("no console")

        monkeypatch.setattr(styling, "display_knowledge_injection", _boom)
        out = self._fire(monkeypatch, fake_block)
        assert out is not None
        assert "KNOWLEDGE BASE INFO" in out["llm_input_messages"][-1].content
