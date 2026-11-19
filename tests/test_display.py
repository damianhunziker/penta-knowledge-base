#!/usr/bin/env python3
"""The corpus section of the ONE injection report.

There used to be a 🧠 panel for memory, a 📚 panel for the corpus and a 💉 line
for the middle ledger — three formats for the same question. Everything now goes
through ``app.core.injection_report``; this file pins the corpus's part of it.

What matters here, and is easy to lose:

* the corpus section names its kind and how many passages it carries
* how it was ORDERED is always shown (cross-encoder vs. lexical fallback) —
  a degraded ordering is the thing an operator cannot otherwise notice
* memory debug OFF hides the BODY but never the ordering line
"""

import re

import pytest

from app.core import injection_report
from app.core.cli_styling import _render_injection_report, display_knowledge_injection
from app.core.global_state import set_memory_debug_mode


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


BODY = (
    "KNOWLEDGE DISCOVERY RESULT\n"
    "[kb:graphic-design/meggs p.267]\n"
    "> ordinary images turned into convincing photographs\n"
)


@pytest.fixture(autouse=True)
def _clean():
    injection_report.reset()
    yield
    injection_report.reset()
    set_memory_debug_mode(False)


class TestCorpusSection:
    def test_section_is_recorded_with_counts(self):
        display_knowledge_injection("task-start", 3, 1226, BODY)
        entries = injection_report.entries()
        assert len(entries) == 1
        assert entries[0].count == 3
        assert entries[0].kind == "knowledge"

    def test_body_is_shown_when_memory_debug_is_on(self, capsys):
        set_memory_debug_mode(True)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "convincing photographs" in out

    def test_body_is_hidden_but_counts_stay_when_debug_is_off(self, capsys):
        set_memory_debug_mode(False)
        display_knowledge_injection("task-start", 3, 1226, BODY)
        out = _plain(capsys.readouterr().out)
        assert "3 item(s)" in out
        assert "convincing photographs" not in out
        assert "bodies hidden" in out

    def test_repeats_replace_instead_of_accumulating(self):
        display_knowledge_injection("task-start", 3, 1226, BODY)
        display_knowledge_injection("task-start", 2, 900, BODY)
        assert len(injection_report.entries()) == 1
        assert injection_report.entries()[0].count == 2


class TestOrderingIsVisible:
    def test_cross_encoder_is_named(self, capsys):
        set_memory_debug_mode(False)
        injection_report.record(
            "knowledge", count=3, body=BODY, ordering="cross_encoder"
        )
        _render_injection_report()
        out = _plain(capsys.readouterr().out)
        assert "ordering: cross-encoder" in out

    def test_lexical_fallback_is_named_with_its_reason(self, capsys):
        set_memory_debug_mode(False)
        injection_report.record(
            "knowledge", count=3, body=BODY,
            ordering="lexical_fallback",
            ordering_note="HTTPError: HTTP Error 500",
        )
        _render_injection_report()
        out = _plain(capsys.readouterr().out)
        assert "ordering: lexical fallback" in out
        assert "HTTP Error 500" in out

    def test_the_header_counts_degraded_sections(self, capsys):
        set_memory_debug_mode(False)
        injection_report.record("episodic", count=2, ordering="cross_encoder")
        injection_report.record(
            "long_term", count=2, ordering="lexical_fallback",
            ordering_note="unreachable",
        )
        _render_injection_report()
        out = _plain(capsys.readouterr().out)
        assert "1 with a degraded ordering" in out


class TestAllTypesInOneReport:
    def test_every_type_appears_in_one_render(self, capsys):
        set_memory_debug_mode(True)
        for kind in ("episodic", "procedural", "long_term", "knowledge"):
            injection_report.record(kind, count=1, body=f"{kind} body",
                                    ordering="cross_encoder")
        _render_injection_report()
        out = _plain(capsys.readouterr().out)
        for kind in ("episodic", "procedural", "long_term", "knowledge"):
            assert f"── {kind}" in out, out
        # One report, ordered broad -> specific, not four separate prints.
        assert out.index("── episodic") < out.index("── knowledge")

    def test_nothing_recorded_renders_nothing(self):
        assert injection_report.render() is None
