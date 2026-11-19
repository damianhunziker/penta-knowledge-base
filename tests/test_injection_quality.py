#!/usr/bin/env python3
"""The injected passages must be CONTENT, and there must be as many as configured.

Two defects made a corpus of 1'600 real chunks look useless in the prompt:

1. **Budget starvation.** ``_trim_to_budget`` handed the whole token budget to
   the first hit (a 3'300-char book page = 898 of 900 tokens) and dropped the
   rest. A task configured for ``KNOWLEDGE_TASK_CHUNKS=3`` received one passage
   and nothing said so.
2. **Index pages winning retrieval.** A book index matches a topic query
   perfectly (every term of the query is in it) but contains only page numbers.
   The model was handed "Dynamic Diagrams' interface design, 179" as the
   passage for a question about colour and could only report that the corpus
   had nothing.
"""

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from langgraph_crew_knowledge_base import runner  # noqa: E402
from langgraph_crew_knowledge_base.render import Chunk, Reference  # noqa: E402


def _chunk(text, book="b", page=1, score=0.8):
    return Chunk(
        text=text,
        reference=Reference(
            kb_ref=f"knowledge_base:corpus/{book}#s:p{page}",
            title="Title",
            authors=("Author",),
            year=2020,
            page_from=page,
            page_to=None,
            section_path=f"p{page}",
            tier="standard",
            book_id=book,
            corpus="corpus",
        ),
        score=score,
    )


PROSE = (
    "Colour is the most relative medium in art. A palette chosen for a web "
    "interface has to work against the background it sits on, and the contrast "
    "between text and ground decides whether the page is readable at all.\n\n"
)


class TestBudgetDoesNotStarve:
    def test_every_requested_chunk_survives(self):
        """3 chunks requested, 3 chunks injected — not 1."""
        chunks = [_chunk(PROSE * 40, page=i) for i in range(1, 4)]
        kept = runner._trim_to_budget(chunks, 900)
        assert len(kept) == 3, f"only {len(kept)} of 3 survived"

    def test_the_budget_is_split_not_consumed(self):
        chunks = [_chunk(PROSE * 40, page=i) for i in range(1, 4)]
        kept = runner._trim_to_budget(chunks, 600)
        budgets = [(len(c.text) + 300) // 4 for c in kept]
        assert len(kept) == 3
        # No single chunk may eat the entire budget.
        assert max(budgets) <= 400

    def test_a_short_chunk_is_untouched(self):
        chunks = [_chunk("Short passage.")]
        kept = runner._trim_to_budget(chunks, 900)
        assert kept[0].text == "Short passage."

    def test_a_long_chunk_is_cut_and_says_so(self):
        chunks = [_chunk(PROSE * 40)]
        kept = runner._trim_to_budget(chunks, 300)
        assert kept[0].text != chunks[0].text
        assert "passage shortened" in kept[0].text
        # The reference stays whole: the reader can still open the section.
        assert kept[0].reference.kb_ref == chunks[0].reference.kb_ref

    def test_zero_budget_keeps_everything(self):
        chunks = [_chunk(PROSE * 40, page=i) for i in range(3)]
        assert len(runner._trim_to_budget(chunks, 0)) == 3


class TestIndexPagesAreDropped:
    def test_a_book_index_is_recognised(self):
        index = "\n".join(
            [
                "computer technologies, 169–183",
                "consumer-oriented Web site design, 181",
                "Dynamic Diagrams' interface design, 179",
                "interaction with, 172–174",
                "navigation in World Wide Web, 176–178",
                "text, use of, 175",
            ]
        )
        assert runner._is_index_page(index) is True

    def test_a_bibliography_is_recognised(self):
        bibliography = "\n".join(
            [
                "Scharff, L., Hill, A. & Ahumada, A. (2000). Discriminability",
                "measures for predicting readability of text. Optics Express.",
                "Schriver, K. A. (1997). Dynamics in document design. Wiley.",
                "Sundström, E. (2007). Den mångsidige målaren. Runica.",
                "Öberg, J. & Kihlman, E. (2012). Perspectives on design. Stockholm.",
            ]
        )
        assert runner._is_index_page(bibliography) is True

    def test_an_image_credits_page_is_recognised(self):
        credits = "\n".join(
            [
                "Silhouettes © 2008 Jupiterimages Corporation.",
                "Photographs of Kennedy and Nixon © Bettman/Corbis.",
                "Photograph of Teletubbies © 1996-2003 Ragdoll Ltd. Used with permission.",
                "Illustration reprinted by permission of the artist.",
                "Cover image courtesy of the publisher archives.",
            ]
        )
        assert runner._is_index_page(credits) is True

    def test_real_prose_is_not_dropped(self):
        assert runner._is_index_page(PROSE * 5) is False

    def test_a_page_with_a_few_numbers_is_prose(self):
        text = (
            "In 1990 Tim Berners-Lee developed the three building blocks of the "
            "Web at CERN in Geneva: HTTP, HTML and the URL scheme that made "
            "documents addressable. Designers then had to learn a new medium.\n\n"
        ) * 4
        assert runner._is_index_page(text) is False


class TestBuildTaskBlockUsesThem:
    @pytest.fixture
    def fake_search(self, monkeypatch):
        hits = [
            _chunk(PROSE * 30, book="real", page=10),
            _chunk("\n".join(f"entry {i}, {100 + i}" for i in range(8)), book="idx", page=216),
            _chunk(PROSE * 20, book="real", page=11),
            _chunk(PROSE * 10, book="real", page=12),
        ]
        monkeypatch.setattr(runner, "_search", lambda q, n, s: hits)
        return hits

    def test_index_page_is_skipped_and_count_is_honoured(self, fake_search, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "3")
        block = runner.build_task_block({"id": "t1", "description": "colour palettes"})
        assert block
        assert "idx" not in block, "the index page must not be injected"
        assert block.count("\n[kb:") == 3

    def test_an_all_index_corpus_still_returns_something(self, fake_search, monkeypatch):
        """Better a weak hit than an empty prompt."""
        index = _chunk("\n".join(f"entry {i}, {100 + i}" for i in range(8)))
        monkeypatch.setattr(runner, "_search", lambda q, n, s: [index])
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "3")
        block = runner.build_task_block({"id": "t1", "description": "colour"})
        assert block
