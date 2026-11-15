#!/usr/bin/env python3
"""Tests for the knowledge base renderer."""

import sys
from pathlib import Path

# Add the plugin to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from langgraph_crew_knowledge_base.render import (
    Chunk,
    Reference,
    derive_broader_refs,
    render_chunk,
    render_read_response,
)


@pytest.fixture
def sample_reference() -> Reference:
    """Create a sample reference for testing."""
    return Reference(
        kb_ref="knowledge_base:academic/meadows-2008/ch3#s:1.2",
        title="Thinking in Systems",
        authors=("Meadows, Donella H.",),
        year=2008,
        page_from=118,
        page_to=119,
        section_path="ch3.1.2",
        tier="foundational",
        book_id="meadows-2008",
        corpus="academic",
    )


@pytest.fixture
def sample_chunk(sample_reference: Reference) -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        text="The leverage points are the places in the system where a small change can yield large shifts in behaviour.",
        reference=sample_reference,
        score=0.81,
    )


class TestRenderChunk:
    """Tests for render_chunk()."""

    def test_renders_header(self, sample_chunk: Chunk):
        """Header contains kb_ref, pages, tier, score."""
        output = render_chunk(sample_chunk)
        assert "[kb:academic/meadows-2008/ch3#s:1.2" in output
        assert "p.118–119" in output
        assert "tier=foundational" in output
        assert "score 0.81" in output

    def test_renders_citation(self, sample_chunk: Chunk):
        """Citation contains author, title, year, section."""
        output = render_chunk(sample_chunk)
        assert "Meadows, Donella H." in output
        assert "*Thinking in Systems*" in output
        assert "(2008)" in output
        assert '"ch3.1.2"' in output

    def test_renders_one_invitation(self, sample_chunk: Chunk):
        """Output contains exactly ONE invitation line."""
        output = render_chunk(sample_chunk)
        # Count invitation lines (lines starting with ↳)
        invitation_lines = [line for line in output.split("\n") if line.startswith("↳")]
        assert len(invitation_lines) == 1, f"Expected 1 invitation, got {len(invitation_lines)}"
        assert "knowledge_base___read(kb_ref=" in invitation_lines[0]

    def test_invitation_contains_full_ref(self, sample_chunk: Chunk):
        """Invitation contains the full kb_ref."""
        output = render_chunk(sample_chunk)
        assert 'kb_ref="knowledge_base:academic/meadows-2008/ch3#s:1.2"' in output


class TestDeriveBroaderRefs:
    """Tests for derive_broader_refs()."""

    def test_section_ref_derives_chapter(self):
        """A section ref derives its chapter."""
        result = derive_broader_refs("knowledge_base:academic/meadows-2008/ch3#s:1.2")
        assert result["section"] == "knowledge_base:academic/meadows-2008/ch3#s:1.2"
        assert result["chapter"] == "knowledge_base:academic/meadows-2008/ch3"

    def test_section_ref_derives_neighbours(self):
        """A section ref derives neighbours call."""
        result = derive_broader_refs("knowledge_base:academic/meadows-2008/ch3#s:1.2")
        assert result["neighbours_call"] is not None
        assert "before=1, after=1" in result["neighbours_call"]

    def test_chapter_ref_derives_book(self):
        """A chapter ref derives its book."""
        result = derive_broader_refs("knowledge_base:academic/meadows-2008/ch3")
        assert result["chapter"] == "knowledge_base:academic/meadows-2008/ch3"
        assert result["book"] == "knowledge_base:academic/meadows-2008"


class TestRenderReadResponse:
    """Tests for render_read_response()."""

    def test_section_response_offers_chapter(self, sample_reference: Reference):
        """Reading a section offers to read the chapter."""
        output = render_read_response(
            text="Full section text...",
            ref=sample_reference,
            budget_used=1000,
            budget_max=2500,
        )
        assert '↳ read chapter' in output
        assert 'knowledge_base:academic/meadows-2008/ch3"' in output

    def test_section_response_offers_neighbours(self, sample_reference: Reference):
        """Reading a section offers neighbours."""
        output = render_read_response(
            text="Full section text...",
            ref=sample_reference,
            budget_used=1000,
            budget_max=2500,
        )
        assert '↳ neighbours' in output

    def test_truncated_response_shows_notice(self, sample_reference: Reference):
        """Truncated response shows budget notice."""
        output = render_read_response(
            text="Full section text...",
            ref=sample_reference,
            budget_used=2500,
            budget_max=2500,
            truncated=True,
        )
        assert "[truncated at 2500/2500 tok" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])