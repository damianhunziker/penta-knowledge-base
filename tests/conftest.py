"""Test configuration for the knowledge-base plugin.

Puts both the repo root and the plugin root on sys.path so that
langgraph_crew_knowledge_base and core modules are importable.
"""

import sys
from pathlib import Path

# Add the plugin root (for langgraph_crew_knowledge_base)
plugin_root = Path(__file__).parent.parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

# Add the repo root (for app.core modules)
repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ── Keep the cross-encoder out of the unit tests ───────────────────────────
#
# The relevance gate prefers a live cross-encoder when one answers, and the
# gateway on this host does. That is correct in production but wrong in a unit
# test: the outcome would depend on a network service being up, and the tests
# that assert WHICH passages survive would change whenever the reranker's
# ordering differs. The gate is exercised on its own in
# pytest/unit/test_memory_gate.py, with the scores injected explicitly.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_cross_encoder(monkeypatch):
    """Disable the cross-encoder for every knowledge-base test."""
    import os

    monkeypatch.setenv("RERANK_ENABLED", "false")
    try:
        from app.core import memory_gate

        monkeypatch.setattr(
            memory_gate, "cross_encoder_scores", lambda *a, **kw: None,
            raising=False,
        )
    except Exception:
        pass
    yield
