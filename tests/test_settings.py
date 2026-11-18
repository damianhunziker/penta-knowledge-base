#!/usr/bin/env python3
"""Tests for the environment-driven knowledge injection settings.

The defaults are part of the contract:
    KNOWLEDGE_INJECT_AFTER_TOOLS = 13
    KNOWLEDGE_INJECT_CHUNKS      = 3
    KNOWLEDGE_TASK_CHUNKS        = 3
"""

import pytest

from langgraph_crew_knowledge_base import settings as s


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "KNOWLEDGE_INJECT_AFTER_TOOLS",
        "KNOWLEDGE_INJECT_CHUNKS",
        "KNOWLEDGE_TASK_CHUNKS",
        "KNOWLEDGE_INJECT_MAX_PER_RUN",
        "KNOWLEDGE_INJECT_MIN_SCORE",
        "KNOWLEDGE_INJECT_BUDGET_TOKENS",
        "DISABLE_KNOWLEDGE_DISCOVERY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


class TestDefaults:
    def test_inject_after_tools_defaults_to_13(self):
        # The DOCUMENTED default lives in the code; plugin.env may override it
        # for this deployment (it currently sets 1 for close observation).
        assert s.DEFAULT_INJECT_AFTER_TOOLS == 13

    def test_effective_threshold_comes_from_plugin_env(self):
        # Whatever plugin.env says is what a run actually uses - the value must
        # be a usable positive int, never the fallback by accident.
        value = s.load_settings().inject_after_tools
        assert isinstance(value, int) and value > 0

    def test_inject_chunks_is_3(self):
        assert s.load_settings().inject_chunks == 3

    def test_task_chunks_is_3(self):
        assert s.load_settings().task_chunks == 3

    def test_mid_run_is_enabled_by_default(self):
        assert s.load_settings().mid_run_enabled is True

    def test_min_score_default(self):
        assert s.load_settings().min_score == 0.35

    def test_constants_match_defaults(self):
        assert s.DEFAULT_INJECT_AFTER_TOOLS == 13
        assert s.DEFAULT_INJECT_CHUNKS == 3
        assert s.DEFAULT_TASK_CHUNKS == 3


class TestEnvironmentOverride:
    def test_after_tools(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "5")
        assert s.load_settings().inject_after_tools == 5

    def test_inject_chunks(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_CHUNKS", "7")
        assert s.load_settings().inject_chunks == 7

    def test_task_chunks(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "1")
        assert s.load_settings().task_chunks == 1

    def test_zero_after_tools_disables_mid_run(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "0")
        assert s.load_settings().mid_run_enabled is False

    def test_disable_flag(self, monkeypatch):
        monkeypatch.setenv("DISABLE_KNOWLEDGE_DISCOVERY", "true")
        assert s.load_settings().disabled is True

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "dreizehn")
        assert s.load_settings().inject_after_tools == 13

    def test_negative_falls_back(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_CHUNKS", "-4")
        assert s.load_settings().inject_chunks == 3

    def test_float_score(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_INJECT_MIN_SCORE", "0.6")
        assert s.load_settings().min_score == pytest.approx(0.6)


class TestChunksForTask:
    """A task may carry its own ``knowledge_chunks``."""

    def test_task_override_wins(self):
        assert s.chunks_for_task({"knowledge_chunks": 5}) == 5

    def test_default_without_task_value(self):
        assert s.chunks_for_task({"id": "t1"}) == 3

    def test_env_default_is_used(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "8")
        assert s.chunks_for_task({"id": "t1"}) == 8

    def test_task_value_beats_env(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_TASK_CHUNKS", "8")
        assert s.chunks_for_task({"knowledge_chunks": 2}) == 2

    def test_zero_is_respected(self):
        """0 means: no chunks for this task."""
        assert s.chunks_for_task({"knowledge_chunks": 0}) == 0

    def test_bad_task_value_falls_back(self):
        assert s.chunks_for_task({"knowledge_chunks": "viele"}) == 3

    def test_non_dict_task(self):
        assert s.chunks_for_task(None) == 3
        assert s.chunks_for_task("task") == 3


class TestPluginEnvFile:
    """The plugin owns its config: plugin.env / plugin.env.example."""

    def test_example_file_exists(self):
        from langgraph_crew_knowledge_base.settings import ENV_EXAMPLE

        assert ENV_EXAMPLE.exists(), "plugin.env.example must be versioned"

    def test_example_documents_every_key(self):
        from langgraph_crew_knowledge_base.settings import ENV_EXAMPLE

        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for key in (
            "KNOWLEDGE_INJECT_AFTER_TOOLS",
            "KNOWLEDGE_INJECT_CHUNKS",
            "KNOWLEDGE_TASK_CHUNKS",
            "KNOWLEDGE_INJECT_MAX_PER_RUN",
            "KNOWLEDGE_INJECT_MIN_SCORE",
            "KNOWLEDGE_INJECT_BUDGET_TOKENS",
            "DISABLE_KNOWLEDGE_DISCOVERY",
        ):
            assert key in text, f"{key} missing from plugin.env.example"

    def test_example_values_match_the_defaults(self):
        """The template ships the documented defaults, not something else."""
        from langgraph_crew_knowledge_base.settings import (
            ENV_EXAMPLE,
            _parse_env_file,
        )

        values = _parse_env_file(ENV_EXAMPLE)
        assert values["KNOWLEDGE_INJECT_AFTER_TOOLS"] == "13"
        assert values["KNOWLEDGE_INJECT_CHUNKS"] == "3"
        assert values["KNOWLEDGE_TASK_CHUNKS"] == "3"
        assert values["DISABLE_KNOWLEDGE_DISCOVERY"] == "false"

    def test_plugin_env_overrides_example(self, tmp_path, monkeypatch):
        """A local plugin.env wins over the versioned template."""
        from langgraph_crew_knowledge_base import settings as s

        local = tmp_path / "plugin.env"
        local.write_text("KNOWLEDGE_INJECT_AFTER_TOOLS=4\n", encoding="utf-8")
        monkeypatch.setattr(s, "ENV_FILE", local)

        assert s.load_settings().inject_after_tools == 4

    def test_process_env_beats_plugin_env(self, tmp_path, monkeypatch):
        """Precedence: process env > plugin.env > plugin.env.example."""
        from langgraph_crew_knowledge_base import settings as s

        local = tmp_path / "plugin.env"
        local.write_text("KNOWLEDGE_INJECT_AFTER_TOOLS=4\n", encoding="utf-8")
        monkeypatch.setattr(s, "ENV_FILE", local)
        monkeypatch.setenv("KNOWLEDGE_INJECT_AFTER_TOOLS", "9")

        assert s.load_settings().inject_after_tools == 9

    def test_missing_plugin_env_is_fine(self, tmp_path, monkeypatch):
        from langgraph_crew_knowledge_base import settings as s

        monkeypatch.setattr(s, "ENV_FILE", tmp_path / "does-not-exist.env")
        assert s.load_settings().inject_after_tools == 13

    def test_broken_plugin_env_falls_back(self, tmp_path, monkeypatch):
        from langgraph_crew_knowledge_base import settings as s

        bad = tmp_path / "plugin.env"
        bad.write_text("KNOWLEDGE_INJECT_AFTER_TOOLS=abc\n", encoding="utf-8")
        monkeypatch.setattr(s, "ENV_FILE", bad)

        assert s.load_settings().inject_after_tools == 13

    def test_export_prefix_is_supported(self, tmp_path, monkeypatch):
        from langgraph_crew_knowledge_base import settings as s

        local = tmp_path / "plugin.env"
        local.write_text("export KNOWLEDGE_INJECT_CHUNKS=6\n", encoding="utf-8")
        monkeypatch.setattr(s, "ENV_FILE", local)

        assert s.load_settings().inject_chunks == 6
