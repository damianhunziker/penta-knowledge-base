# Drop-in plugin for LangGraph Crew (plugins/knowledge-base).
# The PluginHost loads this file: it must expose `plugin` (Plugin instance) or
# `load_plugin()`. The actual code lives in the bundled package.
from langgraph_crew_knowledge_base.plugin import plugin

__all__ = ["plugin"]