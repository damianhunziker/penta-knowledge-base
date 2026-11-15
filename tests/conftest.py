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