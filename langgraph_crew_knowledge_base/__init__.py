"""Knowledge base plugin (penta-knowledge-base) for LangGraph Crew / PENTA.

A shared knowledge corpus with:
  * ``render``          – ONE invitation line per chunk, read tool offers broader
  * ``provider``        – core seam for retrieval, injection, catalog access
  * ``tools``           – knowledge_base___search, knowledge_base___read

Registration is via the PluginHost (provider kind ``knowledge_base``) or the
entry point. ``__init__`` stays import-light so standalone scripts run without
heavy plugin imports.
"""