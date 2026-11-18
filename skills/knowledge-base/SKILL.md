# Knowledge Base

**How to call:** `knowledge_base___*` are **plugin tools and are not bound** —
run every call as
`tool_registry___system_request(tool_name="knowledge_base___<tool>", arguments={...})`.
The quick-start snippets below show the inner tool and its positional arguments
in order.

Use the knowledge base for domain knowledge, reference material, and training memory.

## Quick Start

```
# Search corpus
tool_registry___system_request(tool_name="knowledge_base___search",
                              arguments={"query": "systems thinking"})

# Read section
tool_registry___system_request(tool_name="knowledge_base___read",
                              arguments={"kb_ref": "knowledge_base:academic/meadows-2008/ch3#s:1.2"})

# Ingest from vault
tool_registry___system_request(tool_name="knowledge_base___ingest_from_root",
                              arguments={"root_ref": "project-vault:Knowledge/book.md",
                                         "corpus": "academic"})
```

## kb_ref Grammar

```
knowledge_base:corpus/book-id                    # book
knowledge_base:corpus/book-id/chapter            # chapter  
knowledge_base:corpus/book-id/chapter#s:section  # section
```

## Tools

| Tool | Action |
|------|--------|
| `search` | Query corpus |
| `read` | Get section/chapter content |
| `list` | Browse by corpus/tier |
| `recommend_titles` | Get suggestions by topic |
| `ingest_from_root` | Add MD/PDF from vault |
| `assign_to_agent` | Give agent access to titles (store only) |
| `grant_access` | Give access AND declare it at the chosen scope |
| `access_matrix` | Show who may read what, and at which scope |

## Managing the corpus

`python plugins/knowledge-base/scripts/knowledge_manager.py` (or `/knowledge` in
the chat) lists bases and books, flags books that hold only metadata or bullets
instead of real text, shows every agent binding per scope, and shows how the
bindings differ per customer/project. It assigns and removes bindings through
the same store the tools use.

## Ingestion

Markdown with frontmatter:
```yaml
---
title: Book Title
authors: [Author]
tier: foundational
categories: [topic]
---

# Content...
```

PDF: uses Docling extraction (background job).

## Assignment and access scope

An assignment is "this agent always has these titles in scope". It is stored
with a **scope**, because a title given to an agent in one project must not
silently follow that agent into every other project.

| scope | who gets it | declared in |
|-------|-------------|-------------|
| `project` (default) | one project | `<project>/agents/<agent>.md` |
| `client` | every project of one customer | `<customer>/agents/<agent>.md` |
| `deployment` | every customer | `custom/agents/<agent>.md` |

A run sees `deployment` ∪ its customer ∪ its project. Narrow never leaks wide.

Decide the scope BEFORE granting: ask who should get the title. The safe
default is `project`. `client` and `deployment` affect other scopes than the
one you are in, so they need an explicit instruction —
`authorized_by="user"` (or `"planner"`) plus `authorization_note` — otherwise
ask the human first.

```
# project only (default)
tool_registry___system_request(tool_name="knowledge_base___grant_access",
    arguments={"agent_id": "project_manager_agent", "scope": "project",
               "titles": ["meadows-2008"]})

# whole customer — only when the user ordered it
tool_registry___system_request(tool_name="knowledge_base___grant_access",
    arguments={"agent_id": "project_manager_agent", "scope": "client",
               "titles": ["meadows-2008"], "authorized_by": "user",
               "authorization_note": "User: all Acme projects use this book"})

# every customer — only when the user ordered it
tool_registry___system_request(tool_name="knowledge_base___grant_access",
    arguments={"agent_id": "project_manager_agent", "scope": "deployment",
               "titles": ["meadows-2008"], "authorized_by": "user",
               "authorization_note": "User: make it generally available"})
```

`grant_access` writes both halves of the decision: the store row (what the
agent may read now) and the manifest at that level (the durable declaration).
The manifest merge keeps a hand-written description, body and other
`x-penta` keys — it only touches `knowledge_base`.

`access_matrix` is the same table, read: every grant per agent, per scope, the
database path and where each scope is declared on disk.

`assign_to_agent` still works and writes only the store row.

Agent YAML:
```yaml
x-penta:
  knowledge_base:
    titles: ["meadows-2008"]
    categories: ["systems-thinking"]
```

## Discovery

Planner sets `requires_knowledge_discovery: true` → discovery node runs.

Workflow: scout (search corpus) → researcher (find PDFs) → curator (chunk & assign).