# Knowledge Base Plugin (penta-knowledge-base)

A shared knowledge corpus for PENTA with hierarchical chunking, BGE-M3 embeddings,
reasoning-driven injection, single-invitation rendering, MCP tools, and knowledge
discovery node.

## Installation

```bash
pip install -e ./plugins/knowledge-base
```

## What it provides

- **Provider kind**: `knowledge_base` and `knowledge_discovery`
- **Core seam**: `app/core/knowledge_seam.py`
- **MCP Tools**: search, read, chunk, assign
- **Discovery Node**: knowledge_discovery for the executor loop

## Key design decisions

1. **ONE invitation line per chunk** — the read tool offers broader resolutions
2. **Reasoning-driven injection** — late, delta-only, budget-capped
3. **BGE-M3 hybrid** — dense + sparse from one model
4. **Docling for PDF/EPUB** — proper heading hierarchy, tables, OCR
5. **Background chunking** — fire-and-forget document processing
6. **Agent assignment** — titles as "training memory" for agents

## Root syntax

```
knowledge_base:academic/meadows-2008/ch3#s:1.2
```

## Rendering contract

```
[kb:academic/meadows-2008/ch3#s:1.2 · p.118–119 · tier=foundational · score 0.81]
Meadows, D. H., *Thinking in Systems* (2008), "ch3.1.2"

> …chunk text…

↳ read section   knowledge_base___read(kb_ref="knowledge_base:academic/meadows-2008/ch3#s:1.2")
```

After `read()` is called, the response offers chapter/book/neighbours.

## Tools

| Tool | Purpose |
|------|---------|
| `knowledge_base___search` | Search the corpus by query |
| `knowledge_base___read` | Read a section/chapter/book by kb_ref |
| `knowledge_base___list` | List books by corpus/tier/category |
| `knowledge_base___describe` | Get details on a specific book |
| `knowledge_base___recommend_titles` | Get recommendations by topic |
| `knowledge_base___chunk_document` | Chunk a PDF/EPUB (background) |
| `knowledge_base___chunk_status` | Check chunking job status |
| `knowledge_base___assign_to_agent` | Assign titles to an agent |
| `knowledge_base___list_assignments` | List agent→title mappings |
| `knowledge_base___discover_files` | Find files in vault roots |
| `knowledge_base___ingest_from_root` | Ingest MD/PDF from vault |
| `knowledge_base___ingest_directory` | Batch ingest directory |
| `knowledge_base___list_vault_knowledge` | Find Knowledge/ files |

## File Ingestion

Ingest markdown and PDF files from vault roots:

```python
# Discover files
knowledge_base___discover_files("project-vault:Knowledge", pattern="**/*.md")

# Ingest single file (MD with frontmatter)
knowledge_base___ingest_from_root("project-vault:Knowledge/book.md", "academic")

# Batch ingest directory
knowledge_base___ingest_directory("project-vault", "Knowledge", "academic")

# Find all knowledge files
knowledge_base___list_vault_knowledge("project-vault")
```

**Markdown frontmatter support:**
```yaml
---
title: Systems Thinking Guide
authors: [Donella Meadows]
tier: foundational
categories: [systems-thinking]
tags: [leverage-points]
---

# Content...
```

## Discovery workflow

1. **knowledge_scout** — searches existing corpus, recommends relevant titles
2. **knowledge_researcher** — researches new PDFs, verifies licenses
3. **knowledge_curator** — decides what to chunk, assigns to agents

## Agent assignment (YAML)

```yaml
x-penta:
  knowledge_base:
    titles: ["meadows-2008"]
    categories: ["systems-thinking"]
    tags: ["foundational"]
    auto_discovery: true
```

## Planner integration

The planner can set per-task:
- `requires_knowledge_discovery: true`
- `assigned_titles: ["meadows-2008"]`
- `assigned_categories: ["systems-thinking"]`
- `knowledge_chunks: 5` — how many chunks this task gets at start

## Injection (plugin-owned configuration)

Two moments inject knowledge. Both are tuned in the plugin's own
`plugin.env` — copy `plugin.env.example` and edit it:

```bash
cp plugins/knowledge-base/plugin.env.example plugins/knowledge-base/plugin.env
```

Precedence: **process environment > `plugin.env` > `plugin.env.example` > default**.

| Variable | Default | Meaning |
|---|---|---|
| `KNOWLEDGE_INJECT_AFTER_TOOLS` | `13` | Inject a block after this many tool calls. `0` disables mid-run injection. |
| `KNOWLEDGE_INJECT_CHUNKS` | `3` | Chunks per mid-run block. |
| `KNOWLEDGE_TASK_CHUNKS` | `3` | Chunks when a task starts (cold injection). |
| `KNOWLEDGE_INJECT_MAX_PER_RUN` | `6` | Safety cap on mid-run injections per agent run. |
| `KNOWLEDGE_INJECT_MIN_SCORE` | `0.35` | Retrieval score below which a chunk is dropped. |
| `KNOWLEDGE_INJECT_BUDGET_TOKENS` | `900` | Token ceiling for one block. |
| `DISABLE_KNOWLEDGE_DISCOVERY` | `false` | Switch knowledge discovery off entirely. |

A per-task `knowledge_chunks` overrides `KNOWLEDGE_TASK_CHUNKS` for that task.

**Mid-run** (`pre_model` hook, `injection.py`): runs before every LLM call, so
it counts completed tool turns. After `KNOWLEDGE_INJECT_AFTER_TOOLS` turns it
appends a block to `llm_input_messages` — never to `messages`, so it does not
repeat in later turns. Chunks already injected in the run are skipped, and the
counter restarts after each injection.

**Task start** (knowledge discovery node): the count comes from the task's
`knowledge_chunks` when set, otherwise from `KNOWLEDGE_TASK_CHUNKS`. The
executor injects the result into the worker prompt exactly like the
tool-researcher report.

A malformed value never breaks a run — it logs a warning and falls back to the
default.

## Status

**Phase 0 + D1 complete**: Plugin skeleton, root, seam, renderer, tools, node, workflow, agents.
**Injection complete**: environment-driven mid-run + task-start injection.
