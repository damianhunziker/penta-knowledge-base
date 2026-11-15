# Knowledge Base plugin — instructions

This plugin provides the knowledge base for PENTA. Its rules apply while it is
active.

## Invariants

- **The corpus is the source of truth.** Titles, chunks, and references live in
  the Qdrant collection + SQLite catalog — not in markdown files.
- **One invitation per chunk.** Every rendered chunk ends with ONE `↳ read section`
  line. The read response offers broader resolutions (chapter, neighbours, book).
- **License-first for chunking.** Never chunk a document without verifying the
  license (CC-BY, CC-BY-SA, MIT, public domain, or purchased). Unknown = ask.
- **Background chunking.** `chunk_document` with `background=True` is the default;
  large documents are never chunked synchronously.
- **Assignments are training memory.** When a title is assigned to an agent, the
  agent receives it at run start (catalog + hints) and can read deeper on demand.
- **The planner decides.** `requires_knowledge_discovery: true/false` is a planner
  decision, not an agent's. The planner also suggests `assigned_titles`.

## Load the skill

`skills___skill_loader(skill_name="knowledge-base")` — before the first knowledge
base action. It has the tool map, the kb_ref grammar, and the discovery workflow.

## Tools

| Tool | Purpose |
|------|---------|
| `knowledge_base___search` | Search the corpus by query |
| `knowledge_base___read` | Read a section/chapter/book by kb_ref |
| `knowledge_base___list` | List books by corpus/tier/category |
| `knowledge_base___describe` | Get details on a specific book |
| `knowledge_base___recommend_titles` | Get recommendations by topic |
| `knowledge_base___chunk_document` | Chunk a PDF/EPUB from URL |
| `knowledge_base___chunk_status` | Check chunking job status |
| `knowledge_base___assign_to_agent` | Assign titles to an agent |
| `knowledge_base___list_assignments` | List agent→title mappings |
| `knowledge_base___discover_files` | Find files in vault roots |
| `knowledge_base___ingest_from_root` | Ingest MD/PDF from vault |
| `knowledge_base___ingest_directory` | Batch ingest directory |
| `knowledge_base___list_vault_knowledge` | Find Knowledge/ files |

## File ingestion from vaults

The knowledge base can ingest files from vault roots:

```
project-vault:Knowledge/book.md    → chunked into corpus
shared-vault:Reference/manual.pdf  → PDF extraction → chunks
```

**Markdown frontmatter** is parsed for metadata:
- `title`, `authors`, `tier`, `categories`, `tags`

**PDF files** use Docling for extraction (when installed).

Use `discover_files` to find files, `ingest_from_root` to process them.

## kb_ref grammar

```
knowledge_base:corpus/book-id                    # book level
knowledge_base:corpus/book-id/chapter-id         # chapter level
knowledge_base:corpus/book-id/chapter-id#s:sec   # section level
```

Example: `knowledge_base:academic/meadows-2008/ch3#s:1.2`

## Discovery workflow

1. **knowledge_scout** — searches existing corpus, recommends relevant titles
2. **knowledge_researcher** — researches new PDFs, verifies licenses
3. **knowledge_curator** — decides what to chunk, assigns to agents

The workflow is defined in `workflows/knowledge-discovery.yaml` and the agents
in `agents/*.md`. Changing the flow starts in the YAML, not in Python.

## Agent yaml assignment

```yaml
x-penta:
  knowledge_base:
    titles: ["meadows-2008"]
    categories: ["systems-thinking"]
    tags: ["foundational"]
    auto_discovery: true
    injection:
      cold: true
      warm: true
      budget_tokens: 1200
```

## Planner integration

The planner can set per-task:
- `requires_knowledge_discovery: true`
- `assigned_titles: ["meadows-2008"]`
- `assigned_categories: ["systems-thinking"]`