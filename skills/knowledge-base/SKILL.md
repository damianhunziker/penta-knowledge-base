# Knowledge Base

Use the knowledge base for domain knowledge, reference material, and training memory.

## Quick Start

```
# Search corpus
knowledge_base___search("systems thinking")

# Read section
knowledge_base___read("knowledge_base:academic/meadows-2008/ch3#s:1.2")

# Ingest from vault
knowledge_base___ingest_from_root("project-vault:Knowledge/book.md", "academic")
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
| `assign_to_agent` | Give agent access to titles |

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

## Assignment

Assign titles to agents as training memory:
```
knowledge_base___assign_to_agent("developer", book_ids=["meadows-2008"])
```

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