# Knowledge Base Plugin

Load skill: `skills___skill_loader(skill_name="knowledge-base")`

## Rules

- **One invitation per chunk.** Read response offers broader resolutions.
- **License-first.** Never chunk without verified license (CC-BY, MIT, purchased).
- **Background chunking.** Large documents chunk asynchronously.
- **Assignments are training memory.** Agents get assigned titles at run start.
- **Planner decides discovery.** `requires_knowledge_discovery: true/false` is a planner decision.

## Tools

| Tool | Purpose |
|------|---------|
| `search` | Query corpus |
| `read` | Get content by kb_ref |
| `list` | Browse books |
| `recommend_titles` | Suggestions by topic |
| `discover_files` | Find files in vault |
| `ingest_from_root` | Add MD/PDF from vault |
| `assign_to_agent` | Give agent titles |

## kb_ref

```
knowledge_base:corpus/book/chapter#s:section
```

## Ingestion

From vault roots:
```
knowledge_base___ingest_from_root("project-vault:Knowledge/book.md", "academic")
```

Markdown frontmatter: `title`, `authors`, `tier`, `categories`, `tags`

## Assignment

```yaml
x-penta:
  knowledge_base:
    titles: ["meadows-2008"]
    categories: ["systems-thinking"]
```