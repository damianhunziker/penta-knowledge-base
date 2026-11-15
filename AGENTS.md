# Knowledge Base plugin — instructions

This plugin owns the knowledge corpus (books, academic literature). Its rules apply while it is active.

- **The corpus is the source of truth.** Titles, chunks, and references live in Qdrant + SQLite — not in markdown files.
- **Load the `knowledge-base` skill before the first knowledge action** in a session (`skills___skill_loader(skill_name="knowledge-base")`). It has the tool map, kb_ref grammar, and ingestion workflow.
- **One invitation per chunk.** Every rendered chunk ends with one `↳ read section` line. The read response offers broader resolutions.
- **License-first for chunking.** Never chunk a document without verifying the license (CC-BY, CC-BY-SA, MIT, public domain, or purchased). Unknown license = ask the user.
- **Background chunking.** `chunk_document` with `background=True` is the default; large documents are never chunked synchronously.
- **Assignments are training memory.** When a title is assigned to an agent, the agent receives it at run start and can read deeper on demand.
- **The planner decides discovery.** `requires_knowledge_discovery: true/false` is a planner decision. The planner also suggests `assigned_titles`.
- Ingest from vault roots: `knowledge_base___ingest_from_root("project-vault:Knowledge/book.md", corpus)`. Markdown frontmatter provides metadata.- **Injection is tuned in `plugin.env`.** Copy `plugin.env.example`: `KNOWLEDGE_INJECT_AFTER_TOOLS` (default 13), `KNOWLEDGE_INJECT_CHUNKS` (3), `KNOWLEDGE_TASK_CHUNKS` (3). A task may override the start count with `knowledge_chunks`. See `settings.py`.
- **Injected blocks are tail-safe.** They go into `llm_input_messages` as a `runtime_reminder` HumanMessage, never into `messages` — they must not repeat in later turns.
