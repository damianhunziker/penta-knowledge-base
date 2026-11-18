# Knowledge Base plugin — instructions

This plugin owns the knowledge corpus (books, academic literature). Its rules apply while it is active.

- **The corpus is the source of truth.** Titles, chunks, and references live in Qdrant + SQLite — not in markdown files.
- **Load the `knowledge-base` skill before the first knowledge action** in a session (`tool_registry___system_request(tool_name="skills___skill_loader", arguments={"skill_name": "knowledge-base"})`). It has the tool map, kb_ref grammar, and ingestion workflow.
- **One invitation per chunk.** Every rendered chunk ends with one `↳ read section` line. The read response offers broader resolutions.
- **License-first for chunking.** Never chunk a document without verifying the license (CC-BY, CC-BY-SA, MIT, public domain, or purchased). Unknown license = ask the user.
- **Background chunking.** `chunk_document` with `background=True` is the default; large documents are never chunked synchronously.
- **Assignments are training memory.** When a title is assigned to an agent, the agent receives it at run start and can read deeper on demand.
- **The planner decides discovery.** `requires_knowledge_discovery: true/false` is a planner decision. The planner also suggests `assigned_titles`.
- Ingest from vault roots: `tool_registry___system_request(tool_name="knowledge_base___ingest_from_root", arguments={"root_ref": "project-vault:Knowledge/book.md", "corpus": …})`. Markdown frontmatter provides metadata.
- **Injection is tuned in `plugin.env`.** Copy `plugin.env.example`: `KNOWLEDGE_INJECT_AFTER_TOOLS` (default 13), `KNOWLEDGE_INJECT_CHUNKS` (3), `KNOWLEDGE_TASK_CHUNKS` (3). A task may override the start count with `knowledge_chunks`. See `settings.py`.
- **Injected blocks are tail-safe.** They go into `llm_input_messages` as a `runtime_reminder` HumanMessage, never into `messages` — they must not repeat in later turns.
- **Grants carry a scope.** `project` (default) · `client` (one customer) · `deployment` (every customer). A run sees deployment ∪ its customer ∪ its project — narrow never leaks wide. `client`/`deployment` need `authorized_by="user"` + a note, otherwise ask the human.
- **`grant_access` writes both halves**: the store row and the manifest at that level (merged, never replaced). `access_matrix` shows who may read what, and where.
- **Managing what exists:** `scripts/knowledge_manager.py` (or `/knowledge` in the chat) browses bases, books, bindings and namespaces, and edits them. It is the same write path as the tools — never a second implementation.
- **A base = a corpus.** An agent is given a base, not a chunk. A book whose chunks are frontmatter or bullets is marked `!` (a summary, not the book) and should not be bound.
