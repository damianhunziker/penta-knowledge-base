#!/usr/bin/env python3
"""
Knowledge Base File Ingestion — ingest MD and PDF files from vault roots.

Supports:
- project-vault, shared-vault, client-vault, host-workspace
- Markdown files with YAML frontmatter (title, authors, tier, categories)
- PDF files (via Docling when available, or placeholder)
- Batch ingestion from directories
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── FRONTMATTER PARSING ──────────────────────────────────────────────────────

# Regex to match YAML frontmatter in markdown files
_FRONTMATTER_RE = re.compile(
    r'^---\s*\n(.*?)\n---\s*\n',
    re.DOTALL
)


def parse_frontmatter(content: str) -> Dict[str, Any]:
    """
    Parse YAML frontmatter from markdown content.
    
    Returns:
        Dict with frontmatter fields, or empty dict if none found.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}
    
    frontmatter_text = match.group(1)
    result = {}
    
    # Simple YAML-like parsing (no external deps)
    for line in frontmatter_text.split('\n'):
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # Remove quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        # Parse lists (simple: [a, b, c])
        if value.startswith('[') and value.endswith(']'):
            items = value[1:-1].split(',')
            value = [item.strip().strip('"\'') for item in items if item.strip()]
        
        # Parse numbers
        elif value.isdigit():
            value = int(value)
        elif re.match(r'^\d+\.\d+$', value):
            value = float(value)
        
        result[key] = value
    
    return result


def extract_title_from_markdown(content: str, path: str) -> str:
    """Extract title from markdown: frontmatter > first H1 > filename."""
    # Try frontmatter
    fm = parse_frontmatter(content)
    if fm.get('title'):
        return fm['title']
    
    # Try first H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Fall back to filename
    return Path(path).stem


# ── ROOT RESOLUTION ───────────────────────────────────────────────────────────

def resolve_root_path(root_ref: str) -> str:
    """
    Resolve a root reference to an actual filesystem path.
    
    Args:
        root_ref: Either:
                  - A full ref like "project-vault:Knowledge/book.md"
                  - Just the root name "project-vault"
                  - A direct filesystem path (no colon, or starts with /)
    
    Returns:
        Absolute filesystem path, translated to the host path when the
        container path does not exist here.
    
    Raises:
        ValueError: If root is unknown or unreachable
    """
    return _host_path(_resolve_root_path(root_ref))


def _resolve_root_path(root_ref: str) -> str:
    # Handle direct filesystem paths (no colon or starts with /)
    if root_ref.startswith('/') or (':' not in root_ref and os.path.isabs(root_ref)):
        return root_ref
    
    # Handle Windows paths (e.g., C:\path)
    if len(root_ref) > 2 and root_ref[1] == ':' and root_ref[2] in ('\\', '/'):
        return root_ref
    
    try:
        from app.core.roots import parse, spec_for, RootUnknown, RootNotReachable
    except ImportError:
        # Fallback for standalone testing
        return _resolve_root_path_fallback(root_ref)
    
    try:
        # Check if it's a full ref with path
        if ':' in root_ref:
            root_name, relpath = parse(root_ref)
            spec_for(root_name)  # raises RootUnknown for a typo in the root
            vault_path = _get_vault_path_for_root(root_name)
            if vault_path:
                rel = relpath.replace("/", os.sep)
                # A caller may write the path relative to the customer vault
                # ("Projects/<p>/Knowledge/x.md") although project-vault is
                # rooted at the project already. Dropping the duplicated
                # prefix keeps both notations working instead of producing
                # "…/Projects/<p>/Projects/<p>/…".
                project_base = os.path.basename(vault_path)
                dup = os.path.join("Projects", project_base) + os.sep
                if rel.startswith(dup):
                    rel = rel[len(dup):]
                resolved = os.path.join(vault_path, rel)
                if os.path.exists(resolved):
                    return resolved
                found = _probe_vault_for(relpath, vault_path)
                if found:
                    return found
                return resolved
            # Fallback: use the spec path if available
            if spec.path:
                return os.path.join(spec.path, relpath)
        else:
            # Just the root name
            spec = spec_for(root_ref)
            return _get_vault_path_for_root(root_ref) or spec.path
    except (RootUnknown, RootNotReachable) as e:
        raise ValueError(f"Cannot resolve root: {e}")

    raise ValueError(f"Cannot resolve root: {root_ref}")


def _probe_vault_for(relpath: str, vault_path: str) -> Optional[str]:
    """Find ``relpath`` inside ``vault_path`` (also one project level deeper).

    Handles the three notations a caller may use, all relative to different
    roots:

    * ``Knowledge/book.md``                          → project root
    * ``Projects/<p>/Knowledge/book.md``             → customer vault
    * ``Projects/<p>/Knowledge/book.md`` with a
      project-rooted base                            → prefix is stripped first

    The search is deterministic: an exact match against the base wins, then
    project-level candidates are tried in sorted order, so the same input
    always resolves to the same file.
    """
    rel = relpath.replace("/", os.sep)
    try:
        direct = os.path.join(vault_path, rel)
        if os.path.exists(direct):
            return direct

        # Candidate suffixes of the relpath: a caller may have stacked
        # segments ("Projects/<customer>/Projects/<project>/Knowledge/x.md"
        # although the real tree is "<customer>/Projects/<project>/…"). Every
        # path is tried from each separator onwards, longest first, so the
        # most specific match wins.
        suffixes = [rel]
        parts = rel.split(os.sep)
        for i in range(1, len(parts)):
            suffixes.append(os.sep.join(parts[i:]))

        for suffix in suffixes:
            candidates = [os.path.join(vault_path, suffix)]
            for customer in sorted(os.listdir(vault_path)):
                projects_dir = os.path.join(vault_path, customer, "Projects")
                if not os.path.isdir(projects_dir):
                    continue
                for project in sorted(os.listdir(projects_dir)):
                    candidates.append(os.path.join(projects_dir, project, suffix))
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
    except OSError:
        return None
    return None


def _host_path(container_path: str) -> str:
    """Translate a container-side vault path to the host path when needed.

    Everything inside the app container sees ``/vault/...``. When a tool runs
    on the host (a plugin speaking to Qdrant directly), that path does not
    exist — ``VAULT_HOST_ROOT`` names the directory the container mounts at
    ``VAULT_CONTAINER_ROOT``. When the path already exists, it is returned
    unchanged, so both deployment shapes work.
    """
    if not container_path:
        return container_path
    if os.path.exists(container_path):
        return container_path
    try:
        container_root = os.getenv("VAULT_CONTAINER_ROOT", "/vault").rstrip("/")
        host_root = os.getenv("VAULT_HOST_ROOT", "").rstrip("/")
        if host_root and container_path.startswith(container_root + "/"):
            mapped = host_root + container_path[len(container_root):]
            if os.path.exists(mapped):
                return mapped
    except Exception:  # noqa: BLE001 - path mapping must not raise
        pass
    return container_path


def _resolve_root_path_fallback(root_ref: str) -> str:
    """Fallback root resolution for standalone testing."""
    # Map root names to common paths
    root_map = {
        'project-vault': os.getenv('PROJECT_VAULT_PATH', ''),
        'shared-vault': os.getenv('SHARED_VAULT_PATH', ''),
        'client-vault': os.getenv('CLIENT_VAULT_PATH', ''),
        'host-workspace': '/host-workspace',
        'tmp': '/tmp',
    }
    
    if ':' in root_ref and not root_ref.startswith('/'):
        root_name, relpath = root_ref.split(':', 1)
        base = root_map.get(root_name, '')
        if base:
            return os.path.join(base, relpath)
    
    return root_map.get(root_ref, root_ref)


def _active_vault_path() -> Optional[str]:
    """Absolute path of the active client vault (``None`` when unknown).

    Resolution order, most reliable first:

    1. ``ACTIVE_VAULT_PATH`` from the run's namespace (set by
       ``set_namespace``) — an absolute path, used as-is;
    2. the current session context (customer + project) from the project DB,
       joined with the vault base;
    3. the customer vault root when only the customer is known.

    Everything is guarded: an unavailable helper means "unknown", not a crash.
    """
    # 1) absolute active vault path
    try:
        from app.core.global_state import get_active_vault_path

        active = get_active_vault_path()
        if active and os.path.isabs(active) and os.path.isdir(active):
            return active
    except Exception:  # noqa: BLE001 - helper optional
        pass

    # 2) session context (customer + project) under the vault base
    try:
        from app.core.configuration import VAULTS_BASE_PATH, get_current_session_id
        from app.customer_project_db import customer_project_db

        session_id = get_current_session_id()
        if session_id and session_id not in ("default", "global"):
            context = customer_project_db.get_current_context(session_id) or {}
            customer = context.get("customer")
            project = context.get("project")
            if customer and project:
                for candidate in (
                    os.path.join(VAULTS_BASE_PATH, customer, "Projects", project),
                    os.path.join(VAULTS_BASE_PATH, customer),
                ):
                    if os.path.isdir(candidate):
                        return candidate
    except Exception:  # noqa: BLE001
        pass

    return None


def _project_vault_base() -> Optional[str]:
    """Directory that ``project-vault:`` is rooted at.

    ``project-vault:`` is the *project* tree (``…/<customer>/Projects/<p>``),
    not the customer vault above it. Callers (vaultforge, the planner) write
    ``Projects/<p>/Knowledge/x.md`` relative to the customer vault, so a naive
    join produced ``…/Projects/<p>/Projects/<p>/…``.

    Resolution: the active project directory when a session knows it, else the
    customer vault (where the probe below can still find the file).
    """
    try:
        from app.core.configuration import VAULTS_BASE_PATH, get_current_session_id
        from app.customer_project_db import customer_project_db

        session_id = get_current_session_id()
        if session_id and session_id not in ("default", "global"):
            context = customer_project_db.get_current_context(session_id) or {}
            customer = context.get("customer")
            project = context.get("project")
            if customer and project:
                candidate = os.path.join(
                    VAULTS_BASE_PATH, customer, "Projects", project
                )
                if os.path.isdir(candidate):
                    return candidate
    except Exception:  # noqa: BLE001 - context optional
        pass

    active = _active_vault_path()
    return active


def _get_vault_path_for_root(root_name: str) -> Optional[str]:
    """Base directory of a vault root (``None`` when it cannot be resolved).

    ``project-vault``/``client-vault``/: the project tree when a session knows
    it, else the client vault (the probe in :func:`_resolve_root_path` then
    searches the project level). ``shared-vault`` resolves through the
    configured shared base.
    """
    name = str(root_name or "").strip().rstrip(":")
    try:
        if name in ("client-vault", "project-vault", "vault"):
            project_base = _project_vault_base()
            if project_base:
                return project_base
            from app.core.configuration import VAULTS_BASE_PATH

            return VAULTS_BASE_PATH or None

        if name == "shared-vault":
            from app.core.configuration import SHARED_VAULT_BASE_PATH

            return SHARED_VAULT_BASE_PATH or None

        if name == "host-workspace":
            return "/host-workspace"
        if name == "tmp":
            return "/tmp"
    except Exception as exc:  # noqa: BLE001 - resolution must not crash a run
        logger.warning("Could not resolve vault path for %s: %s", name, exc)

    return None


# ── INGESTION TOOLS ───────────────────────────────────────────────────────────

def knowledge_base___discover_files(
    root: str,
    pattern: str = "**/*.md",
    max_files: int = 100,
) -> str:
    """
    Discover files in a vault root matching a pattern.
    
    Args:
        root: Root name or full ref (e.g., "project-vault" or "project-vault:Knowledge")
        pattern: Glob pattern (default: "**/*.md" for all markdown files)
        max_files: Maximum files to return
        
    Returns:
        JSON string with discovered files
    """
    try:
        base_path = resolve_root_path(root)
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    
    if not base_path or not os.path.isdir(base_path):
        return json.dumps({
            "root": root,
            "resolved_path": base_path,
            "error": "Path does not exist or is not a directory",
            "files": [],
        }, indent=2)
    
    files = []
    base = Path(base_path)
    
    try:
        for path in base.glob(pattern):
            if len(files) >= max_files:
                break
            
            rel_path = str(path.relative_to(base))
            ext = path.suffix.lower()
            
            files.append({
                "path": rel_path,
                "root_ref": f"{root}:{rel_path}",
                "extension": ext,
                "size": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
            })
    except Exception as e:
        return json.dumps({
            "root": root,
            "resolved_path": base_path,
            "error": str(e),
            "files": [],
        }, indent=2)
    
    return json.dumps({
        "root": root,
        "resolved_path": base_path,
        "pattern": pattern,
        "files": files,
        "total": len(files),
        "truncated": len(files) >= max_files,
    }, indent=2)


def knowledge_base___ingest_from_root(
    root_ref: str,
    corpus: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    tier: str = "standard",
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    chunking: str = "auto",  # auto, markdown, pdf, skip
    background: bool = True,
) -> str:
    """
    Ingest a file from a vault root into the knowledge base.
    
    Supports:
    - .md files: Markdown with optional YAML frontmatter
    - .pdf files: PDF extraction via Docling
    - .epub files: EPUB extraction via Docling
    
    Args:
        root_ref: Full root reference (e.g., "project-vault:Knowledge/book.md")
        corpus: Corpus to add the content to
        title: Override title (extracted from file if not provided)
        authors: Override authors (extracted from frontmatter if not provided)
        tier: Content tier (foundational, standard, specialist, task)
        categories: Categories for the content
        tags: Tags for the content
        chunking: Chunking strategy (auto, markdown, pdf, skip)
        background: Run chunking in background
        
    Returns:
        JSON string with ingestion status
    """
    import uuid
    
    try:
        file_path = resolve_root_path(root_ref)
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    
    if not file_path or not os.path.isfile(file_path):
        return json.dumps({
            "error": f"File not found: {root_ref}",
            "resolved_path": file_path,
        }, indent=2)
    
    # Determine file type and extraction method
    ext = Path(file_path).suffix.lower()
    
    if chunking == "auto":
        if ext == '.md':
            chunking = "markdown"
        elif ext in ('.pdf', '.epub'):
            chunking = "pdf"
        else:
            chunking = "skip"
    
    # Read and process the file
    result = {
        "root_ref": root_ref,
        "resolved_path": file_path,
        "corpus": corpus,
        "file_type": ext,
        "chunking": chunking,
        "job_id": str(uuid.uuid4())[:8],
    }
    
    if chunking == "skip":
        return json.dumps({
            **result,
            "status": "skipped",
            "message": f"File type {ext} not supported for chunking",
        }, indent=2)
    
    # Process markdown
    if chunking == "markdown":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from frontmatter
            fm = parse_frontmatter(content)
            
            # Use provided values or fall back to frontmatter
            final_title = title or fm.get('title') or extract_title_from_markdown(content, file_path)
            final_authors = authors or fm.get('authors', [])
            if isinstance(final_authors, str):
                final_authors = [final_authors]
            final_categories = categories or fm.get('categories', [])
            if isinstance(final_categories, str):
                final_categories = [final_categories]
            final_tags = tags or fm.get('tags', [])
            if isinstance(final_tags, str):
                final_tags = [final_tags]
            final_tier = tier or fm.get('tier', 'standard')
            
            # Generate book_id from title
            book_id = re.sub(r'[^a-z0-9]+', '-', final_title.lower()).strip('-')
            if not book_id:
                book_id = str(uuid.uuid4())[:8]
            
            # Actually chunk + embed + upsert into Qdrant.
            from langgraph_crew_knowledge_base.indexer import index_markdown
            from langgraph_crew_knowledge_base.tools import _qdrant

            client = _qdrant()
            if client is None:
                raise RuntimeError(
                    "Qdrant unreachable - cannot index the document"
                )

            indexed = index_markdown(
                client,
                file_path,
                book_id=book_id,
                title=final_title,
                corpus=corpus,
                authors=final_authors,
                year=fm.get("year"),
                tier=final_tier,
                categories=final_categories,
                tags=final_tags,
                source_ref=root_ref,
            )

            result.update({
                "status": indexed.get("status", "indexed"),
                "title": final_title,
                "authors": final_authors,
                "tier": final_tier,
                "categories": final_categories,
                "tags": final_tags,
                "book_id": book_id,
                "kb_ref": f"knowledge_base:{corpus}/{book_id}",
                "chunks": indexed.get("chunks", 0),
                "sections": indexed.get("sections", 0),
                "collection": indexed.get("collection", ""),
                "frontmatter": fm,
                "message": (
                    f"Indexed {indexed.get('chunks', 0)} chunk(s) from "
                    f"{final_title} into corpus '{corpus}'"
                ),
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
            })
    
    # Process PDF/EPUB: extract to markdown, then the SAME indexing path the
    # markdown branch uses. There is no queue: a "queued" job with no worker
    # looked like a slow ingest and silently never produced a chunk.
    elif chunking in ("pdf", "epub"):
        try:
            from langgraph_crew_knowledge_base.extract import (
                PdfExtractionError,
                to_markdown,
            )

            final_title = title or Path(file_path).stem
            final_authors = list(authors or [])
            extracted = to_markdown(file_path)
            markdown = extracted["markdown"]

            book_id = re.sub(r"[^a-z0-9]+", "-", final_title.lower()).strip("-")
            if not book_id:
                book_id = str(uuid.uuid4())[:8]

            indexed = _index_text(
                markdown,
                book_id=book_id,
                title=final_title,
                corpus=corpus,
                authors=final_authors,
                year=None,
                tier=tier,
                categories=list(categories or []),
                tags=list(tags or []),
                source_ref=root_ref,
            )

            result.update({
                "status": indexed.get("status", "indexed"),
                "title": final_title,
                "authors": final_authors,
                "tier": tier,
                "categories": list(categories or []),
                "tags": list(tags or []),
                "book_id": book_id,
                "kb_ref": f"knowledge_base:{corpus}/{book_id}",
                "chunks": indexed.get("chunks", 0),
                "sections": indexed.get("sections", 0),
                "collection": indexed.get("collection", ""),
                "extraction": extracted["kind"],
                "message": (
                    f"Indexed {indexed.get('chunks', 0)} chunk(s) from "
                    f"{final_title} ({extracted['kind']}) into corpus '{corpus}'"
                ),
            })
        except PdfExtractionError as exc:
            result.update({
                "status": "error",
                "error": str(exc),
                "message": (
                    f"{Path(file_path).name}: {exc}"
                ),
            })
        except Exception as e:  # noqa: BLE001 - report, never crash the caller
            result.update({"status": "error", "error": str(e)})

    return json.dumps(result, indent=2)


def _index_text(
    markdown: str,
    *,
    book_id: str,
    title: str,
    corpus: str,
    authors: List[str],
    year: Optional[int],
    tier: str,
    categories: List[str],
    tags: List[str],
    source_ref: str,
) -> Dict[str, Any]:
    """Chunk + embed + upsert already-extracted text.

    Shared by the markdown and the PDF/EPUB branch: one indexing path, so a book
    from a PDF and a book from markdown cannot end up in different shapes.
    """
    from langgraph_crew_knowledge_base.indexer import index_text
    from langgraph_crew_knowledge_base.tools import _qdrant

    client = _qdrant()
    if client is None:
        raise RuntimeError("Qdrant unreachable - cannot index the document")

    return index_text(
        client,
        markdown,
        book_id=book_id,
        title=title,
        corpus=corpus,
        authors=authors,
        year=year,
        tier=tier,
        categories=categories,
        tags=tags,
        source_ref=source_ref,
    )


def knowledge_base___ingest_directory(
    root: str,
    directory: str = "",
    corpus: str = "auto",  # auto, or specific corpus name
    pattern: str = "**/*.{md,pdf,epub}",
    max_files: int = 50,
    background: bool = True,
    recursive: bool = True,
) -> str:
    """
    Batch ingest files from a directory in a vault root.
    
    Args:
        root: Root name (e.g., "project-vault")
        directory: Subdirectory within root (empty for root)
        corpus: Corpus name, or "auto" to derive from directory
        pattern: Glob pattern for files
        max_files: Maximum files to process
        background: Run ingestions in background
        recursive: Search recursively
        
    Returns:
        JSON string with batch ingestion status
    """
    # Build the root ref
    if directory:
        root_ref = f"{root}:{directory}"
    else:
        root_ref = root
    
    # Discover files
    discover_result = json.loads(knowledge_base___discover_files(root_ref, pattern, max_files))
    
    if "error" in discover_result:
        return json.dumps(discover_result, indent=2)
    
    files = discover_result.get("files", [])
    
    if not files:
        return json.dumps({
            "root": root,
            "directory": directory,
            "pattern": pattern,
            "files_found": 0,
            "message": "No files found matching pattern",
        }, indent=2)
    
    # Determine corpus name
    if corpus == "auto":
        # Derive from directory name
        corpus = Path(directory).name if directory else root.replace('-', '_')
    
    # Queue files for ingestion
    jobs = []
    for file_info in files:
        file_root_ref = file_info["root_ref"]
        ext = file_info["extension"]
        
        # Skip unsupported files
        if ext not in ('.md', '.pdf', '.epub'):
            continue
        
        ingest_result = json.loads(
            knowledge_base___ingest_from_root(
                file_root_ref,
                corpus,
                chunking="auto",
                background=background,
            )
        )
        
        jobs.append({
            "file": file_root_ref,
            "job_id": ingest_result.get("job_id"),
            "status": ingest_result.get("status"),
            "title": ingest_result.get("title"),
        })
    
    return json.dumps({
        "root": root,
        "directory": directory,
        "corpus": corpus,
        "pattern": pattern,
        "files_found": len(files),
        "jobs_queued": len(jobs),
        "jobs": jobs,
        "message": f"Queued {len(jobs)} files for ingestion into corpus '{corpus}'",
    }, indent=2)


def knowledge_base___list_vault_knowledge(
    root: str = "project-vault",
) -> str:
    """
    List potential knowledge files in a vault root.
    
    Scans for .md, .pdf, .epub files in the Knowledge/ subdirectory.
    
    Args:
        root: Root name (default: project-vault)
        
    Returns:
        JSON string with categorized file list
    """
    # Check common knowledge directories
    knowledge_dirs = [
        "Knowledge",
        "knowledge",
        "Library",
        "library",
        "Reference",
        "reference",
    ]
    
    all_files = {
        "markdown": [],
        "pdf": [],
        "epub": [],
        "total": 0,
    }
    
    for subdir in knowledge_dirs:
        result = json.loads(
            knowledge_base___discover_files(
                f"{root}:{subdir}",
                pattern="**/*.{md,pdf,epub}",
                max_files=100,
            )
        )
        
        for file_info in result.get("files", []):
            ext = file_info["extension"]
            if ext == ".md":
                all_files["markdown"].append(file_info)
            elif ext == ".pdf":
                all_files["pdf"].append(file_info)
            elif ext == ".epub":
                all_files["epub"].append(file_info)
    
    all_files["total"] = (
        len(all_files["markdown"]) + 
        len(all_files["pdf"]) + 
        len(all_files["epub"])
    )
    
    return json.dumps({
        "root": root,
        "files": all_files,
        "summary": {
            "markdown": len(all_files["markdown"]),
            "pdf": len(all_files["pdf"]),
            "epub": len(all_files["epub"]),
            "total": all_files["total"],
        },
    }, indent=2)


# ── EXPORTS ───────────────────────────────────────────────────────────────────

__all__ = [
    # Discovery
    "knowledge_base___discover_files",
    "knowledge_base___list_vault_knowledge",
    # Ingestion
    "knowledge_base___ingest_from_root",
    "knowledge_base___ingest_directory",
    # Helpers
    "parse_frontmatter",
    "extract_title_from_markdown",
    "resolve_root_path",
]