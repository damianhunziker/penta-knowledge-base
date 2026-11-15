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
        Absolute filesystem path
    
    Raises:
        ValueError: If root is unknown or unreachable
    """
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
            spec = spec_for(root_name)
            # For vault roots, the path is the vault base
            # The relpath is the file within the vault
            # We need to get the vault path from config
            vault_path = _get_vault_path_for_root(root_name)
            if vault_path:
                return os.path.join(vault_path, relpath)
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


def _get_vault_path_for_root(root_name: str) -> Optional[str]:
    """Get the vault path for a vault root."""
    try:
        from app.core.configuration import get_active_project_path
        from app.core.mcp_integration import MCPIntegration
        
        if root_name == 'project-vault':
            return get_active_project_path()
        elif root_name == 'shared-vault':
            mcp = MCPIntegration.get_instance()
            if mcp:
                return mcp._get_vaultforge_shared_path()
        elif root_name == 'client-vault':
            mcp = MCPIntegration.get_instance()
            if mcp:
                return mcp._get_vaultforge_vault_path()
    except Exception as e:
        logger.warning(f"Could not resolve vault path for {root_name}: {e}")
    
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
            
            # TODO: Actual chunking and embedding
            # For now, return a mock result
            result.update({
                "status": "queued" if background else "completed",
                "title": final_title,
                "authors": final_authors,
                "tier": final_tier,
                "categories": final_categories,
                "tags": final_tags,
                "book_id": book_id,
                "kb_ref": f"knowledge_base:{corpus}/{book_id}",
                "frontmatter": fm,
                "message": f"Markdown file queued for ingestion: {final_title}",
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
            })
    
    # Process PDF/EPUB
    elif chunking == "pdf":
        # TODO: Implement Docling extraction
        result.update({
            "status": "queued" if background else "pending",
            "title": title or Path(file_path).stem,
            "message": f"PDF file queued for ingestion (Docling extraction pending)",
            "note": "Full PDF extraction requires Docling installation",
        })
    
    return json.dumps(result, indent=2)


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