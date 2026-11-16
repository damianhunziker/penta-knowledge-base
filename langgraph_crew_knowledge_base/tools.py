#!/usr/bin/env python3
"""
Knowledge Base Tools — MCP tools for chunking, discovery, and assignment.

These are the tools exposed by the knowledge_base plugin for:
- Document chunking
- Corpus search
- Title recommendations
- Agent assignment
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── MOCK DATA (until Qdrant/SQLite are connected) ─────────────────────────────

_MOCK_CATALOG = {
    "corpora": {
        "academic": {
            "name": "Academic Literature",
            "books": {
                "meadows-2008": {
                    "title": "Thinking in Systems",
                    "authors": ["Meadows, Donella H."],
                    "year": 2008,
                    "tier": "foundational",
                    "categories": ["systems-thinking", "complexity"],
                    "chunks": 234,
                },
                "kahneman-2011": {
                    "title": "Thinking, Fast and Slow",
                    "authors": ["Kahneman, Daniel"],
                    "year": 2011,
                    "tier": "foundational",
                    "categories": ["decision-making", "cognitive-science"],
                    "chunks": 456,
                },
            },
        },
    },
    "total_books": 2,
    "total_chunks": 690,
}

_MOCK_ASSIGNMENTS: Dict[str, List[str]] = {}

_MOCK_CHUNKING_JOBS: Dict[str, Dict] = {}

#: Qdrant collection that holds the indexed knowledge chunks.
#: One collection, ``corpus``/``tier``/``book_id`` live in the payload — the
#: same shape deep-reader uses, so a book can be searched corpus-scoped.
KB_COLLECTION = os.getenv("KNOWLEDGE_COLLECTION", "knowledge_base_chunks")

#: Embedding model used for the corpus (dense, 1024 dims for BGE-M3).
KB_EMBED_MODEL = os.getenv("KNOWLEDGE_EMBED_MODEL", "BAAI/bge-m3")
KB_EMBED_DIM = int(os.getenv("KNOWLEDGE_EMBED_DIM", "1024"))

_qdrant_client = None
_embedder = None


def _qdrant_urls():
    """Candidate Qdrant URLs, most specific first.

    The same code runs on the host and inside the app container, and the
    correct address differs:

    * container  → ``QDRANT_URL`` (``http://qdrant:6333``, the compose service)
    * host       → ``QDRANT_HOST_URL`` (``http://127.0.0.1:16333``)

    Trying both (cheap, cached) keeps one plugin working in both shapes
    instead of reporting "connection refused" in whichever one is not first.
    """
    urls = []
    for value in (
        os.getenv("QDRANT_URL"),
        getattr(_settings(), "qdrant_url", ""),
        os.getenv("QDRANT_HOST_URL"),
        getattr(_settings(), "qdrant_host_url", ""),
        "http://127.0.0.1:6333",
    ):
        url = str(value or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _settings():
    try:
        from config import settings

        return settings
    except Exception:  # noqa: BLE001 - settings optional in tests
        return None


def _qdrant():
    """Lazy Qdrant client (``None`` when no reachable instance exists).

    The client is bound to the first URL that answers a health probe, so a
    later search call does not fail because the process guessed the wrong
    address.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # noqa: BLE001
        logger.debug("[KNOWLEDGE_BASE] qdrant-client missing: %s", exc)
        return None

    api_key = os.getenv("QDRANT_API_KEY") or getattr(_settings(), "qdrant_api_key", None)
    for url in _qdrant_urls():
        try:
            client = QdrantClient(
                url=url, api_key=api_key or None, timeout=5, check_compatibility=False
            )
            client.get_collections()  # probe
            _qdrant_client = client
            logger.debug("[KNOWLEDGE_BASE] Qdrant at %s", url)
            return client
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.debug("[KNOWLEDGE_BASE] Qdrant %s unreachable: %s", url, exc)
    return None


def _embedder_model():
    """Embedding backend for queries (same server as the indexer)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from langgraph_crew_knowledge_base.indexer import _embeddings

        _embedder = _embeddings()
        return _embedder
    except Exception as exc:  # noqa: BLE001 - fall back to keyword search
        logger.debug("[KNOWLEDGE_BASE] embedding backend unavailable: %s", exc)
        return None


def _collection_ready() -> bool:
    """True when the knowledge collection exists AND holds points."""
    client = _qdrant()
    if client is None:
        return False
    try:
        if not client.collection_exists(KB_COLLECTION):
            return False
        return (client.get_collection(KB_COLLECTION).points_count or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def _vector_search(
    query: str,
    top_k: int,
    corpora: Optional[List[str]] = None,
    tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dense vector search over the indexed corpus (``[]`` when unavailable)."""
    if not _collection_ready():
        return []
    model = _embedder_model()
    if model is None:
        return []
    client = _qdrant()
    try:
        vector = model.embed_query(query)
        filt = None
        conditions = []
        if corpora:
            conditions.append(
                {"key": "corpus", "match": {"any": list(corpora)}}
            )
        if tier:
            conditions.append({"key": "tier", "match": {"value": tier}})
        if conditions:
            from qdrant_client import models as qm

            filt = qm.Filter(must=[qm.FieldCondition(**c) for c in conditions])
        # Newer qdrant-client removed ``search()`` in favour of
        # ``query_points()``; support both so either version works.
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=KB_COLLECTION,
                query=vector,
                limit=max(1, int(top_k)),
                query_filter=filt,
                with_payload=True,
            )
            hits = list(response.points)
        else:
            hits = client.search(
                collection_name=KB_COLLECTION,
                query_vector=vector,
                limit=max(1, int(top_k)),
                query_filter=filt,
                with_payload=True,
            )
    except Exception as exc:  # noqa: BLE001 - retrieval is best effort
        logger.debug("[KNOWLEDGE_BASE] vector search failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        book_id = payload.get("book_id") or ""
        corpus = payload.get("corpus") or ""
        section = payload.get("section_path") or ""
        kb_ref = payload.get("kb_ref") or (
            f"knowledge_base:{corpus}/{book_id}"
            + (f"#s:{section}" if section else "")
        )
        results.append(
            {
                "kb_ref": kb_ref,
                "title": payload.get("title") or "",
                "authors": payload.get("authors") or [],
                "year": payload.get("year"),
                "tier": payload.get("tier") or "",
                "book_id": book_id,
                "corpus": corpus,
                "section_path": section,
                "page_from": payload.get("page_from"),
                "page_to": payload.get("page_to"),
                "text": payload.get("text") or "",
                "score": float(hit.score or 0.0),
            }
        )
    return results


def kb_status() -> Dict[str, Any]:
    """Honest state of the corpus (used by tools and the estimate bar)."""
    client = _qdrant()
    if client is None:
        return {"ready": False, "reason": "Qdrant unreachable", "chunks": 0}
    try:
        if not client.collection_exists(KB_COLLECTION):
            return {
                "ready": False,
                "reason": f"collection '{KB_COLLECTION}' does not exist",
                "chunks": 0,
            }
        info = client.get_collection(KB_COLLECTION)
        count = info.points_count or 0
        if count == 0:
            return {
                "ready": False,
                "reason": f"collection '{KB_COLLECTION}' is empty",
                "chunks": 0,
            }
        return {"ready": True, "reason": "", "chunks": count}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "reason": str(exc), "chunks": 0}


# ── TOOLS ─────────────────────────────────────────────────────────────────────

def knowledge_base___search(
    query: str,
    corpora: Optional[List[str]] = None,
    tier: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """
    Search the knowledge base for relevant chunks.
    
    Args:
        query: The search query
        corpora: List of corpora to search (default: all)
        tier: Filter by tier (foundational, standard, specialist, task)
        top_k: Maximum results to return
        
    Returns:
        JSON string with search results
    """
    # Real retrieval first: dense vector search over the indexed corpus.
    vector_hits = _vector_search(query, top_k, corpora, tier)
    if vector_hits:
        return json.dumps(
            {
                "query": query,
                "results": vector_hits[:top_k],
                "total": len(vector_hits),
                "source": "qdrant",
            },
            indent=2,
        )

    # Nothing indexed (or Qdrant unreachable): say so instead of inventing a
    # silent empty answer. The old keyword mock below still runs so a demo
    # catalog behaves as before, but the result is marked as not real.
    status = kb_status()
    logger.debug("[KNOWLEDGE_BASE] no vector hits (%s)", status.get("reason"))

    results = []
    
    # Mock: find matching books by title, author, or category
    query_lower = query.lower()
    
    # Keywords that map to categories
    category_keywords = {
        "systems thinking": "systems-thinking",
        "system": "systems-thinking",
        "complexity": "complexity",
        "decision": "decision-making",
        "cognitive": "cognitive-science",
        "thinking": None,  # matches multiple
    }
    
    for corpus_id, corpus_data in _MOCK_CATALOG["corpora"].items():
        if corpora and corpus_id not in corpora:
            continue
        for book_id, book_info in corpus_data["books"].items():
            if tier and book_info.get("tier") != tier:
                continue
            
            # Check title
            if query_lower in book_info["title"].lower():
                results.append({
                    "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                    "title": book_info["title"],
                    "authors": book_info["authors"],
                    "year": book_info["year"],
                    "tier": book_info["tier"],
                    "score": 0.90,
                    "chunk_count": book_info["chunks"],
                })
                continue
            
            # Check authors
            if any(query_lower in a.lower() for a in book_info["authors"]):
                results.append({
                    "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                    "title": book_info["title"],
                    "authors": book_info["authors"],
                    "year": book_info["year"],
                    "tier": book_info["tier"],
                    "score": 0.85,
                    "chunk_count": book_info["chunks"],
                })
                continue
            
            # Check categories
            for keyword, category in category_keywords.items():
                if keyword in query_lower and category in book_info.get("categories", []):
                    results.append({
                        "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                        "title": book_info["title"],
                        "authors": book_info["authors"],
                        "year": book_info["year"],
                        "tier": book_info["tier"],
                        "score": 0.75,
                        "chunk_count": book_info["chunks"],
                    })
                    break
    
    return json.dumps({
        "query": query,
        "results": results[:top_k],
        "total": len(results),
        "source": "mock-catalog",
        "warning": (
            "No indexed corpus: "
            + (status.get("reason") or "collection empty")
            + ". Ingest documents first (knowledge_base___ingest_from_root)."
        ),
    }, indent=2)


def knowledge_base___read(
    kb_ref: str,
    budget_tokens: Optional[int] = None,
    before: int = 0,
    after: int = 0,
) -> str:
    """
    Read a section/chapter/book from the knowledge base.
    
    Args:
        kb_ref: Knowledge base reference (e.g., "knowledge_base:academic/meadows-2008/ch3#s:1.2")
        budget_tokens: Maximum tokens to return
        before: Number of sections before to include
        after: Number of sections after to include
        
    Returns:
        Rendered text with invitations to broader resolutions
    """
    from langgraph_crew_knowledge_base.render import (
        derive_broader_refs,
        render_read_response,
        Reference,
    )
    
    # TODO: Implement actual retrieval from Qdrant
    # For now, return a placeholder
    
    # Parse the kb_ref
    if not kb_ref.startswith("knowledge_base:"):
        return f"Invalid kb_ref format: {kb_ref}"
    
    path = kb_ref.split(":", 1)[1]
    parts = path.split("/")
    
    if len(parts) < 2:
        return f"Invalid kb_ref: {kb_ref}"
    
    corpus = parts[0]
    book_id = parts[1]
    
    # Get book info from mock catalog
    book_info = _MOCK_CATALOG["corpora"].get(corpus, {}).get("books", {}).get(book_id, {})
    
    if not book_info:
        return f"Book not found: {kb_ref}"
    
    # Create a reference
    ref = Reference(
        kb_ref=kb_ref,
        title=book_info["title"],
        authors=tuple(book_info["authors"]),
        year=book_info.get("year"),
        page_from=None,
        page_to=None,
        section_path="",
        tier=book_info.get("tier", "standard"),
        book_id=book_id,
        corpus=corpus,
    )
    
    # Mock content
    text = f"[Content from {book_info['title']} would appear here...]\n\n"
    text += f"This is a placeholder for the actual content from {kb_ref}.\n"
    text += "The real implementation will fetch chunks from Qdrant.\n"
    
    return render_read_response(
        text=text,
        ref=ref,
        budget_used=100,
        budget_max=budget_tokens or 2500,
        truncated=False,
    )


def knowledge_base___list(
    corpus: Optional[str] = None,
    tier: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """
    List books in the knowledge base.
    
    Args:
        corpus: Filter by corpus
        tier: Filter by tier
        category: Filter by category
        
    Returns:
        JSON string with book list
    """
    books = []
    
    for corpus_id, corpus_data in _MOCK_CATALOG["corpora"].items():
        if corpus and corpus_id != corpus:
            continue
        for book_id, book_info in corpus_data["books"].items():
            if tier and book_info.get("tier") != tier:
                continue
            if category and category not in book_info.get("categories", []):
                continue
            books.append({
                "book_id": book_id,
                "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                "title": book_info["title"],
                "authors": book_info["authors"],
                "year": book_info["year"],
                "tier": book_info["tier"],
                "categories": book_info.get("categories", []),
                "chunk_count": book_info["chunks"],
            })
    
    return json.dumps({
        "corpus": corpus,
        "tier": tier,
        "category": category,
        "books": books,
        "total": len(books),
    }, indent=2)


def knowledge_base___describe(book_id: str) -> str:
    """
    Get details about a specific book.
    
    Args:
        book_id: The book identifier (e.g., "meadows-2008")
        
    Returns:
        JSON string with book details
    """
    for corpus_id, corpus_data in _MOCK_CATALOG["corpora"].items():
        if book_id in corpus_data["books"]:
            book_info = corpus_data["books"][book_id]
            return json.dumps({
                "book_id": book_id,
                "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                "title": book_info["title"],
                "authors": book_info["authors"],
                "year": book_info["year"],
                "tier": book_info["tier"],
                "categories": book_info.get("categories", []),
                "chunk_count": book_info["chunks"],
                "corpus": corpus_id,
            }, indent=2)
    
    return json.dumps({"error": f"Book not found: {book_id}"}, indent=2)


def knowledge_base___recommend_titles(
    topic: str,
    top_k: int = 5,
) -> str:
    """
    Get title recommendations based on a topic.
    
    Args:
        topic: The topic to search for
        top_k: Maximum recommendations to return
        
    Returns:
        JSON string with recommendations
    """
    # TODO: Implement actual semantic search + topic matching
    # For now, match on title, categories, and authors
    
    recommendations = []
    topic_lower = topic.lower()
    
    # Category mappings
    category_keywords = {
        "systems thinking": "systems-thinking",
        "system": "systems-thinking",
        "complexity": "complexity",
        "decision": "decision-making",
        "decision making": "decision-making",
        "cognitive": "cognitive-science",
        "cognitive science": "cognitive-science",
    }
    
    for corpus_id, corpus_data in _MOCK_CATALOG["corpora"].items():
        for book_id, book_info in corpus_data["books"].items():
            relevance = 0.0
            
            # Check title
            if topic_lower in book_info["title"].lower():
                relevance = 0.9
            
            # Check categories
            for keyword, category in category_keywords.items():
                if keyword in topic_lower and category in book_info.get("categories", []):
                    relevance = max(relevance, 0.7)
            
            # Check authors
            if any(topic_lower in a.lower() for a in book_info["authors"]):
                relevance = max(relevance, 0.5)
            
            if relevance > 0:
                recommendations.append({
                    "book_id": book_id,
                    "kb_ref": f"knowledge_base:{corpus_id}/{book_id}",
                    "title": book_info["title"],
                    "authors": book_info["authors"],
                    "relevance": relevance,
                    "tier": book_info["tier"],
                    "categories": book_info.get("categories", []),
                })
    
    # Sort by relevance
    recommendations.sort(key=lambda x: x["relevance"], reverse=True)
    
    return json.dumps({
        "topic": topic,
        "recommendations": recommendations[:top_k],
        "total": len(recommendations),
    }, indent=2)


def knowledge_base___chunk_document(
    source_path: str,
    corpus: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    tier: str = "standard",
    categories: Optional[List[str]] = None,
    background: bool = True,
) -> str:
    """
    Chunk a document (PDF/EPUB) into the knowledge base.
    
    Args:
        source_path: Path or URL to the document
        corpus: Corpus to add to
        title: Document title (extracted if not provided)
        authors: Document authors (extracted if not provided)
        tier: Content tier (foundational, standard, specialist, task)
        categories: Categories to assign
        background: Run in background (default: true)
        
    Returns:
        JSON string with job status
    """
    job_id = str(uuid.uuid4())[:8]
    
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "source_path": source_path,
        "corpus": corpus,
        "title": title,
        "authors": authors,
        "tier": tier,
        "categories": categories or [],
        "chunks_created": 0,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }
    
    _MOCK_CHUNKING_JOBS[job_id] = job
    
    # TODO: Implement actual background chunking
    # For now, mark as processing and complete
    
    if background:
        # Simulate background processing
        job["status"] = "processing"
        return json.dumps({
            "job_id": job_id,
            "status": "queued",
            "message": f"Chunking job {job_id} queued for {source_path}",
            "check_status": f"knowledge_base___chunk_status(job_id='{job_id}')",
        }, indent=2)
    else:
        # Simulate synchronous processing
        job["status"] = "completed"
        job["progress"] = 100
        job["chunks_created"] = 42
        job["completed_at"] = datetime.now().isoformat()
        return json.dumps({
            "job_id": job_id,
            "status": "completed",
            "chunks_created": 42,
            "message": f"Successfully chunked {source_path}",
        }, indent=2)


def knowledge_base___chunk_status(job_id: str) -> str:
    """
    Check the status of a chunking job.
    
    Args:
        job_id: The job identifier
        
    Returns:
        JSON string with job status
    """
    job = _MOCK_CHUNKING_JOBS.get(job_id)
    if not job:
        return json.dumps({"error": f"Job not found: {job_id}"}, indent=2)
    
    return json.dumps({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "chunks_created": job["chunks_created"],
        "error": job.get("error"),
        "source_path": job["source_path"],
    }, indent=2)


def knowledge_base___assign_to_agent(
    agent_id: str,
    book_ids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    assigned_by: str = "self",
) -> str:
    """
    Assign titles/categories to an agent.
    
    Args:
        agent_id: The agent identifier
        book_ids: List of book IDs to assign
        categories: List of categories to assign
        tags: List of tags to assign
        assigned_by: Who assigned ('planner', 'self', 'yaml')
        
    Returns:
        JSON string with assignment status
    """
    if agent_id not in _MOCK_ASSIGNMENTS:
        _MOCK_ASSIGNMENTS[agent_id] = []
    
    assigned = []
    
    if book_ids:
        for book_id in book_ids:
            assignment = f"book:{book_id}"
            if assignment not in _MOCK_ASSIGNMENTS[agent_id]:
                _MOCK_ASSIGNMENTS[agent_id].append(assignment)
                assigned.append(assignment)
    
    if categories:
        for cat in categories:
            assignment = f"category:{cat}"
            if assignment not in _MOCK_ASSIGNMENTS[agent_id]:
                _MOCK_ASSIGNMENTS[agent_id].append(assignment)
                assigned.append(assignment)
    
    if tags:
        for tag in tags:
            assignment = f"tag:{tag}"
            if assignment not in _MOCK_ASSIGNMENTS[agent_id]:
                _MOCK_ASSIGNMENTS[agent_id].append(assignment)
                assigned.append(assignment)
    
    return json.dumps({
        "agent_id": agent_id,
        "assigned": assigned,
        "total_assignments": len(_MOCK_ASSIGNMENTS[agent_id]),
        "assigned_by": assigned_by,
        "message": f"Assigned {len(assigned)} items to agent {agent_id}",
    }, indent=2)


def knowledge_base___list_assignments(
    agent_id: Optional[str] = None,
) -> str:
    """
    List assignments for one or all agents.
    
    Args:
        agent_id: Specific agent, or all agents if omitted
        
    Returns:
        JSON string with assignments
    """
    if agent_id:
        assignments = _MOCK_ASSIGNMENTS.get(agent_id, [])
        return json.dumps({
            "agent_id": agent_id,
            "assignments": assignments,
        }, indent=2)
    else:
        return json.dumps({
            "agents": {aid: assignments for aid, assignments in _MOCK_ASSIGNMENTS.items()},
            "total_agents": len(_MOCK_ASSIGNMENTS),
        }, indent=2)


def knowledge_base___get_catalog() -> str:
    """
    Get the full knowledge base catalog.
    
    Returns:
        JSON string with catalog
    """
    return json.dumps(_MOCK_CATALOG, indent=2)


# ── PUBLIC EXPORTS ────────────────────────────────────────────────────────────

__all__ = [
    # Search and read
    "knowledge_base___search",
    "knowledge_base___read",
    "knowledge_base___list",
    "knowledge_base___describe",
    # Recommendations
    "knowledge_base___recommend_titles",
    # Chunking
    "knowledge_base___chunk_document",
    "knowledge_base___chunk_status",
    # Assignment
    "knowledge_base___assign_to_agent",
    "knowledge_base___list_assignments",
    # Catalog
    "knowledge_base___get_catalog",
    # File ingestion (from vault roots)
    "knowledge_base___discover_files",
    "knowledge_base___ingest_from_root",
    "knowledge_base___ingest_directory",
    "knowledge_base___list_vault_knowledge",
]