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


def _qdrant_api_key() -> str:
    """The Qdrant API key, from the environment or the settings object.

    ``config.settings`` does not always expose the key (the field name differs
    between deployments), and a missing key is what the client reports as a
    bare ``401`` — on the host that made every other instance look unusable and
    the plugin fall back to a *different* Qdrant that answered without a key.
    """
    for name in ("QDRANT_API_KEY", "QDRANT__SERVICE__API_KEY"):
        value = os.getenv(name)
        if value:
            return value.strip()
    settings = _settings()
    if settings is not None:
        for attr in ("qdrant_api_key", "QDRANT_API_KEY"):
            value = getattr(settings, attr, None)
            if value:
                return str(value).strip()
        # pydantic Settings keep the raw env in ``model_extra`` when the model
        # does not declare the field explicitly.
        extra = getattr(settings, "model_extra", None) or {}
        value = extra.get("QDRANT_API_KEY") or extra.get("qdrant_api_key")
        if value:
            return str(value).strip()
    return ""


def _qdrant():
    """Lazy Qdrant client (``None`` when no reachable instance exists).

    The client is bound to the first URL that answers a health probe **and**
    knows this plugin's collection. Probing the health endpoint alone is not
    enough: on a workstation ``127.0.0.1:6333`` is often a *different* Qdrant
    instance that answers happily but has no ``knowledge_base_chunks``, so the
    corpus looked empty while the real one (behind ``QDRANT_HOST_URL``) was
    never reached. An instance without the collection is still remembered as a
    last resort, so a fresh deployment can create it on the first ingest.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # noqa: BLE001
        logger.debug("[KNOWLEDGE_BASE] qdrant-client missing: %s", exc)
        return None

    api_key = _qdrant_api_key()
    fallback = None
    for url in _qdrant_urls():
        try:
            client = QdrantClient(
                url=url, api_key=api_key or None, timeout=5, check_compatibility=False
            )
            client.get_collections()  # probe reachability
            try:
                has_collection = client.collection_exists(KB_COLLECTION)
            except Exception:  # noqa: BLE001 - older client / permission detail
                has_collection = False
            if has_collection:
                _qdrant_client = client
                logger.debug("[KNOWLEDGE_BASE] Qdrant at %s (has %s)", url, KB_COLLECTION)
                return client
            if fallback is None:
                fallback = (url, client)
                logger.debug(
                    "[KNOWLEDGE_BASE] Qdrant at %s answers but has no %s; "
                    "trying the next candidate", url, KB_COLLECTION,
                )
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.debug("[KNOWLEDGE_BASE] Qdrant %s unreachable: %s", url, exc)

    if fallback is not None:
        url, client = fallback
        logger.debug("[KNOWLEDGE_BASE] no instance has %s; using %s", KB_COLLECTION, url)
        _qdrant_client = client
        return client
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
    Index a document (PDF/EPUB/MD) into the knowledge base.

    This used to be a stub: it answered "queued" with a job id, recorded it in
    an in-memory dict and created 42 fake chunks — nothing reached Qdrant, so a
    finished-looking call left an empty corpus. It now does the work and reports
    what it actually wrote.

    A PDF is extracted to text first (Docling when installed, PyMuPDF
    otherwise) and then indexed by the same code path as markdown, so both end
    up in the same shape (:mod:`extract`, :func:`indexer.index_text`).

    Args:
        source_path: Path to the document (vault path or absolute path)
        corpus: Corpus to add to
        title: Document title (derived from the file name if omitted)
        authors: Document authors
        tier: Content tier (foundational, standard, specialist, task)
        categories: Categories to assign
        background: Kept for API compatibility. Indexing is synchronous because
            there is no worker to drain a queue; a "queued" answer that nothing
            processes is what made this tool look broken.

    Returns:
        JSON string with the ingestion result (status, chunks, book_id)
    """
    job_id = str(uuid.uuid4())[:8]
    try:
        from langgraph_crew_knowledge_base.ingest import (
            knowledge_base___ingest_from_root,
        )

        # One implementation: the vault ingest already resolves a path, picks
        # the extraction, indexes and reports. Calling it keeps this tool from
        # drifting from the code that actually writes the corpus.
        raw = knowledge_base___ingest_from_root(
            root_ref=source_path,
            corpus=corpus,
            title=title,
            authors=authors,
            tier=tier,
            categories=categories,
            background=False,
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the caller
        return json.dumps(
            {"job_id": job_id, "status": "error", "error": str(exc)},
            indent=2,
        )

    payload = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    payload.setdefault("job_id", job_id)
    if payload.get("error") and payload.get("status") != "error":
        payload["status"] = "error"
    if background:
        # Say plainly that nothing is running in the background, so a caller
        # does not poll chunk_status for a job that already finished.
        payload["note"] = (
            "indexing is synchronous: this result is final, there is no "
            "background job to poll"
        )
    return json.dumps(payload, indent=2)


def knowledge_base___chunk_status(job_id: str) -> str:
    """
    Check the status of a chunking job.

    Jobs are recorded only while they run (indexing is synchronous since the
    stub was removed), so an unknown id is the normal answer rather than an
    error: the result was already returned by ``chunk_document``/
    ``ingest_from_root``.

    Args:
        job_id: The job identifier

    Returns:
        JSON string with job status
    """
    job = _MOCK_CHUNKING_JOBS.get(job_id)
    if not job:
        return json.dumps({
            "job_id": job_id,
            "status": "unknown",
            "message": (
                "no such job. Indexing is synchronous — the ingest already "
                "returned the final result (status, chunks, book_id)."
            ),
        }, indent=2)

    return json.dumps({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "chunks_created": job.get("chunks_created"),
        "error": job.get("error"),
        "source_path": job.get("source_path"),
    }, indent=2)


def knowledge_base___assign_to_agent(
    agent_id: str,
    book_ids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    assigned_by: str = "self",
    scope: str = "project",
    customer: str = "",
    project: str = "",
) -> str:
    """
    Assign titles/categories to an agent, at a chosen SCOPE.

    An assignment is "this agent always has these titles in scope": the
    knowledge discovery adds them to every task the agent runs there, without
    searching first. Assignments are persisted (SQLite), so one run can set them
    and the next run still sees them.

    `scope` is the decision that matters — who should get the title:

        "deployment"  every agent of every customer (the generally available
                      corpus; written to the deployment/"custom" rung)
        "client"      every project of one customer
        "project"     one project only (the default — narrowest, safest)

    A run sees deployment ∪ its customer ∪ its project; a narrower assignment
    never leaks into another scope.

    Args:
        agent_id: The agent identifier (e.g. "digital_marketing_manager_agent")
        book_ids: Book ids to assign (the ``book_id`` reported by the ingest)
        categories: Categories to assign
        tags: Tags to assign
        assigned_by: Who assigned ('planner', 'self', 'yaml', 'operator')
        scope: 'project' (default) | 'client' | 'deployment'
        customer: Customer for client/project scope (default: active scope)
        project: Project for project scope (default: active scope)

    Returns:
        JSON string with the newly added entries and the agent's full list
    """
    from langgraph_crew_knowledge_base.assignments import assign

    result = assign(
        agent_id,
        books=book_ids or (),
        categories=categories or (),
        tags=tags or (),
        assigned_by=assigned_by,
        scope=scope,
        customer=customer,
        project=project,
    )
    if result.get("error"):
        return json.dumps(result, indent=2)

    added = result.get("assigned") or []
    target = result.get("scope") or {}
    where = target.get("scope", scope)
    if where == "client":
        where += f" ({target.get('customer')})"
    elif where == "project":
        where += f" ({target.get('customer')}/{target.get('project')})"
    return json.dumps({
        "agent_id": agent_id,
        "assigned": added,
        "scope": target,
        "database": result.get("database"),
        "total_assignments": len(result.get("all") or []),
        "all_assignments": result.get("all") or [],
        "assigned_by": assigned_by,
        "message": (
            f"Assigned {len(added)} new item(s) to {agent_id} at {where}"
            + ("" if added else " (all already present)")
        ),
    }, indent=2)


def knowledge_base___list_assignments(
    agent_id: Optional[str] = None,
    scope: Optional[str] = None,
) -> str:
    """
    List assignments for one or all agents.

    With no `scope`, one agent is listed ladder-aware: what that agent sees
    from the ACTIVE customer/project (deployment ∪ client ∪ project). With a
    `scope` ('deployment' | 'client' | 'project' | '*'), exactly that rung is
    listed.

    Args:
        agent_id: Specific agent, or all agents if omitted
        scope: Optional rung filter ('deployment', 'client', 'project', '*' = all)

    Returns:
        JSON string with assignments
    """
    from langgraph_crew_knowledge_base.assignments import (
        list_all,
        list_for_agent,
        current_scope,
    )

    live = current_scope()
    if agent_id:
        return json.dumps({
            "agent_id": agent_id,
            "scope": scope or "ladder",
            "active_customer": live.get("customer") or None,
            "active_project": live.get("project") or None,
            "assignments": list_for_agent(agent_id, scope=scope),
        }, indent=2)

    everything = list_all(scope=scope)
    return json.dumps({
        "agents": everything,
        "scope": scope or "all",
        "total_agents": len(everything),
    }, indent=2)


def knowledge_base___grant_access(
    agent_id: str,
    scope: str = "project",
    customer: str = "",
    project: str = "",
    titles: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    level: str = "",
    authorized_by: str = "",
    authorization_note: str = "",
    assigned_by: str = "self",
) -> str:
    """
    Grant knowledge access to an agent AND record it where that scope lives.

    `assign_to_agent` says what an agent may read inside the knowledge store.
    This tool additionally writes the same grant into the agent's MANIFEST at
    the matching description-language level, so the decision becomes part of
    the deployment rather than a row only this plugin knows about:

        scope 'deployment' -> custom/agents/<agent>.md      (everyone)
        scope 'client'     -> <customer>/agents/<agent>.md  (that customer)
        scope 'project'    -> <project>/agents/<agent>.md   (that project)

    Decide the scope by who should get the title. Narrow is the safe default:
    a project-level grant cannot surprise another project. Wider scopes
    (client, deployment) affect other scopes than the one you are in, so they
    need an explicit instruction — pass `authorized_by="user"` (or "planner")
    plus `authorization_note`, otherwise the call is refused and you should ask
    the human first.

    `level` may override the manifest level independently of `scope` (defaults
    to the scope). Note that the repo is read-only inside the container, so a
    `custom`/`deployment` manifest write is refused there: the store row is
    still written (the grant is active) and the response says where the
    operator has to apply the file.

    Args:
        agent_id: Agent key, e.g. "project_manager_agent"
        scope: 'project' (default) | 'client' | 'deployment'
        customer: Customer for client/project scope (default: active scope)
        project: Project for project scope (default: active scope)
        titles: Book ids to grant
        categories: Categories to grant
        tags: Tags to grant
        level: Manifest level override ('project' | 'client' | 'custom')
        authorized_by: 'user' or 'planner' — required for client/deployment
        authorization_note: What exactly was ordered, and by whom
        assigned_by: Who set the store row ('self', 'operator', 'planner', …)

    Returns:
        JSON string with the store result and the manifest write result
    """
    import os

    from langgraph_crew_knowledge_base.assignments import assign, current_scope

    live = current_scope()
    customer = (customer or live.get("customer") or "").strip()
    project = (project or live.get("project") or "").strip()

    store = assign(
        agent_id,
        books=titles or (),
        categories=categories or (),
        tags=tags or (),
        assigned_by=assigned_by,
        scope=scope,
        customer=customer,
        project=project,
    )
    if store.get("error"):
        return json.dumps({"stored": False, "store": store}, indent=2)

    manifest = _write_manifest_grant(
        agent_id,
        target=store.get("scope") or {},
        titles=titles or [],
        categories=categories or [],
        tags=tags or [],
        level=level,
        authorized_by=authorized_by,
        authorization_note=authorization_note,
    )

    if not manifest.get("written"):
        where = "store row written; manifest not written (see manifest.error)"
    elif (level or (store.get("scope") or {}).get("scope")) == "project":
        where = "this project only (active for the next run, no reload needed)"
    else:
        where = "wider scope written — call runtime___spec_reload to activate"

    return json.dumps({
        "agent_id": agent_id,
        "scope": store.get("scope"),
        "stored": bool(store.get("assigned") is not None),
        "assigned": store.get("assigned") or [],
        "database": store.get("database"),
        "manifest": manifest,
        "active": where,
        "path": manifest.get("path"),
    }, indent=2)


def _manifest_path_for(name: str, level: str) -> Optional[str]:
    """``<level>/agents/<name>.md`` when that exact file is the one to merge.

    ``None`` means "write a fresh manifest at ``level``" — either the file does
    not exist, or the ladder resolves to a more specific level whose manifest
    must not be touched.
    """
    import os

    try:
        from app.core.spec.authoring import _custom_root as authoring_custom_root
        from app.core.spec.authoring import _project_root as authoring_project_root
        from app.core.spec.loader import spec_roots
    except Exception:  # noqa: BLE001
        return None

    try:
        if level == "custom":
            root = authoring_custom_root()
        elif level == "project":
            root = authoring_project_root()
        else:
            # client: <client_vaults>/<customer>/agents/<name>.md
            from app.core import access_scope

            scope = access_scope.current_scope()
            client_root = str(getattr(scope, "client_root", "") or "")
            if not client_root:
                return None
            root = client_root
    except Exception:  # noqa: BLE001 - authoring raises with its own message
        return None
    if not root:
        return None
    candidate = os.path.join(root, "agents", f"{name}.md")

    # The ladder may resolve to a MORE specific file than the level being
    # written (a project manifest beats the custom one). Merging into that file
    # would put the grant in the wrong scope silently — merge only into a file
    # that belongs to this very level.
    for entry in spec_roots():
        if entry.level != level:
            continue
        try:
            if os.path.realpath(entry.path) == os.path.realpath(root):
                return candidate
        except OSError:
            continue
    return None


def _write_manifest_grant(
    agent_id: str,
    *,
    target: Dict[str, Any],
    titles: List[str],
    categories: List[str],
    tags: List[str],
    level: str = "",
    authorized_by: str = "",
    authorization_note: str = "",
) -> Dict[str, Any]:
    """Write/merge ``x-penta.knowledge_base`` into the agent's manifest.

    Merges with what is already declared at that level (a manifest may carry
    other x-penta keys, or other titles): the grant adds, it does not replace.
    """
    import os

    import yaml

    from app.core.spec.authoring import AuthoringError, write_manifest

    scope_name = str(target.get("scope") or "project")
    level = (level or "").strip() or (
        "custom" if scope_name == "deployment" else scope_name
    )
    name = str(agent_id or "").strip().removesuffix("_agent")
    if not name:
        return {"written": False, "error": "agent_id is required"}

    if not (titles or categories or tags):
        return {
            "written": False,
            "skipped": "no titles/categories/tags given — nothing to declare",
        }

    # Merge into the file that is actually there, not into a fresh template.
    # A hand-written manifest carries a description, a body and other x-penta
    # keys; replacing it with a generated stub would silently destroy them, so
    # the existing frontmatter is read as raw YAML and only the
    # knowledge_base block is touched. The body is kept verbatim.
    data: Dict[str, Any] = {}
    body_text = ""
    same_level = _manifest_path_for(name, level)
    target_file = same_level
    if same_level is None:
        # No manifest of this exact level yet: read the most specific one that
        # exists to keep its human-written description and body, while the
        # knowledge block is written at the requested level (the ladder has to
        # still resolve to the level with the new grant).
        try:
            from app.core.spec.loader import load_agent_manifest

            resolved = load_agent_manifest(name)
            target_file = getattr(resolved, "source_path", None)
        except Exception:  # noqa: BLE001 - best effort
            target_file = None
    if target_file and os.path.isfile(target_file):
        try:
            from app.core.spec.models import split_frontmatter

            data, body_text = split_frontmatter(
                open(target_file, encoding="utf-8").read()
            )
        except Exception:  # noqa: BLE001 - a broken file gets a minimal header
            data, body_text = {}, ""

    if not data.get("name"):
        data.update(
            {
                "spec_version": 1,
                "kind": "agent",
                "name": name,
                "description": (
                    f"Knowledge access for {name}: corpus grants declared at "
                    f"{level} level."
                ),
            }
        )

    x_penta = data.get("x-penta")
    if not isinstance(x_penta, dict):
        x_penta = {} if x_penta in (None, "") else {"_penta": x_penta}
        data["x-penta"] = x_penta
    knowledge = x_penta.get("knowledge_base")
    if not isinstance(knowledge, dict):
        knowledge = {}
        x_penta["knowledge_base"] = knowledge

    existing_titles = [str(t) for t in (knowledge.get("titles") or [])]
    existing_categories = [str(c) for c in (knowledge.get("categories") or [])]
    existing_tags = [str(t) for t in (knowledge.get("tags") or [])]

    merged_titles = list(dict.fromkeys(existing_titles + list(titles)))
    merged_categories = list(dict.fromkeys(existing_categories + list(categories)))
    merged_tags = list(dict.fromkeys(existing_tags + list(tags)))

    knowledge["titles"] = merged_titles
    knowledge["categories"] = merged_categories
    knowledge["tags"] = merged_tags

    header = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    content = f"---\n{header}---\n{body_text}"

    try:
        result = write_manifest(
            "agent",
            name,
            content,
            level=level,
            overwrite=True,
            authorized_by=authorized_by,
            authorization_note=authorization_note,
        )
    except AuthoringError as exc:
        return {
            "written": False,
            "level": level,
            "error": str(exc),
            "hint": (
                "the store row IS written — the agent has access from this "
                "run on. Only the manifest (the durable declaration for that "
                "level) was refused."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - reporting, never crashing
        return {"written": False, "level": level, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "written": bool(result.get("written")),
        "level": level,
        "path": result.get("path"),
        "titles": merged_titles,
        "categories": merged_categories,
        "tags": merged_tags,
    }


def knowledge_base___access_matrix() -> str:
    """
    Show WHO may read WHAT, and at which scope — the access overview table.

    Answers the question an operator actually asks ("where is this title
    available?"): every grant per agent, per scope, with the filesystem place
    that grant is declared in — so the same table can be written to, not only
    read.

    Returns:
        JSON string: scopes, per-agent grants, and per-title coverage
    """
    import os

    from langgraph_crew_knowledge_base.assignments import (
        current_scope,
        db_path,
        grants,
    )

    rows = grants()
    live = current_scope()

    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    by_value: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        entry = {
            "kind": row["kind"],
            "value": row["value"],
            "scope": row["scope"],
            "customer": row["customer"],
            "project": row["project"],
        }
        by_agent.setdefault(row["agent_id"], []).append(entry)
        by_value.setdefault(f"{row['kind']}:{row['value']}", []).append(
            {"agent_id": row["agent_id"], **entry}
        )

    # Where each scope is declared on disk (the writable half of the table).
    ladder = {
        "deployment": "custom/agents/<agent>.md      → every customer",
        "client": "clients/<customer>/agents/<agent>.md → one customer",
        "project": "<project>/agents/<agent>.md      → one project",
        "legacy": "rows written before scoping existed (read as deployment)",
    }

    return json.dumps(
        {
            "database": str(db_path()),
            "writable": os.access(str(db_path().parent), os.W_OK),
            "active_scope": live,
            "scopes": ["deployment", "client", "project", "legacy"],
            "ladder": ladder,
            "agents": by_agent,
            "titles": by_value,
            "total_rows": len(rows),
            "how_to_write": (
                "knowledge_base___grant_access(agent_id, scope=…, titles=…, "
                "authorized_by='user', authorization_note=…) — the same table, "
                "written instead of read"
            ),
        },
        indent=2,
    )


def _build_catalog_from_qdrant() -> Dict[str, Any]:
    """Build the catalog by scanning the indexed chunks in Qdrant."""
    client = _qdrant()
    if client is None:
        return {"corpora": {}, "total_books": 0, "total_chunks": 0}
    try:
        if not client.collection_exists(KB_COLLECTION):
            return {"corpora": {}, "total_books": 0, "total_chunks": 0}
        
        # Scroll through all points to aggregate books
        points, _ = client.scroll(KB_COLLECTION, limit=1000, with_payload=True)
        
        corpora: Dict[str, Dict[str, Any]] = {}
        book_chunks: Dict[str, int] = {}
        
        for p in points:
            payload = p.payload or {}
            book_id = payload.get("book_id") or "unknown"
            corpus = payload.get("corpus") or "default"
            
            # Track chunk count
            book_chunks[book_id] = book_chunks.get(book_id, 0) + 1
            
            # Build corpus entry
            if corpus not in corpora:
                corpora[corpus] = {
                    "name": corpus.replace("-", " ").title(),
                    "books": {},
                }
            
            # Build book entry (first chunk wins for metadata)
            if book_id not in corpora[corpus]["books"]:
                corpora[corpus]["books"][book_id] = {
                    "title": payload.get("title") or book_id,
                    "authors": payload.get("authors") or [],
                    "year": payload.get("year"),
                    "tier": payload.get("tier") or "standard",
                    "categories": payload.get("categories") or [],
                    "chunks": 0,
                }
            corpora[corpus]["books"][book_id]["chunks"] += 1
        
        return {
            "corpora": corpora,
            "total_books": sum(len(c["books"]) for c in corpora.values()),
            "total_chunks": sum(book_chunks.values()),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("[KNOWLEDGE_BASE] catalog build failed: %s", exc)
        return {"corpora": {}, "total_books": 0, "total_chunks": 0}


def knowledge_base___get_catalog() -> str:
    """
    Get the full knowledge base catalog.
    
    Returns:
        JSON string with catalog
    """
    # Build from Qdrant (real data) first
    catalog = _build_catalog_from_qdrant()
    if catalog["total_chunks"] > 0:
        return json.dumps(catalog, indent=2)
    
    # Fall back to mock catalog for demo purposes
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
    # Assignment (store) + access control (store + manifest)
    "knowledge_base___assign_to_agent",
    "knowledge_base___list_assignments",
    "knowledge_base___grant_access",
    "knowledge_base___access_matrix",
    # Catalog
    "knowledge_base___get_catalog",
    # File ingestion (from vault roots)
    "knowledge_base___discover_files",
    "knowledge_base___ingest_from_root",
    "knowledge_base___ingest_directory",
    "knowledge_base___list_vault_knowledge",
]