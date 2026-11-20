#!/usr/bin/env python3
"""Interactive knowledge-base manager (plugin CLI: ``knowledge-manager``).

What is in the corpus, which agent may read which base, where that differs per
customer and per project — and the same view as an editor: assign and unassign
an agent, at the scope you choose.

    python scripts/knowledge_manager.py            # interactive
    python scripts/knowledge_manager.py --list     # bases + books, then exit
    python scripts/knowledge_manager.py --bindings # grant table, then exit
    python scripts/knowledge_manager.py --namespaces
    python scripts/knowledge_manager.py --grant cmo_agent --book <id> --scope deployment \\
        --authorized-by user --note "operator: everyone may use it"
    python scripts/knowledge_manager.py --unassign cmo_agent --book <id> --scope project

In the REPL::

    bases                       list the bases (= corpora) and their books
    books [query]               list books, filter by title/author
    bindings [query]            who may read what, at which scope
    namespaces                  how the bindings differ per customer/project
    effective <agent> <c>/<p>   what one agent actually sees in a namespace
    search <query>              semantic search over the corpus
    assign <agent>              guided: pick scope, customer/project, base/book
    unassign <agent>            guided: pick an existing grant and remove it
    help / quit

The manager only ever writes through
:mod:`langgraph_crew_knowledge_base.assignments` and
:mod:`langgraph_crew_knowledge_base.manager`, i.e. the same code the agent tools
use. A change made here is the change an agent sees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_root = os.path.dirname(script_dir)
# The repo root is the plugin's grandparent (``<repo>/plugins/<name>/scripts``):
# that is where ``app/`` and ``.env`` live.
project_root = os.path.dirname(os.path.dirname(plugin_root))
for _path in (plugin_root, project_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The corpus lives in the deployment's Qdrant, reached with the API key from
# `.env`. A bare `python script.py` starts with an empty environment, so every
# probe would get a 401 and the manager would report an empty corpus while the
# data is right there. Load it before anything imports `config`.
try:
    from dotenv import load_dotenv

    for _candidate in (os.path.join(project_root, ".env"),
                       os.path.join(plugin_root, "plugin.env")):
        if os.path.isfile(_candidate):
            load_dotenv(_candidate, override=False)
            break
except Exception:  # noqa: BLE001 - env loading is best effort
    pass

# ── ANSI styles ─────────────────────────────────────────────────────────────


class Style:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{Style.RESET}"


def _hr(width: int = 78) -> None:
    print(_c("─" * width, Style.DIM))


SCOPES = ("deployment", "client", "project")
SCOPE_HELP = {
    "deployment": "every customer of this deployment (custom/agents/…)",
    "client": "every project of ONE customer",
    "project": "one project only (the safe default)",
}


# ── views ───────────────────────────────────────────────────────────────────


def show_bases(manager, query: str = "") -> None:
    books = manager.books
    rows = manager.bases
    if query:
        keep = {
            b["corpus"] for b in manager.find(query, books, ("title", "book_id", "corpus", "authors"))
        }
        rows = [b for b in rows if b["base"] in keep]

    print()
    print(_c("  BASES (a base = a corpus: the set of books an agent can be given)", Style.BOLD + Style.CYAN))
    _hr()
    if not rows:
        print(f"  {_c('The corpus is empty.', Style.YELLOW)}")
        print(f"  {_c('Ingest a real document:', Style.DIM)}")
        # Built outside the f-string: a backslash inside braces is PEP 701
        # syntax (Python 3.12+) and a hard SyntaxError on the 3.9 floor this
        # project declares — the whole file failed to import, so `/knowledge`
        # died with an "Error in chat loop" instead of running.
        hint = (
            '  knowledge_base___ingest_from_root('
            'root_ref="project-vault:Knowledge/book.pdf", corpus="…")'
        )
        print(f"  {_c(hint, Style.DIM)}")
        return

    for base in rows:
        real = base["real_books"]
        total = len(base["books"])
        flag = "" if real == total else _c(f"  ⚠ {total - real} without real text", Style.YELLOW)
        # Nested f-string with a repeated quote is PEP 701 too (3.12+).
        summary = _c(f"{total} book(s) · {base['chunks']} chunk(s)", Style.DIM)
        print(
            f"\n  {_c(base['base'], Style.BOLD + Style.GREEN)}"
            f"  {summary}{flag}"
        )
        for book in sorted(base["books"], key=lambda b: b["title"].lower()):
            mark = _c("✓", Style.GREEN) if book["is_real"] else _c("!", Style.YELLOW)
            authors = ", ".join(book["authors"])[:28] or "—"
            print(
                f"    {mark} {book['title'][:44]:<44} {_c(authors, Style.DIM):<40}"
                f" {book['chunks']:>3} ch  {_c(book['book_id'], Style.DIM)}"
            )
    print()
    print(f"  {_c('✓ real text   ! thin/metadata only (a summary, not the book)', Style.DIM)}")
    print()


def show_bindings(manager, query: str = "") -> None:
    rows = manager.bindings
    if query:
        rows = manager.find(query, rows, ("agent_id", "title", "value", "base", "customer", "project", "scope"))

    print()
    print(_c("  BINDINGS (who may read what, and where)", Style.BOLD + Style.CYAN))
    _hr()
    if not rows:
        print(f"  {_c('No agent has anything assigned.', Style.YELLOW)}")
        # Single quotes around the hint: `"` inside a double-quoted f-string is
        # PEP 701 (3.12+) and made this whole file unimportable on 3.9.
        print(f"  {_c('Use `assign <agent>` to give an agent access.', Style.DIM)}")
        print()
        return

    for scope in ("deployment", "legacy", "client", "project"):
        group = [r for r in rows if r["scope"] == scope]
        if not group:
            continue
        label = "legacy (pre-scope, read as deployment)" if scope == "legacy" else scope
        print(f"\n  {_c(label.upper(), Style.BOLD + Style.MAGENTA)}  {_c(SCOPE_HELP.get(scope, ''), Style.DIM)}")
        for row in group:
            where = ""
            if scope == "client":
                where = _c(f"  [{row['customer']}]", Style.BLUE)
            elif scope == "project":
                where = _c(f"  [{row['customer']}/{row['project']}]", Style.BLUE)
            title = row["title"][:40]
            if row["missing"]:
                title = _c(f"{title}  ⚠ no longer in the corpus", Style.RED)
            print(f"    {_c(row['agent_id'], Style.GREEN):<38} {title}{where}")
    print()
    print(f"  {_c('{0} binding(s) · store: {1}'.format(len(rows), manager.db_path), Style.DIM)}")
    print()


def show_namespaces(manager) -> None:
    rows = manager.namespaces
    print()
    print(_c("  NAMESPACES (where the bindings differ)", Style.BOLD + Style.CYAN))
    _hr()
    if not rows:
        print(f"  {_c('No customer/project-specific binding — everything is deployment-wide.', Style.DIM)}")
        print()
        return
    print(f"  {'NAMESPACE':<44} {'GRANTS':>6} {'PROJECT':>8} {'CLIENT':>7}  AGENTS")
    for row in rows:
        print(
            f"  {row['label'][:44]:<44} {row['grants']:>6} {row['project_specific']:>8}"
            f" {row['client_wide']:>7}  {_c(', '.join(row['agents'])[:30], Style.DIM)}"
        )
    print()
    print(f"  {_c('Deployment grants reach every namespace and are not listed here.', Style.DIM)}")
    print()


def show_effective(manager, agent_id: str, namespace: str) -> None:
    customer, _, project = namespace.partition("/")
    rows = manager.effective(agent_id, customer, project)
    print()
    print(_c(f"  EFFECTIVE for {agent_id} in {namespace or '(no namespace)'}", Style.BOLD + Style.CYAN))
    _hr()
    if not rows:
        print(f"  {_c('Nothing — this agent searches the corpus per task.', Style.YELLOW)}")
        print()
        return
    for row in rows:
        print(f"    {_c(row['scope'], Style.MAGENTA):<24} {row['title'][:50]}")
    print()


def show_profiles() -> None:
    """Per-book score profiles and the outlier report.

    A profile is an ingest diagnostic, not a runtime threshold: measured on this
    deployment the books differ by ~0.04 in median while the spread inside one
    book is ~0.13-0.19, so a per-book threshold would move by +/-0.02 and change
    nothing. What it catches is a book that scores systematically low.
    """
    from langgraph_crew_knowledge_base.indexer import report_outliers
    from langgraph_crew_knowledge_base.state_paths import resolve_db_path
    import json as _json

    base = resolve_db_path(None).parent / "book_profiles"
    print()
    print(_c("  BOOK SCORE PROFILES (ingest diagnostic, not a runtime threshold)", Style.BOLD + Style.CYAN))
    _hr()
    if not base.is_dir():
        print(f"  {_c('No profiles yet. Run with --profile-books.', Style.YELLOW)}")
        print()
        return

    for path in sorted(base.glob("*.json")):
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        print(
            f"    {data.get('book_id', path.stem)[:44]:46} "
            f"median {data.get('median', 0):.3f}  "
            f"{_c(f'p10 {data.get(chr(112)+chr(49)+chr(48), 0):.3f}  p90 {data.get(chr(112)+chr(57)+chr(48), 0):.3f}', Style.DIM)}"
            f"  n={data.get('samples', 0)}"
        )

    out = report_outliers()
    if out:
        print(f"\n  {_c('OUTLIERS', Style.BOLD + Style.YELLOW)}")
        for entry in out:
            print(f"    {_c(entry['book_id'], Style.RED)}  z={entry['z']}  {entry['hint']}")
    else:
        print(f"\n  {_c('No outlier: every book scores within 2 sigma of the deployment median.', Style.DIM)}")
    print()


def profile_corpus(manager) -> None:
    """Measure the profile of every book currently in the corpus."""
    from langgraph_crew_knowledge_base.indexer import profile_book, write_profile
    from langgraph_crew_knowledge_base.tools import _qdrant

    client = _qdrant()
    if client is None:
        print(f"  {_c('Qdrant unreachable', Style.RED)}")
        return
    print()
    print(_c("  MEASURING BOOK PROFILES", Style.BOLD + Style.CYAN))
    _hr()
    for book in manager.books:
        profile = profile_book(client, book["book_id"], book["corpus"])
        if profile:
            write_profile(book["book_id"], profile)
            print(f"    {_c('OK', Style.GREEN)} {book['title'][:44]:46} median {profile['median']:.3f}")
        else:
            print(f"    {_c('--', Style.YELLOW)} {book['title'][:44]:46} no scores")
    print()


def show_search(manager, query: str, top_k: int = 5) -> None:
    print()
    print(_c(f"  SEARCH “{query}”", Style.BOLD + Style.CYAN))
    _hr()
    try:
        from langgraph_crew_knowledge_base.tools import knowledge_base___search

        payload = json.loads(knowledge_base___search(query=query, top_k=top_k))
    except Exception as exc:  # noqa: BLE001 - the manager must not die on a search
        print(f"  {_c(f'search failed: {exc}', Style.RED)}")
        print()
        return

    results = payload.get("results") or []
    if not results:
        print(f"  {_c('No hits.', Style.YELLOW)}")
        print()
        return
    for hit in results:
        text = (hit.get("text") or "").strip().replace("\n", " ")
        score = hit.get("score", 0)
        print(f"    {_c(format(score, '.2f'), Style.GREEN)}  {hit.get('title', '')[:46]}")
        print(f"         {_c(text[:120] + ('…' if len(text) > 120 else ''), Style.DIM)}")
        print(f"         {_c(hit.get('kb_ref', ''), Style.BLUE)}")
    print()


# ── write paths ─────────────────────────────────────────────────────────────


def do_assign(manager, agent_id: str, book_id: str, scope: str, customer: str, project: str,
              authorized_by: str = "", note: str = "") -> bool:
    """Store a binding. Returns True on success."""
    from langgraph_crew_knowledge_base.assignments import assign

    result = assign(
        agent_id,
        books=[book_id],
        scope=scope,
        customer=customer,
        project=project,
        assigned_by=authorized_by or "manager",
    )
    if result.get("error"):
        print(f"  {_c('✗ ' + str(result['error']), Style.RED)}")
        if scope in ("client", "deployment"):
            print(f"  {_c('  Wider than a project needs --authorized-by user|planner and --note.', Style.DIM)}")
        return False

    added = result.get("assigned") or []
    target = result.get("scope") or {}
    where = target.get("scope", scope)
    if target.get("customer"):
        where += f" {target['customer']}"
        if target.get("project"):
            where += f"/{target['project']}"
    if added:
        print(f"  {_c('✓', Style.GREEN)} {agent_id} can now read {book_id} at {_c(where, Style.MAGENTA)}")
    else:
        print(f"  {_c('•', Style.YELLOW)} already assigned at {where} — nothing to do")
    return True


def do_unassign(manager, agent_id: str, book_id: str, scope: str, customer: str, project: str) -> bool:
    from langgraph_crew_knowledge_base.assignments import unassign

    ok = unassign(agent_id, "book", book_id, scope=scope, customer=customer, project=project)
    if ok:
        print(f"  {_c('✓', Style.GREEN)} removed {book_id} from {agent_id} ({scope})")
    else:
        print(f"  {_c('•', Style.YELLOW)} no such binding")
    return ok


# ── interactive prompts ─────────────────────────────────────────────────────


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{Style.BOLD}{Style.BLUE}{prompt}{Style.RESET}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def _pick(prompt: str, options, render=str, allow_empty: bool = False):
    """Numbered pick list; returns the chosen item (or None)."""
    options = list(options)
    if allow_empty:
        options = [None] + options
    for i, option in enumerate(options, start=1):
        label = "(none)" if option is None else render(option)
        print(f"    {_c(str(i), Style.GREEN)}) {label}")
    raw = _ask(prompt)
    if not raw.isdigit() or not (1 <= int(raw) <= len(options)):
        return None
    return options[int(raw) - 1]


def guided_assign(manager, agent_id: str) -> None:
    books = manager.books
    if not books:
        print(f"  {_c('The corpus is empty — ingest a real document first.', Style.YELLOW)}")
        return

    print()
    print(_c("  ASSIGN ACCESS", Style.BOLD + Style.CYAN))
    _hr()

    # 1. scope ---------------------------------------------------------------
    print(f"\n  {_c('1. Where should this apply?', Style.BOLD)}")
    scope = _pick(
        "   scope",
        SCOPES,
        render=lambda s: f"{_c(s, Style.MAGENTA):<22} {_c(SCOPE_HELP[s], Style.DIM)}",
    )
    if scope is None:
        print(f"  {_c('cancelled', Style.DIM)}")
        return

    customer = project = ""
    if scope == "client":
        customer = _ask("   customer")
        if not customer:
            print(f"  {_c('✗ a client scope needs a customer', Style.RED)}")
            return
    elif scope == "project":
        customer = _ask("   customer")
        project = _ask("   project")
        if not (customer and project):
            print(f"  {_c('✗ a project scope needs customer and project', Style.RED)}")
            return

    # 2. what ----------------------------------------------------------------
    print(f"\n  {_c('2. What should the agent be able to read?', Style.BOLD)}")
    hits = manager.find(_ask("   filter (empty = all)"), books, ("title", "book_id", "corpus", "authors"))
    if not hits:
        print(f"  {_c('no book matches', Style.YELLOW)}")
        return
    book = _pick("   book", hits, render=lambda b: f"{b['title'][:50]}  {_c(b['book_id'], Style.DIM)}")
    if book is None:
        print(f"  {_c('cancelled', Style.DIM)}")
        return

    # 3. authorisation for a wider scope -------------------------------------
    authorizer = note = ""
    if scope in ("client", "deployment"):
        print(f"\n  {_c('3. Wider than one project — who ordered it?', Style.BOLD)}")
        authorizer = _ask("   authorized by", "user")
        note = _ask("   note (what was ordered)")
        if not note:
            print(f"  {_c('✗ a wider scope needs a note — nothing written', Style.RED)}")
            return

    do_assign(manager, agent_id, book["book_id"], scope, customer, project, authorizer, note)


def guided_unassign(manager, agent_id: str) -> None:
    rows = [b for b in manager.bindings if b["agent_id"] == agent_id and b["kind"] == "book"]
    print()
    print(_c(f"  REMOVE A BINDING of {agent_id}", Style.BOLD + Style.CYAN))
    _hr()
    if not rows:
        print(f"  {_c('this agent has no book bindings', Style.YELLOW)}")
        print()
        return
    row = _pick(
        "   binding",
        rows,
        render=lambda r: (
            f"{r['title'][:40]:<42} {_c(r['scope'], Style.MAGENTA):<22}"
            f" {_c('/'.join(x for x in (r['customer'], r['project']) if x), Style.BLUE)}"
        ),
    )
    if row is None:
        print(f"  {_c('cancelled', Style.DIM)}")
        return
    do_unassign(manager, agent_id, row["value"], row["scope"], row["customer"], row["project"])


# ── wiring ──────────────────────────────────────────────────────────────────


class Manager:
    """Read snapshot + write actions, with a refresh after every change."""

    def __init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        from langgraph_crew_knowledge_base import manager as M
        from langgraph_crew_knowledge_base.assignments import db_path

        self._M = M
        self.books = M.corpus_rows()
        self.bases = M.bases(self.books)
        self.bindings = M.binding_rows(self.books)
        self.namespaces = M.namespaces(self.bindings)
        self.db_path = str(db_path())

    @property
    def find(self):
        return self._M.find

    def effective(self, agent_id: str, customer: str = "", project: str = "") -> list:
        return self._M.effective_for(agent_id, customer, project, self.bindings)


HEADER = f"""
{Style.BOLD}{Style.CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║              KNOWLEDGE BASE MANAGER — corpus and agent bindings           ║
╚══════════════════════════════════════════════════════════════════════════╝{Style.RESET}

{Style.DIM}  bases          the bases (= corpora) and their books
  books [query]  books, filterable by title / author / base
  bindings [q]   who may read what, at which scope
  namespaces     how the bindings differ per customer / project
  effective <agent> <customer>/<project>   what one agent sees there
  search <query> semantic search over the corpus
  assign <agent>     guided: scope, namespace, base, book
  unassign <agent>   guided: pick an existing binding and remove it
  help · quit{Style.RESET}
"""


def run_repl(manager: Manager) -> None:
    print(HEADER)
    print(f"  {_c('✓', Style.GREEN)} {len(manager.bases)} base(s), {len(manager.books)} book(s), "
          f"{len(manager.bindings)} binding(s)")
    print(f"  {_c(f'store: {manager.db_path}', Style.DIM)}")

    while True:
        try:
            raw = input(f"\n{Style.BOLD}{Style.BLUE}kb> {Style.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break
        if cmd in ("help", "?"):
            print(HEADER)
        elif cmd in ("bases", "base", "corpora"):
            show_bases(manager, rest)
        elif cmd in ("books", "list"):
            show_bases(manager, rest)
        elif cmd in ("bindings", "grants"):
            show_bindings(manager, rest)
        elif cmd == "namespaces":
            show_namespaces(manager)
        elif cmd == "effective":
            if not rest:
                print(f"  {_c('usage: effective <agent> <customer>/<project>', Style.CYAN)}")
                continue
            bits = rest.split()
            show_effective(manager, bits[0], bits[1] if len(bits) > 1 else "")
        elif cmd == "search":
            if not rest:
                print(f"  {_c('usage: search <query>', Style.CYAN)}")
                continue
            show_search(manager, rest)
        elif cmd == "assign":
            if not rest:
                print(f"  {_c('usage: assign <agent_id>', Style.CYAN)}")
                continue
            guided_assign(manager, rest)
            manager.refresh()
        elif cmd == "unassign":
            if not rest:
                print(f"  {_c('usage: unassign <agent_id>', Style.CYAN)}")
                continue
            guided_unassign(manager, rest)
            manager.refresh()
        else:
            print(f"  {_c('Unknown command: ' + cmd + '. Type help.', Style.RED)}")

    print(f"\n{_c('👋 done', Style.GREEN)}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="knowledge_manager",
        description="Browse and edit the knowledge corpus and agent bindings.",
    )
    parser.add_argument("--bases", action="store_true", help="print the bases and their books, then exit")
    parser.add_argument("--bindings", action="store_true", help="print the binding table, then exit")
    parser.add_argument("--namespaces", action="store_true", help="print the namespace view, then exit")
    parser.add_argument("--search", metavar="QUERY", help="semantic search over the corpus, then exit")
    parser.add_argument("--effective", nargs=2, metavar=("AGENT", "NS"),
                        help="what AGENT sees in namespace CUSTOMER/PROJECT, then exit")
    parser.add_argument("--profiles", action="store_true",
                        help="show per-book score profiles and any outlier, then exit")
    parser.add_argument("--profile-books", action="store_true",
                        help="(re)measure the score profile of every book in the corpus")
    parser.add_argument("--grant", metavar="AGENT", help="assign a book to AGENT (needs --book)")
    parser.add_argument("--unassign", metavar="AGENT", help="remove a book binding of AGENT (needs --book)")
    parser.add_argument("--book", metavar="BOOK_ID", help="the book_id to grant/remove")
    parser.add_argument("--scope", default="project", choices=SCOPES)
    parser.add_argument("--customer", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--authorized-by", default="", help="user|planner — required for client/deployment")
    parser.add_argument("--note", default="", help="what was ordered (required for client/deployment)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manager = Manager()

    if args.bases:
        show_bases(manager)
        return 0
    if args.bindings:
        show_bindings(manager)
        return 0
    if args.namespaces:
        show_namespaces(manager)
        return 0
    if args.search:
        show_search(manager, args.search)
        return 0
    if args.profiles:
        show_profiles()
        return 0
    if args.profile_books:
        profile_corpus(manager)
        show_profiles()
        return 0
    if args.effective:
        customer, _, project = args.effective[1].partition("/")
        show_effective(manager, args.effective[0], args.effective[1])
        return 0
    if args.grant or args.unassign:
        if not args.book:
            print(f"{_c('✗ --book is required with --grant/--unassign', Style.RED)}")
            return 2
        if args.grant:
            ok = do_assign(manager, args.grant, args.book, args.scope,
                           args.customer, args.project, args.authorized_by, args.note)
            return 0 if ok else 1
        ok = do_unassign(manager, args.unassign, args.book, args.scope,
                         args.customer, args.project)
        return 0 if ok else 1

    run_repl(manager)
    return 0


if __name__ == "__main__":
    sys.exit(main())
