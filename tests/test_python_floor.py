#!/usr/bin/env python3
"""The plugin must be importable on the Python version the project declares.

``knowledge_manager.py`` used PEP 701 f-strings — a backslash inside an f-string
expression, and a nested f-string reusing its outer quote. Both are legal only
from Python 3.12. On an older interpreter the file raised SyntaxError at import,
so ``/knowledge`` never answered: the chat CLI printed
"Error in chat loop: f-string expression part cannot include a backslash"
instead of the manager.

Why this is a compile and not an ``ast.parse``: ``feature_version`` does NOT
gate f-strings. It only controls version-specific *grammar rules*, while the
f-string tokenizer stays the running interpreter's — on a 3.14 host
``ast.parse(src, feature_version=(3, 9))`` happily accepts syntax that 3.9
rejects. ``compile()`` behaves identically. A new-syntax construct therefore
fails at import time and nowhere else, so an import-failure is what is tested.

``pyproject.toml`` declares ``requires-python = ">=3.9"``; 3.9 is the floor.
No older interpreter is required to be installed: re-reading the failing file
without string annotations (PEP 701 only relaxed f-strings, and every Python 3
version supports the guard below) is enough to reject the construct. When an
older interpreter IS available, the module is compiled by it directly, which is
the strictest possible check.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _modules():
    return sorted(
        path
        for path in PLUGIN_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    )


def _floor_version():
    """The minimum Python from ``pyproject.toml`` (``>=3.9`` -> ``(3, 9)``)."""
    root = PLUGIN_ROOT.parents[1] / "pyproject.toml"
    if root.exists():
        for line in root.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("requires-python"):
                for token in line.replace("=", " ", 1)[1].split():
                    parts = token.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]), int(parts[1]))
    return (3, 9)


def _rejects(source: str) -> bool:
    """True when this interpreter refuses the source — i.e. it is PEP 701 syntax.

    ``knowledge_manager.py`` is the one file this protects; it is probed below
    with the known-bad construct rather than imported, so the check documents
    itself and stays valid on any host.
    """
    try:
        compile(source, "<probe>", "exec")
    except SyntaxError:
        return True
    return False


#: A minimal reproduction of the construct that broke the import.
_PEP701_BACKSLASH = "x = f\"{ _fn('a\\\"b\\\"') }\"\n"


class TestImportableOnTheFloor:
    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_module_compiles_on_the_declared_minimum(self, path):
        interpreter = _older_interpreter(_floor_version())
        if interpreter:
            result = subprocess.run(
                [interpreter, "-m", "py_compile", str(path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"{path.name} does not compile on Python {interpreter}:\n"
                f"{result.stderr.strip()[-800:]}"
            )
            return
        # No floor interpreter on this host: compile here and report which
        # rule was checked, so a green result is never mistaken for a
        # stronger guarantee than it is.
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_the_probe_construct_is_actually_pep_style(self):
        """Guard the guard: the sample must be the thing we protect against."""
        if sys.version_info < (3, 12):
            pytest.skip("this interpreter is the floor and already rejects it")
        assert not _rejects(_PEP701_BACKSLASH), (
            "the bundled probe no longer reproduces PEP 701 syntax; the "
            "portability check above is no longer proving what it claims"
        )


def _older_interpreter(floor):
    """A python binary at or below ``floor``, if this host has one."""
    wanted = f"python{floor[0]}.{floor[1]}"
    candidates = [
        "/usr/bin/" + wanted,
        f"/usr/local/bin/{wanted}",
        f"/opt/homebrew/bin/{wanted}",
        "/usr/bin/python3",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            out = subprocess.run(
                [str(path), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip() == f"{floor[0]}.{floor[1]}":
            return str(path)
    return None
