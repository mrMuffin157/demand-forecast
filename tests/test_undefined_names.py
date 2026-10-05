"""No pipeline module may read a name it never defines.

The motivating bug: a refactor of `forecast.py` dropped the line that set a variable that
only the POOLED serving path still read; no test exercised that path, so the forecast job
died with a NameError at run time. This is a pyflakes-style check using only the standard
library (`symtable`), so it also runs where no linter is installed: every global a
function reads must be defined at module level or be a builtin."""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([*ROOT.glob("dev/*.py"), *ROOT.glob("src/demandml/*.py"),
                *ROOT.glob("model_lab/*.py")])


def _undefined(path: Path) -> list[str]:
    top = symtable.symtable(path.read_text(), str(path), "exec")
    defined = set(top.get_identifiers()) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    bad = []

    def walk(t):
        for c in t.get_children():
            if c.get_type() == "function":
                for s in c.get_symbols():
                    if s.is_global() and s.is_referenced() and s.get_name() not in defined:
                        bad.append(f"{c.get_name()}(): {s.get_name()}")
            walk(c)
    walk(top)
    return bad


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_function_reads_an_undefined_global(path):
    assert not _undefined(path), f"{path.name}: undefined names {_undefined(path)}"
