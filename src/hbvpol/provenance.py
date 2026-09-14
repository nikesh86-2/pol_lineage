"""Run provenance recorded in every stage summary.

A publication run must be able to state which build of each external tool (and of
``hbvpol`` itself) produced a result.  Tool versions are probed lazily and cached
for the process; a missing tool yields ``None`` rather than failing the stage, and
a probe that hangs or errors is treated the same way.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from functools import cache
from shutil import which

from . import __version__

__all__ = ["provenance"]

#: Version flags tried in order; the first plausible result wins.
_VERSION_FLAGS = ("--version", "-version", "version", "-v")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
#: Output that is clearly not a version string (e.g. a tool treating the flag as
#: an input path) is rejected rather than recorded.
_REJECT_PREFIXES = ("error", "usage", "unknown", "fatal")
_MIN_VERSION_LEN = 3


def _first_line(text: str) -> str:
    text = _ANSI.sub("", text or "").strip()
    return text.splitlines()[0].strip() if text else ""


@cache
def tool_version(executable: str) -> str | None:
    """First plausible version line of ``executable``, or ``None`` if unavailable."""
    path = which(executable)
    if path is None:
        return None
    for flag in _VERSION_FLAGS:
        try:
            result = subprocess.run(
                [path, flag], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            continue
        line = _first_line(result.stdout or result.stderr or "")
        if len(line) < _MIN_VERSION_LEN or line.lower().startswith(_REJECT_PREFIXES):
            continue
        # Reject decorative banners (box-drawing rules and the like).
        if sum(1 for char in line if char.isascii() and char.isalnum()) < 3:
            continue
        return line[:200]
    return None


def provenance(tools: Iterable[str] | None = None) -> dict:
    """``hbvpol`` version plus the probed versions of the named tools."""
    return {
        "hbvpol": __version__,
        "tools": {str(name): tool_version(str(name)) for name in (tools or ())},
    }
