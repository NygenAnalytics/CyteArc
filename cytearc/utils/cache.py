"""The directories where CyteArc keeps files that it downloads once and reuses."""

import os
from pathlib import Path, PurePath

__all__ = ["cache_directory"]


def cache_directory(name: str) -> Path:
    """Return ``cytearc/<name>`` under the user's cache root, without creating it.

    The cache root is ``$XDG_CACHE_HOME``, with ``~`` expanded, or ``~/.cache``
    when the variable is unset or empty.

    Raises:
        ValueError: If ``name`` is not a single directory name.
    """
    if not name or PurePath(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"cache directory name must be one path component, got {name!r}"
        )
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "cytearc" / name
