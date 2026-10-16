"""Earlier name of CyteArc's example-data downloads, kept through CyteArc 1.x.

``Repository``, ``connect``, and ``list_repositories`` are the
``cytearc.datasets`` functions under their earlier path. The CyteBase catalog
SDK is the separate ``cytebase`` package, which CyteArc does not install: run
``pip install cytebase``, then ``import cytebase``. CyteArc opens CyteBase
datasets by their ``hf://`` URIs like any other remote store.
"""

from typing import NoReturn

from .datasets import Repository, connect, list_repositories

__all__ = ["Repository", "connect", "list_repositories"]

_MOVED_TO_CYTEBASE = frozenset(
    {
        "Catalog",
        "DatasetEntry",
        "Release",
        "embedding",
        "embedding_coordinates",
        "embeddings",
        "releases",
    }
)


def __getattr__(name: str) -> NoReturn:
    if name in _MOVED_TO_CYTEBASE:
        raise AttributeError(
            f"cytearc.cytebase.{name} moved to the separate cytebase package. "
            f"Install it with 'pip install cytebase' and use cytebase.{name}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
