"""Public CyteBase releases, each published in its own year-month bucket."""

from dataclasses import dataclass

from ._storage import versioned_bucket


@dataclass(frozen=True, slots=True)
class Release:
    """One public CyteBase release and the bucket that holds it.

    ``version`` is ``YYMM`` or ``YYMM.N``, ``released`` is the ``YYYY-MM``
    month of release, ``notes`` summarizes the release, and ``changelog``
    lists its changes.
    """

    version: str
    released: str
    notes: str
    changelog: tuple[str, ...] = ()

    @property
    def bucket(self) -> str:
        """Return the ``namespace/name`` of this release's bucket."""
        return versioned_bucket(self.version)

    @property
    def latest(self) -> bool:
        """Return whether ``Catalog()`` opens this release by default."""
        return self.version == LATEST


# List each new release first, and point LATEST at it once its catalog is
# published. Tests check the versions, order, months, and notes.
LATEST = "2610"
RELEASES: tuple[Release, ...] = (
    Release(
        version="2610",
        released="2026-10",
        notes=(
            "October 2026 release. First versioned CyteBase build, from a "
            "CELLxGENE Discover snapshot taken 2026-10-10."
        ),
        changelog=(
            "Primary RNA datasets from 376 registration-ready collections, "
            "converted to CyteArc stores.",
        ),
    ),
)


def releases() -> tuple[Release, ...]:
    """Return every public CyteBase release, newest first.

    Each release names its ``version``, its ``bucket``, the ``released``
    month, ``notes``, and a ``changelog``. ``latest`` marks the release that
    ``Catalog()`` opens when neither a bucket nor a version is given; open
    another with ``Catalog(version=...)``.
    """
    return RELEASES
