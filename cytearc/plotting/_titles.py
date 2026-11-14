"""Titles of the label artifacts that plots group or color cells by.

A caller names a label artifact with ``NamedCellArtifact(name, artifact)``,
as ``CellField(key, label=...)`` names a live column. A bare ``ArtifactRef``
takes a title from the artifact's own record: the ``obs`` key that an H5AD
import read, or the column that a snapshot or label transfer copied; the
method and resolution of a Leiden clustering; or else its kind in words, such
as "Cluster labels". The title only labels the figure: summaries, their
groupings and artifact identities never see it.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..metadata.selection import NamedCellArtifact
from ..storage.artifacts import inspect_artifact
from ..storage.refs import ArtifactRef

__all__ = [
    "LabelArtifact",
    "artifact_title",
    "artifact_titles",
    "kind_title",
    "label_artifact",
]

type LabelArtifact = ArtifactRef | NamedCellArtifact

# The parameters in which an artifact records the column or key it copied.
_COPIED_COLUMN_PARAMETERS = ("source_key", "source_column")
# Kinds whose words need more than their first letter capitalized.
_KIND_TITLES = {"hto_identity": "HTO identity"}


def label_artifact(
    value: LabelArtifact | None, argument: str
) -> tuple[ArtifactRef | None, str | None]:
    """Return the artifact of a plot argument and the name its caller gave it.

    Raises:
        TypeError: If ``value`` is neither an ``ArtifactRef``, a
            ``NamedCellArtifact``, nor None.
    """
    if value is None or isinstance(value, ArtifactRef):
        return value, None
    if isinstance(value, NamedCellArtifact):
        return value.artifact, value.name
    raise TypeError(
        f"{argument} must be an ArtifactRef or a NamedCellArtifact, "
        f"not {type(value).__name__}"
    )


def kind_title(kind: str) -> str:
    """An artifact kind in words, such as "Cluster labels" for ``cluster_labels``."""
    return _KIND_TITLES.get(kind, kind.replace("_", " ").capitalize())


def artifact_title(store: Any, artifact: ArtifactRef) -> str:
    """Return the title that a plot gives a label artifact without a name.

    The title comes from the artifact's record: the ``obs`` key that an H5AD
    import read, or the column that a snapshot or a label transfer copied;
    for a Leiden clustering, its method and resolution, such as "Leiden,
    resolution 1.0"; and otherwise its kind in words. A store that holds no
    record of the artifact, such as a plot's test double, gives the kind.
    """
    root = getattr(store, "zw", None)
    status = None if root is None else inspect_artifact(root, artifact)
    parameters = {} if status is None else (status.parameters or {})
    for key in _COPIED_COLUMN_PARAMETERS:
        copied = parameters.get(key)
        if isinstance(copied, str) and copied:
            return copied
    resolution = parameters.get("resolution")
    if (
        status is not None
        and status.operation == "run_leiden_clustering"
        and isinstance(resolution, int | float)
        and not isinstance(resolution, bool)
    ):
        return f"Leiden, resolution {float(resolution)}"
    return kind_title(artifact.kind)


def artifact_titles(
    store: Any,
    artifacts: Sequence[tuple[ArtifactRef, str | None]],
) -> list[str]:
    """Return the title of each of a figure's label artifacts, each with its caller's name or None.

    A name is the title as given. Distinct artifacts whose records give one
    title take the start of their artifact IDs after it, such as "Leiden,
    resolution 1.0 (3f9a2c)", so a figure tells them apart; an artifact
    drawn twice keeps one title.
    """
    defaults = {
        artifact: artifact_title(store, artifact)
        for artifact, name in artifacts
        if name is None
    }
    shared = Counter(defaults.values())
    titles = []
    for artifact, name in artifacts:
        if name is not None:
            titles.append(name)
            continue
        title = defaults[artifact]
        titles.append(
            title if shared[title] == 1 else f"{title} ({artifact.artifact_id[:6]})"
        )
    return titles
