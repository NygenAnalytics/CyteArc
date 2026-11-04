"""What figures call the clones of a clone artifact, and how they label them."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..storage.refs import ArtifactRef
from ._data import store_method

__all__ = ["CloneNames", "clone_names"]


@dataclass(frozen=True, slots=True)
class CloneNames:
    """What one figure calls a clone artifact's clones, and how it labels them.

    Every title, axis and legend of a figure that names clones takes these
    nouns, so a figure of clonotypes names clonotypes.

    Attributes:
        plural: ``"clones"``, ``"clonotypes"`` or ``"clonotype clusters"``.
        singular: ``"clone"``, ``"clonotype"`` or ``"clonotype cluster"``.
        clones: The clone artifact.
    """

    plural: str
    singular: str
    clones: ArtifactRef
    store: Any = field(repr=False, compare=False)

    def labels(self, ids: Sequence[Any]) -> list[str]:
        """Return the label that the figure draws for each clone of ``ids``.

        A clone is labelled by its ID, and a clonotype by its
        representative's junction, with its number where two of ``ids``
        share one, so the clones of one figure are labelled at once.

        Raises:
            TypeError: If the store does not label the clones of figures.
        """
        method = store_method(
            self.store, "_clone_figure_labels", "the figure labels of clones"
        )
        return [str(label) for label in method(self.clones, [str(i) for i in ids])]

    def largest(self, drawn: int) -> str:
        """Title the largest clones drawn: ``"Largest 20 clonotypes"``, or ``"Largest clonotype"``."""
        return (
            f"Largest {self.singular}"
            if drawn == 1
            else f"Largest {drawn} {self.plural}"
        )


def clone_names(store: Any, clones: ArtifactRef) -> CloneNames:
    """Return what figures call the clones of ``clones``, from the store.

    Raises:
        TypeError: If the store does not say what figures call clones, or
            ``clones`` is not an ArtifactRef.
        ValueError: If ``clones`` is not a clone artifact.
        KeyError: If the store holds no such artifact.
        RuntimeError: If the artifact is incomplete.
    """
    plural, singular = store_method(
        store, "_clone_figure_nouns", "what figures call clones"
    )(clones)
    return CloneNames(str(plural), str(singular), clones, store)
