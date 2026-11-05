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

    def labels(
        self,
        ids: Sequence[Any],
        kept_apart: Sequence[tuple[str, ...]] | None = None,
    ) -> list[str]:
        """Return the label that the figure draws for each clone of ``ids``.

        A clone is labelled by its ID, and a clonotype by its
        representative's junction, with its number where two of ``ids``
        share one, so the clones of one figure are labelled at once.
        ``kept_apart`` gives each clone's labels in the columns that the
        clones keep apart, where a figure draws a clone once for each of
        them, as a clonotype that spans receptor types is drawn once per
        type; a clone drawn with two of them also shows them, such as
        ``CASSA (TCR)`` and ``CASSA (BCR)``.

        Raises:
            TypeError: If the store does not label the clones of figures.
            ValueError: If ``kept_apart`` does not give the labels of each
                clone of ``ids``.
        """
        method = store_method(
            self.store, "_clone_figure_labels", "the figure labels of clones"
        )
        names = [str(i) for i in ids]
        labels = [str(label) for label in method(self.clones, names)]
        if kept_apart is None:
            return labels
        if len(kept_apart) != len(names):
            raise ValueError(
                f"kept_apart must give the labels of each of the {len(names)} "
                f"clones, not {len(kept_apart)}"
            )
        drawn: dict[str, set[tuple[str, ...]]] = {}
        for name, kept in zip(names, kept_apart, strict=True):
            drawn.setdefault(name, set()).add(tuple(kept))
        return [
            f"{label} ({', '.join(kept)})" if len(drawn[name]) > 1 else label
            for label, name, kept in zip(labels, names, kept_apart, strict=True)
        ]

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
