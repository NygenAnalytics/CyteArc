"""Records of persisted regulon results."""

from dataclasses import dataclass, field

import pandas as pd

from ..storage.refs import ArtifactRef

__all__ = ["GrnResult"]

_ADJACENCY_COLUMNS = ("TF", "target", "importance", "correlation")


@dataclass(frozen=True, slots=True, eq=False)
class GrnResult:
    """A persisted transcription-factor to target importance network.

    Attributes:
        adjacencies: One row per link with ``TF``, ``target``, ``importance``,
            and ``correlation`` columns, sorted by decreasing importance.
            ``correlation`` is the Pearson correlation of the factor and the
            target over the scored cells, which pySCENIC uses to separate
            activating from repressing links.
        regulators: Names of the factors offered to every target model.
        artifact: Immutable ``grn_adjacencies`` artifact backing the result.
        assay: Name of the RNA assay whose counts were modelled.
        cell_selection: Cells whose counts were modelled.
        feature_selection: Genes that were modelled as targets and regulators.
        method: Inference method, ``"grnboost2"``.
        random_seed: Random state of every target's regressor.
    """

    adjacencies: pd.DataFrame = field(repr=False)
    regulators: tuple[str, ...] = field(repr=False)
    artifact: ArtifactRef
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    method: str
    random_seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, ArtifactRef)
            or self.artifact.kind != "grn_adjacencies"
            or self.artifact.scope != "assay"
            or self.artifact.assay != self.assay
        ):
            raise ValueError("GRN artifact must belong to the result assay")
        if tuple(self.adjacencies.columns) != _ADJACENCY_COLUMNS:
            raise ValueError(
                "GRN adjacencies must have TF, target, importance, and correlation columns"
            )
        if self.method != "grnboost2":
            raise ValueError(f"Unknown GRN method: {self.method!r}")
        if not self.regulators:
            raise ValueError("A GRN result needs at least one regulator")
