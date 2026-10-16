from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...matrix import ChunkedArray
from ...storage.refs import ArtifactRef
from ...utils.arrays import has_duplicates

__all__ = ["EnrichmentResult"]


@dataclass(frozen=True, slots=True, eq=False)
class EnrichmentResult:
    """A persisted gene-set or peak-set enrichment result.

    Attributes:
        data: Lazy score matrix with cells as rows and sources as columns.
            chromVAR stores z-scores here, with NaN where a cell has too few
            expected fragments in a set.
        source_names: Source names aligned to the score columns.
        source_sizes: Matched target counts aligned to ``source_names``.
        cell_index: Assay cell indices aligned to the score rows.
        artifact: Immutable enrichment artifact backing the result.
        storage_path: Zarr path that owns the result.
        assay: Name of the assay used for scoring: an RNA assay for WAGGR and
            AUCell, an ATAC assay for chromVAR.
        cell_selection: Immutable cell selection used for scoring.
        feature_selection: Feature-selection artifact used for scoring.
        method: Enrichment method: ``"waggr"``, ``"aucell"``, or ``"chromvar"``.
        deviations: chromVAR's bias-corrected deviations, aligned to ``data``;
            None for other methods.
    """

    data: ChunkedArray = field(repr=False)
    source_names: np.ndarray = field(repr=False)
    source_sizes: np.ndarray = field(repr=False)
    cell_index: np.ndarray = field(repr=False)
    artifact: ArtifactRef
    storage_path: str
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    method: str
    deviations: ChunkedArray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            self.artifact.kind != "enrichment_scores"
            or self.artifact.scope != "assay"
            or self.artifact.assay != self.assay
        ):
            raise ValueError("Enrichment artifact must belong to the result assay")
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Enrichment feature_selection must belong to the result assay"
            )
        if (
            not isinstance(self.cell_selection, ArtifactRef)
            or self.cell_selection.kind != "cell_selection"
            or self.cell_selection.scope != "datastore"
            or self.cell_selection.assay is not None
        ):
            raise ValueError(
                "Enrichment cell_selection must be a datastore cell selection"
            )
        if len(self.data.shape) != 2:
            raise ValueError("Enrichment data must be two-dimensional")
        if (
            self.source_names.ndim != 1
            or self.source_sizes.ndim != 1
            or self.cell_index.ndim != 1
        ):
            raise ValueError("Enrichment metadata arrays must be one-dimensional")
        if self.data.shape[0] != len(self.cell_index):
            raise ValueError("Enrichment cell indices do not align with score rows")
        if self.data.shape[1] != len(self.source_names):
            raise ValueError("Enrichment source names do not align with score columns")
        if len(self.source_names) != len(self.source_sizes):
            raise ValueError("Enrichment source names and sizes must be aligned")
        if has_duplicates(self.source_names):
            raise ValueError("Enrichment source names must be unique")
        if has_duplicates(self.cell_index):
            raise ValueError("Enrichment cell indices must be unique")
        if np.any(self.source_sizes <= 0):
            raise ValueError("Enrichment source sizes must be positive")
        if self.method not in {"waggr", "aucell", "chromvar"}:
            raise ValueError(f"Unknown enrichment method: {self.method!r}")
        if (self.deviations is None) != (self.method != "chromvar"):
            raise ValueError("chromVAR results, and only they, carry deviations")
        if self.deviations is not None and self.deviations.shape != self.data.shape:
            raise ValueError("Enrichment deviations must align with the scores")

    def to_pandas(self) -> pd.DataFrame:
        """Read the scores as a cells-by-sources DataFrame.

        The columns are the source names, and the index holds the assay cell
        indices in the order of the score rows.
        """
        return pd.DataFrame(
            np.asarray(self.data.compute()),
            index=pd.Index(self.cell_index, name="cell_index"),
            columns=self.source_names,
        )
