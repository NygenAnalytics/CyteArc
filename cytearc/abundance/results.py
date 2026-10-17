"""Result records of the differential-abundance tests."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metadata.selection import CellField
from ..storage.refs import ArtifactRef
from .propeller import PropellerFit, transformed_proportions

__all__ = [
    "CategoryTestResult",
    "NeighborhoodCounts",
    "NeighborhoodTestResult",
    "Neighborhoods",
]


@dataclass(frozen=True, slots=True, eq=False)
class CategoryTestResult:
    """A propeller test of category proportions between sample groups.

    ``table`` holds one row per category, indexed by category label, with
    ``baseline_prop``, one ``prop_mean_<group>`` column per group,
    ``prop_ratio`` (two groups only), ``t_statistic`` or ``f_statistic``,
    ``p_value``, and the Benjamini-Hochberg ``fdr``. With two groups the
    contrast is ``groups[0]`` minus ``groups[1]``.
    """

    fit: PropellerFit
    categories: ArtifactRef | None
    category_field: CellField | None
    cell_selection: ArtifactRef | None
    sample_by: str
    condition_by: str
    robust_requested: bool
    n_cells: int
    artifact: ArtifactRef | None = None

    @property
    def table(self) -> pd.DataFrame:
        """Per-category statistics."""
        return self.fit.table

    @property
    def method(self) -> str:
        """``"t_test"`` for two groups or ``"anova"`` for more."""
        return self.fit.method

    @property
    def groups(self) -> tuple[str, ...]:
        """Group order; the two-group contrast is the first minus the second."""
        return self.fit.groups

    def counts(self) -> pd.DataFrame:
        """Cells per category (rows) and sample (columns)."""
        return pd.DataFrame(
            self.fit.counts,
            index=pd.Index(self.fit.categories, name="category"),
            columns=pd.Index(self.fit.samples, name=self.sample_by),
        )

    def proportions(self, *, transformed: bool = False) -> pd.DataFrame:
        """Per-sample category proportions, or their transform."""
        proportions, transform = transformed_proportions(
            self.fit.counts, self.fit.transform
        )
        values: np.ndarray = transform if transformed else proportions
        return pd.DataFrame(
            values,
            index=pd.Index(self.fit.categories, name="category"),
            columns=pd.Index(self.fit.samples, name=self.sample_by),
        )

    def sample_groups(self) -> pd.Series:
        """The group of each sample."""
        return pd.Series(
            self.fit.sample_groups,
            index=pd.Index(self.fit.samples, name=self.sample_by),
            name=self.condition_by,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Neighborhoods:
    """Milo neighbourhoods on one kNN graph.

    ``index_rows`` are positions in the graph's ``cell_selection`` and
    ``index_cells`` the matching physical cell rows. ``sizes`` count the
    cells of each neighbourhood, and ``kth_distance`` is the Euclidean
    distance from each index cell to its k-th nearest neighbour.
    """

    artifact: ArtifactRef
    neighbors: ArtifactRef
    cell_selection: ArtifactRef
    index_rows: np.ndarray
    index_cells: np.ndarray
    sizes: np.ndarray
    kth_distance: np.ndarray
    k: int
    n_cells: int
    prop: float
    refined: bool
    seed: int

    @property
    def n_neighborhoods(self) -> int:
        """Number of neighbourhoods."""
        return int(self.index_rows.size)

    def table(self) -> pd.DataFrame:
        """One row per neighbourhood: its index cell, size, and k-distance."""
        return pd.DataFrame(
            {
                "index_cell": self.index_cells,
                "index_row": self.index_rows,
                "size": self.sizes,
                "kth_distance": self.kth_distance,
            },
            index=pd.RangeIndex(self.n_neighborhoods, name="neighborhood"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodCounts:
    """Cells per neighbourhood (rows) and sample (columns).

    Samples are ordered by first appearance along the graph's cells, as
    miloR's ``countCells`` orders a character sample column.
    """

    artifact: ArtifactRef
    neighborhoods: ArtifactRef
    sample_by: str
    samples: tuple[str, ...]
    counts: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """The counts with neighbourhood rows and sample columns."""
        return pd.DataFrame(
            self.counts,
            index=pd.RangeIndex(self.counts.shape[0], name="neighborhood"),
            columns=pd.Index(self.samples, name=self.sample_by),
        )


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodTestResult:
    """A Milo differential-abundance test.

    ``table`` holds one row per neighbourhood with ``index_cell``, ``size``,
    ``log_fc`` (log2), ``log_cpm``, ``f_statistic``, ``p_value``, the
    Benjamini-Hochberg ``fdr``, ``spatial_fdr``, and ``tested``.
    Neighbourhoods below ``min_mean`` are not tested and hold missing
    statistics. ``design`` is the samples x coefficients design matrix and
    ``contrast`` the weights of the tested contrast over its columns.
    """

    counts: ArtifactRef
    neighborhoods: ArtifactRef
    sample_by: str
    condition_by: str
    covariates: tuple[str, ...]
    design: pd.DataFrame
    contrast: dict[str, float]
    table: pd.DataFrame
    norm_factors: pd.Series
    lib_size: pd.Series
    common_dispersion: float
    average_ql_dispersion: float | None
    norm_method: str
    min_mean: float
    robust: bool
    legacy: bool
    fdr_weighting: str
    artifact: ArtifactRef | None = None
