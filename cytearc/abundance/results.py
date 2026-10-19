"""Result records of differential-abundance tests on a datastore.

``DataStore.abundance`` returns these records and keeps each as an immutable
artifact. They carry the references of the artifacts they were computed
from, so a result can be traced back to its cells, graph, and samples.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metadata.selection import CellField
from ..storage.artifacts import ArtifactRef
from .propeller import PropellerFit

__all__ = [
    "CategoryTestResult",
    "NeighborhoodCounts",
    "NeighborhoodTestResult",
    "Neighborhoods",
]


@dataclass(frozen=True, slots=True, eq=False)
class CategoryTestResult:
    """A propeller test of category proportions (Phipson et al. 2022).

    ``fit`` holds the counts, proportions, and statistics; its ``table`` has
    one row per category, described in ``cytearc.abundance.propeller``.
    The categories come from the ``categories`` artifact or the
    ``category_field`` column, over the cells of ``cell_selection`` (all
    cells when None). ``sample_by`` and ``condition_by`` name the sample and
    condition columns. ``robust_requested`` is the requested robust setting;
    ``fit.robust`` is the one used, which is off for two or fewer
    categories. ``artifact`` is None when the result was not saved.
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
        """One row per category with proportions, statistics, and FDR."""
        return self.fit.table

    @property
    def method(self) -> str:
        """``"t_test"`` for two groups, ``"anova"`` for more."""
        return self.fit.method

    @property
    def groups(self) -> tuple[str, ...]:
        """The tested conditions, in order; t-statistics are the first minus the second."""
        return self.fit.groups

    def _frame(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            values,
            index=pd.Index(self.fit.categories, name="category"),
            columns=pd.Index(self.fit.samples, name=self.sample_by),
        )

    def counts(self) -> pd.DataFrame:
        """Cells per category (rows) and sample (columns)."""
        return self._frame(self.fit.counts)

    def proportions(self, *, transformed: bool = False) -> pd.DataFrame:
        """Each sample's category proportions, or their logit or arcsine transform."""
        return self._frame(
            self.fit.transformed if transformed else self.fit.proportions
        )

    def sample_groups(self) -> pd.Series:
        """The condition of each tested sample."""
        return pd.Series(
            self.fit.sample_groups,
            index=pd.Index(self.fit.samples, name=self.sample_by),
            name=self.condition_by,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Neighborhoods:
    """Milo neighbourhoods on a stored kNN graph (Dann et al. 2022).

    Neighbourhood ``j`` is the order-1 neighbourhood of its index cell on
    the undirected graph of ``neighbors``. ``index_rows`` are the index
    cells' positions in the graph's ``cell_selection`` and ``index_cells``
    their physical cell rows. ``sizes`` counts each neighbourhood's cells
    and ``kth_distance`` is the Euclidean distance from its index cell to
    its k-th nearest neighbour. ``prop`` of the graph's ``n_cells`` cells
    were drawn with ``seed`` and, when ``refined``, moved to the cell
    nearest the median position of their neighbours.
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
        """The number of neighbourhoods."""
        return int(np.asarray(self.index_rows).size)

    def table(self) -> pd.DataFrame:
        """One row per neighbourhood: index row, index cell, size, and k-distance."""
        return pd.DataFrame(
            {
                "index_row": np.asarray(self.index_rows, dtype=np.int64),
                "index_cell": np.asarray(self.index_cells, dtype=np.int64),
                "size": np.asarray(self.sizes, dtype=np.int64),
                "kth_distance": np.asarray(self.kth_distance, dtype=np.float64),
            },
            index=pd.RangeIndex(self.n_neighborhoods, name="neighborhood"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodCounts:
    """Cells per neighbourhood (rows) and sample (columns).

    ``samples`` are the labels of the ``sample_by`` column, in order of
    first appearance along the graph's cells.
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
    """A Milo test of neighbourhood counts between conditions.

    ``table`` has one row per neighbourhood with ``index_cell``, ``size``,
    the log2 fold change ``log_fc``, ``log_cpm``, the quasi-likelihood
    ``f_statistic`` and ``p_value``, the Benjamini-Hochberg ``fdr``, the
    ``spatial_fdr`` of Lun, Richard and Marioni (2017), and ``tested``;
    neighbourhoods below ``min_mean`` are not tested and have missing
    statistics. ``design`` is the samples x columns design, with
    ``covariates`` then ``condition_by``, and ``contrast`` the tested
    weights over its columns. ``norm_factors`` and ``lib_size`` hold one
    value per sample. ``artifact`` is None when the result was not saved.
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
